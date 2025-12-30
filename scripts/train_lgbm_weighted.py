import argparse
import gc
from dataclasses import dataclass
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class Config:
    train_path: str
    test_path: str
    target_col: str
    id_col: str
    n_splits: int
    seed: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    early_stopping_rounds: int
    n_jobs: int
    use_oof_target_encoding: bool
    te_smooth: float
    out_csv: str
    out_oof_artifact: str | None
    artifact_name: str
    limit_train_rows: int | None
    limit_test_rows: int | None


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train weighted LightGBM model; optionally dump OOF artifact.")
    p.add_argument("--train", dest="train_path", default="data/train.csv")
    p.add_argument("--test", dest="test_path", default="data/test.csv")
    p.add_argument("--target-col", default="diagnosed_diabetes")
    p.add_argument("--id-col", default="id")

    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=4)

    p.add_argument(
        "--use-oof-target-encoding",
        action="store_true",
        help="Leakage-safe OOF target encoding for object columns (replaces label encoding).",
    )
    p.add_argument("--te-smooth", type=float, default=10.0, help="Smoothing strength for target encoding.")

    p.add_argument("--out", dest="out_csv", default="submission_weighted_lgbm.csv")
    p.add_argument(
        "--oof-artifact",
        dest="out_oof_artifact",
        default=None,
        help="If set, write a standardized .npz artifact for OOF blending.",
    )
    p.add_argument("--name", dest="artifact_name", default="weighted_lgbm")
    p.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help="Optional: read only first N training rows (for smoke tests).",
    )
    p.add_argument(
        "--limit-test-rows",
        type=int,
        default=None,
        help="Optional: read only first N test rows (for smoke tests).",
    )

    a = p.parse_args()
    return Config(
        train_path=a.train_path,
        test_path=a.test_path,
        target_col=a.target_col,
        id_col=a.id_col,
        n_splits=a.n_splits,
        seed=a.seed,
        n_estimators=a.n_estimators,
        learning_rate=a.learning_rate,
        num_leaves=a.num_leaves,
        early_stopping_rounds=a.early_stopping_rounds,
        n_jobs=a.n_jobs,
        use_oof_target_encoding=bool(a.use_oof_target_encoding),
        te_smooth=float(a.te_smooth),
        out_csv=a.out_csv,
        out_oof_artifact=a.out_oof_artifact,
        artifact_name=a.artifact_name,
        limit_train_rows=a.limit_train_rows,
        limit_test_rows=a.limit_test_rows,
    )


def _oof_target_encode(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: StratifiedKFold,
    *,
    smooth: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF target mean encoding for object columns.

    Returns encoded (X_enc, X_test_enc). For each categorical column `c`, creates
    a numeric column `c__te` and drops the original.
    """

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if not cat_cols:
        return X.copy(), X_test.copy()

    X_base = X.copy()
    X_test_base = X_test.copy()
    for c in cat_cols:
        X_base[c] = X_base[c].astype(str).fillna("__NA__")
        X_test_base[c] = X_test_base[c].astype(str).fillna("__NA__")

    global_mean = float(np.mean(y))

    X_enc = X_base.drop(columns=cat_cols)
    X_test_enc = X_test_base.drop(columns=cat_cols)

    for c in cat_cols:
        out_col = f"{c}__te"
        oof_col = np.empty(len(X_base), dtype=np.float64)

        for tr_idx, va_idx in folds.split(X_base, y):
            s = X_base.iloc[tr_idx][c]
            y_tr = y[tr_idx]
            stats = pd.DataFrame({"key": s.values, "y": y_tr}).groupby("key")["y"].agg(["mean", "count"])
            enc_map = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
            oof_col[va_idx] = X_base.iloc[va_idx][c].map(enc_map).fillna(global_mean).astype(np.float64).values

        stats_full = pd.DataFrame({"key": X_base[c].values, "y": y}).groupby("key")["y"].agg(["mean", "count"])
        enc_map_full = (stats_full["mean"] * stats_full["count"] + global_mean * smooth) / (stats_full["count"] + smooth)
        test_col = X_test_base[c].map(enc_map_full).fillna(global_mean).astype(np.float64).values

        X_enc[out_col] = oof_col
        X_test_enc[out_col] = test_col

    return X_enc, X_test_enc


def train_weighted_model(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    print("Loading data...")
    train = pd.read_csv(cfg.train_path, nrows=cfg.limit_train_rows)
    test = pd.read_csv(cfg.test_path, nrows=cfg.limit_test_rows)

    y = train[cfg.target_col].astype(int).to_numpy()
    train_ids = train[cfg.id_col].astype(np.int64).to_numpy()
    test_ids = test[cfg.id_col].astype(np.int64).to_numpy()

    # --- Step 1: Adversarial Validation / Weight Calculation ---
    print("\n--- Step 1: Calculating Adversarial Weights ---")

    train_adv = train.drop(columns=[cfg.target_col, cfg.id_col]).copy()
    test_adv = test.drop(columns=[cfg.id_col]).copy()

    train_adv["is_test"] = 0
    test_adv["is_test"] = 1

    adv_data = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
    y_adv = adv_data["is_test"].to_numpy()
    X_adv = adv_data.drop(columns=["is_test"])

    # Encode categoricals for the adversarial model.
    for c in X_adv.columns:
        if X_adv[c].dtype == "object":
            le = LabelEncoder()
            X_adv[c] = le.fit_transform(X_adv[c].astype(str))

    adv_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )

    adv_model.fit(X_adv, y_adv)

    X_train_encoded = X_adv.iloc[: len(train)]
    p_test = adv_model.predict_proba(X_train_encoded)[:, 1]

    p_test = np.clip(p_test, 1e-5, 1 - 1e-5)
    weights = p_test / (1 - p_test)
    weights = np.clip(weights, 0, 10.0)
    weights = weights / weights.mean()

    print(f"Weights calculated. Mean: {weights.mean():.4f}, Max: {weights.max():.4f}, Min: {weights.min():.4f}")

    del train_adv, test_adv, adv_data, X_adv, y_adv, adv_model, X_train_encoded
    gc.collect()

    # --- Step 2: Weighted Training ---
    print("\n--- Step 2: Training Weighted Model ---")

    X = train.drop(columns=[cfg.target_col, cfg.id_col])
    X_test = test.drop(columns=[cfg.id_col])

    del train, test
    gc.collect()

    if cfg.use_oof_target_encoding:
        skf_for_te = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        X, X_test = _oof_target_encode(X, y, X_test, skf_for_te, smooth=cfg.te_smooth)
    else:
        for c in X.columns:
            if X[c].dtype == "object":
                combined = pd.concat([X[c], X_test[c]], axis=0).astype(str)
                le = LabelEncoder()
                le.fit(combined)
                X[c] = le.transform(X[c].astype(str))
                X_test[c] = le.transform(X_test[c].astype(str))
                del combined
                gc.collect()

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    oof_preds = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)

    X_vals = X.values
    X_test_vals = X_test.values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = weights[tr_idx]

        clf = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )

        clf.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds)],
        )

        oof_preds[va_idx] = clf.predict_proba(X_va)[:, 1]
        test_preds += clf.predict_proba(X_test_vals)[:, 1] / cfg.n_splits

        auc = roc_auc_score(y_va, oof_preds[va_idx])
        print(f"Fold {fold} AUC: {auc:.5f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Weighted CV AUC: {overall_auc:.5f}")

    sub = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": test_preds})

    out_dir = os.path.dirname(cfg.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(cfg.out_csv, index=False)
    print(f"Submission saved to {cfg.out_csv}")

    if cfg.out_oof_artifact:
        art_dir = os.path.dirname(cfg.out_oof_artifact)
        if art_dir:
            os.makedirs(art_dir, exist_ok=True)
        np.savez(
            cfg.out_oof_artifact,
            name=np.array(cfg.artifact_name, dtype=object),
            train_id=train_ids,
            y=y.astype(np.int64),
            oof=oof_preds.astype(np.float64),
            test_id=test_ids,
            test=test_preds.astype(np.float64),
        )
        print(f"OOF artifact saved to {cfg.out_oof_artifact}")

    return train_ids, y, test_ids, test_preds, float(overall_auc)


if __name__ == "__main__":
    cfg = _parse_args()
    train_weighted_model(cfg)
