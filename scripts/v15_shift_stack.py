import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostClassifier


def _dump_pred_npz(
    *,
    path: str,
    name: str,
    train_id: np.ndarray,
    y: np.ndarray,
    oof: np.ndarray,
    test_id: np.ndarray,
    test: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array(name, dtype=object),
        train_id=train_id.astype(np.int64),
        y=y.astype(np.int64),
        oof=oof.astype(np.float64),
        test_id=test_id.astype(np.int64),
        test=test.astype(np.float64),
    )


TARGET = "diagnosed_diabetes"
ID_COL = "id"


@dataclass
class Config:
    data_dir: str = "data"
    orig_csv: str = "data/orig2/diabetes_dataset.csv"
    n_splits: int = 5
    seed: int = 42
    out: str = "submission_v15_shift_stack.csv"

    # Domain classifier
    domain_num_leaves: int = 63
    domain_lr: float = 0.05
    domain_n_estimators: int = 400

    # Base models
    lgb_lr: float = 0.03
    lgb_num_leaves: int = 127
    lgb_min_data_in_leaf: int = 80
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    lgb_bagging_freq: int = 1
    lgb_lambda_l2: float = 2.0
    lgb_n_estimators: int = 4000
    lgb_early_stopping: int = 200

    cat_depth: int = 5
    cat_lr: float = 0.03
    cat_n_estimators: int = 6000
    cat_early_stopping: int = 250


def _read_data(cfg: Config):
    train = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))
    return train, test, sample


def _add_orig_mean_count_features(train: pd.DataFrame, test: pd.DataFrame, orig_csv: str):
    """Adds orig_mean_* and orig_count_* computed from external orig dataset.

    This is intentionally simple + leakage-safe: orig stats come only from external data.
    """
    orig = pd.read_csv(orig_csv)
    # Some orig datasets may not have 'id' and may have target column.
    # Keep only overlapping feature columns; exclude target + domain feature.
    candidate_cols = [c for c in test.columns if c not in {TARGET, "p_test"}]

    if TARGET not in orig.columns:
        raise SystemExit(f"orig_csv missing target column '{TARGET}': {orig_csv}")

    feature_cols = [c for c in candidate_cols if c in orig.columns]
    if not feature_cols:
        raise SystemExit("No overlapping feature columns found between test and orig dataset")

    # Grouping on raw float values is extremely high-cardinality and unstable.
    # For numeric columns, we bin (quantiles) and compute stats on the bin.
    numeric_cols = orig[feature_cols].select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    def _bin_series(ref: pd.Series, s: pd.Series, n_bins: int = 100) -> pd.Series:
        # Use quantile-based bins from orig to avoid leaking labels.
        # This avoids grouping on raw float values (too sparse).
        ref2 = ref.dropna()
        if ref2.empty:
            return pd.Series(pd.NA, index=s.index)

        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(ref2.to_numpy(), qs))
        if len(edges) < 3:
            # Not enough unique values to bin meaningfully
            return s

        # pd.cut expects monotonically increasing edges
        return pd.cut(s, bins=edges, include_lowest=True)

    for col in categorical_cols:
        means = orig.groupby(col, dropna=False)[TARGET].mean()
        means.name = f"orig_mean_{col}"
        train = train.merge(means, how="left", on=col)
        test = test.merge(means, how="left", on=col)

        cnt = orig.groupby(col, dropna=False).size().reset_index(name=f"orig_count_{col}")
        train = train.merge(cnt, how="left", on=col)
        test = test.merge(cnt, how="left", on=col)

    for col in numeric_cols:
        bin_name = f"__bin_{col}"
        orig[bin_name] = _bin_series(orig[col], orig[col])
        train[bin_name] = _bin_series(orig[col], train[col])
        test[bin_name] = _bin_series(orig[col], test[col])

        means = orig.groupby(bin_name, dropna=False)[TARGET].mean()
        means.name = f"orig_mean_bin_{col}"
        train = train.merge(means, how="left", left_on=bin_name, right_index=True)
        test = test.merge(means, how="left", left_on=bin_name, right_index=True)

        cnt = orig.groupby(bin_name, dropna=False).size()
        cnt.name = f"orig_count_bin_{col}"
        train = train.merge(cnt, how="left", left_on=bin_name, right_index=True)
        test = test.merge(cnt, how="left", left_on=bin_name, right_index=True)

        train.drop(columns=[bin_name], inplace=True)
        test.drop(columns=[bin_name], inplace=True)

    # Fill any missing counts with 0 (unseen categories)
    for c in train.columns:
        if c.startswith("orig_count_"):
            train[c] = train[c].fillna(0)
            test[c] = test[c].fillna(0)

    return train, test


def _prep_lgb_matrix(df: pd.DataFrame):
    # Convert object columns to category for LightGBM
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("category")
    return df


def _train_domain_classifier(cfg: Config, train_X: pd.DataFrame, test_X: pd.DataFrame):
    """Train a train-vs-test classifier to estimate p(test | x)."""
    X_all = pd.concat([train_X, test_X], axis=0, ignore_index=True)
    y_all = np.concatenate([np.zeros(len(train_X), dtype=np.int32), np.ones(len(test_X), dtype=np.int32)])

    X_all = _prep_lgb_matrix(X_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    oof = np.zeros(len(X_all), dtype=np.float64)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": cfg.domain_lr,
        "num_leaves": cfg.domain_num_leaves,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": cfg.seed,
    }

    for tr_idx, va_idx in skf.split(X_all, y_all):
        dtr = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx])
        dva = lgb.Dataset(X_all.iloc[va_idx], label=y_all[va_idx])
        model = lgb.train(
            params,
            dtr,
            num_boost_round=cfg.domain_n_estimators,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = model.predict(X_all.iloc[va_idx], num_iteration=model.best_iteration)

    p_test = oof  # cross-fit estimate
    p_test = np.clip(p_test, 1e-4, 1 - 1e-4)

    p_test_train = p_test[: len(train_X)]
    p_test_test = p_test[len(train_X) :]

    return p_test_train, p_test_test


def _shift_groups(p_test_train: np.ndarray, n_bins: int = 5):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p_test_train, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    groups = np.digitize(p_test_train, edges[1:-1], right=False)
    return groups


def _cat_fit_predict(
    cfg: Config,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: list[str],
):
    # CatBoost likes raw categoricals; ensure object types are strings
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].astype(str)
        return out

    X_tr2 = _prep(X_tr)
    X_va2 = _prep(X_va)
    X_te2 = _prep(X_te)

    # CatBoost wants cat_features as indices
    cat_idx = [X_tr2.columns.get_loc(c) for c in cat_cols] if cat_cols else []

    model = CatBoostClassifier(
        iterations=cfg.cat_n_estimators,
        depth=cfg.cat_depth,
        learning_rate=cfg.cat_lr,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=cfg.seed,
        verbose=200,
        use_best_model=True,
        od_type="Iter",
        od_wait=cfg.cat_early_stopping,
    )
    model.fit(X_tr2, y_tr, eval_set=(X_va2, y_va), cat_features=cat_idx)

    va_pred = model.predict_proba(X_va2)[:, 1]
    te_pred = model.predict_proba(X_te2)[:, 1]
    return va_pred, te_pred


def main():
    p = argparse.ArgumentParser(description="V15: shift-aware stacking / mixture-of-experts")
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--orig-csv", default=Config.orig_csv)
    p.add_argument("--out", default=Config.out)
    p.add_argument("--splits", type=int, default=Config.n_splits)
    p.add_argument(
        "--preds-dir",
        default="",
        help=(
            "Optional directory to dump OOF/test prediction artifacts as .npz files "
            "(keys: train_id,y,oof,test_id,test,name)."
        ),
    )
    args = p.parse_args()

    cfg = Config(data_dir=args.data_dir, orig_csv=args.orig_csv, n_splits=args.splits, out=args.out)

    train, test, sample = _read_data(cfg)

    y = train[TARGET].astype(int).to_numpy()
    X_train_base = train.drop(columns=[TARGET])
    X_test_base = test.copy()

    # Domain probability (train-vs-test)
    p_test_train, p_test_test = _train_domain_classifier(cfg, X_train_base, X_test_base)

    # Add p_test as a feature for all models
    X_train_base = X_train_base.copy()
    X_test_base = X_test_base.copy()
    X_train_base["p_test"] = p_test_train
    X_test_base["p_test"] = p_test_test

    # Add external orig encodings for one of the experts
    X_train_orig, X_test_orig = _add_orig_mean_count_features(
        X_train_base.copy(), X_test_base.copy(), cfg.orig_csv
    )

    # Define shift-aware groups for splitting
    groups = _shift_groups(p_test_train, n_bins=5)

    # Shift-aware CV: stratify on label but enforce groups based on domain probability quantiles.
    cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    # Base expert OOF/test predictions
    oof_lgb_raw = np.zeros(len(train), dtype=np.float64)
    oof_lgb_orig = np.zeros(len(train), dtype=np.float64)
    oof_cat = np.zeros(len(train), dtype=np.float64)

    test_lgb_raw = np.zeros(len(test), dtype=np.float64)
    test_lgb_orig = np.zeros(len(test), dtype=np.float64)
    test_cat = np.zeros(len(test), dtype=np.float64)

    # Identify categorical columns
    cat_cols = X_train_base.select_dtypes(include=["object", "category"]).columns.tolist()
    # CatBoost can't take ID as numeric category? It can, but usually not helpful.
    # Keep as-is; the model will ignore or use it.

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train_base, y, groups=groups), 1):
        # Force some shift mixing by sorting within fold not possible; instead, we’ll reweight using p_test.
        X_tr = X_train_base.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = X_train_base.iloc[va_idx].copy()
        y_va = y[va_idx]

        X_tr_o = X_train_orig.iloc[tr_idx].copy()
        X_va_o = X_train_orig.iloc[va_idx].copy()

        # Density ratio weights ~ p(test|x) / (1 - p(test|x))
        w_tr = (X_tr["p_test"].to_numpy() / (1.0 - X_tr["p_test"].to_numpy()))
        w_tr = np.clip(w_tr, 0.1, 20.0)

        # LightGBM raw expert
        X_tr_l = _prep_lgb_matrix(X_tr.copy())
        X_va_l = _prep_lgb_matrix(X_va.copy())
        X_te_l = _prep_lgb_matrix(X_test_base.copy())

        dtr = lgb.Dataset(X_tr_l, label=y_tr, weight=w_tr)
        dva = lgb.Dataset(X_va_l, label=y_va)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": cfg.lgb_lr,
            "num_leaves": cfg.lgb_num_leaves,
            "min_data_in_leaf": cfg.lgb_min_data_in_leaf,
            "feature_fraction": cfg.lgb_feature_fraction,
            "bagging_fraction": cfg.lgb_bagging_fraction,
            "bagging_freq": cfg.lgb_bagging_freq,
            "lambda_l2": cfg.lgb_lambda_l2,
            "verbosity": -1,
            "seed": cfg.seed + fold,
        }

        m_raw = lgb.train(
            params,
            dtr,
            num_boost_round=cfg.lgb_n_estimators,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(cfg.lgb_early_stopping, verbose=False)],
        )

        pred_va_raw = m_raw.predict(X_va_l, num_iteration=m_raw.best_iteration)
        pred_te_raw = m_raw.predict(X_te_l, num_iteration=m_raw.best_iteration)

        oof_lgb_raw[va_idx] = pred_va_raw
        test_lgb_raw += pred_te_raw / cfg.n_splits

        # LightGBM + orig features expert
        X_tr_ol = _prep_lgb_matrix(X_tr_o.copy())
        X_va_ol = _prep_lgb_matrix(X_va_o.copy())
        X_te_ol = _prep_lgb_matrix(X_test_orig.copy())

        dtr_o = lgb.Dataset(X_tr_ol, label=y_tr, weight=w_tr)
        dva_o = lgb.Dataset(X_va_ol, label=y_va)

        params_o = dict(params)
        params_o["seed"] = cfg.seed + 100 + fold

        m_orig = lgb.train(
            params_o,
            dtr_o,
            num_boost_round=cfg.lgb_n_estimators,
            valid_sets=[dva_o],
            callbacks=[lgb.early_stopping(cfg.lgb_early_stopping, verbose=False)],
        )

        pred_va_orig = m_orig.predict(X_va_ol, num_iteration=m_orig.best_iteration)
        pred_te_orig = m_orig.predict(X_te_ol, num_iteration=m_orig.best_iteration)

        oof_lgb_orig[va_idx] = pred_va_orig
        test_lgb_orig += pred_te_orig / cfg.n_splits

        # CatBoost expert (use base features + p_test; avoid huge orig-encoded matrix here)
        X_tr_c = X_tr.copy()
        X_va_c = X_va.copy()
        X_te_c = X_test_base.copy()

        va_pred_cat, te_pred_cat = _cat_fit_predict(
            cfg,
            X_tr=X_tr_c,
            y_tr=y_tr,
            X_va=X_va_c,
            y_va=y_va,
            X_te=X_te_c,
            cat_cols=cat_cols,
        )
        oof_cat[va_idx] = va_pred_cat
        test_cat += te_pred_cat / cfg.n_splits

        auc_raw = roc_auc_score(y_va, pred_va_raw)
        auc_orig = roc_auc_score(y_va, pred_va_orig)
        auc_cat = roc_auc_score(y_va, va_pred_cat)
        print(f"Fold {fold}: AUC lgb_raw={auc_raw:.5f} | lgb_orig={auc_orig:.5f} | cat={auc_cat:.5f}")

    # Meta-model (ridge logistic regression) on OOF predictions + p_test (+PA if available)
    meta_train = pd.DataFrame(
        {
            "p_lgb_raw": oof_lgb_raw,
            "p_lgb_orig": oof_lgb_orig,
            "p_cat": oof_cat,
            "p_test": p_test_train,
        }
    )
    meta_test = pd.DataFrame(
        {
            "p_lgb_raw": test_lgb_raw,
            "p_lgb_orig": test_lgb_orig,
            "p_cat": test_cat,
            "p_test": p_test_test,
        }
    )

    pa_col = "physical_activity_minutes_per_week"
    if pa_col in X_train_base.columns:
        meta_train[pa_col] = X_train_base[pa_col].astype(float).fillna(X_train_base[pa_col].median())
        meta_test[pa_col] = X_test_base[pa_col].astype(float).fillna(X_train_base[pa_col].median())

    # Scale meta features (helps LR stability)
    scaler = StandardScaler()
    X_meta_tr = scaler.fit_transform(meta_train.values)
    X_meta_te = scaler.transform(meta_test.values)

    meta = LogisticRegression(
        C=0.25,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=cfg.seed,
    )
    # Cross-fit meta-model AUC (avoid optimistic in-sample evaluation)
    oof_meta = np.zeros(len(train), dtype=np.float64)
    for tr_idx, va_idx in cv.split(meta_train, y, groups=groups):
        m_tr = scaler.fit_transform(meta_train.iloc[tr_idx].values)
        m_va = scaler.transform(meta_train.iloc[va_idx].values)
        meta_fold = LogisticRegression(
            C=0.25,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=cfg.seed,
        )
        meta_fold.fit(m_tr, y[tr_idx])
        oof_meta[va_idx] = meta_fold.predict_proba(m_va)[:, 1]

    meta_auc = roc_auc_score(y, oof_meta)
    print("=" * 60)
    print(f"Meta OOF AUC (cross-fit): {meta_auc:.5f}")

    # Fit meta on full data for test inference
    scaler = StandardScaler()
    X_meta_tr = scaler.fit_transform(meta_train.values)
    X_meta_te = scaler.transform(meta_test.values)
    meta.fit(X_meta_tr, y)

    final_test = meta.predict_proba(X_meta_te)[:, 1]

    sub = pd.DataFrame({ID_COL: sample[ID_COL], TARGET: final_test})
    sub.to_csv(cfg.out, index=False)
    print(f"Saved {cfg.out}")

    if args.preds_dir:
        train_id = train[ID_COL].to_numpy(dtype=np.int64)
        test_id = test[ID_COL].to_numpy(dtype=np.int64)
        base = os.path.basename(cfg.out).replace(".csv", "")
        out_dir = args.preds_dir

        _dump_pred_npz(
            path=os.path.join(out_dir, f"{base}__lgb_raw.npz"),
            name=f"{base}__lgb_raw",
            train_id=train_id,
            y=y,
            oof=oof_lgb_raw,
            test_id=test_id,
            test=test_lgb_raw,
        )
        _dump_pred_npz(
            path=os.path.join(out_dir, f"{base}__lgb_orig.npz"),
            name=f"{base}__lgb_orig",
            train_id=train_id,
            y=y,
            oof=oof_lgb_orig,
            test_id=test_id,
            test=test_lgb_orig,
        )
        _dump_pred_npz(
            path=os.path.join(out_dir, f"{base}__cat.npz"),
            name=f"{base}__cat",
            train_id=train_id,
            y=y,
            oof=oof_cat,
            test_id=test_id,
            test=test_cat,
        )
        _dump_pred_npz(
            path=os.path.join(out_dir, f"{base}__meta.npz"),
            name=f"{base}__meta",
            train_id=train_id,
            y=y,
            oof=oof_meta,
            test_id=test_id,
            test=final_test,
        )
        print(f"Wrote prediction artifacts -> {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
