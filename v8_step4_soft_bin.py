import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold


warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 10
TARGET = "diagnosed_diabetes"


def to_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category")
    return out


def fit_domain_classifier_get_p_test(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 42):
    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_dom = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "verbose": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_all))
    for tr, va in kf.split(X_all, y_dom):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_all.iloc[tr],
            y_dom[tr],
            eval_set=[(X_all.iloc[va], y_dom[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va] = m.predict_proba(X_all.iloc[va])[:, 1]

    dom_auc = roc_auc_score(y_dom, oof)
    print(f"Domain (train vs test) CV AUC: {dom_auc:.5f}")

    final = lgb.LGBMClassifier(**params)
    final.fit(X_all, y_dom)
    p_all = final.predict_proba(X_all)[:, 1]
    p_train = p_all[: len(X_train)]
    p_test = p_all[len(X_train) :]
    return p_train, p_test, dom_auc


def density_ratio_weights(p: np.ndarray, clip=(0.2, 5.0)):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    w = p / (1 - p)
    w = np.clip(w, clip[0], clip[1])
    w = w / np.mean(w)
    return w


def make_shift_groups(p_test: np.ndarray, n_bins: int = 50):
    qs = np.quantile(p_test, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) <= 2:
        return np.zeros_like(p_test, dtype=int)
    groups = np.digitize(p_test, qs[1:-1], right=True)
    return groups


def weighted_auc(y_true, y_score, sample_weight):
    return roc_auc_score(y_true, y_score, sample_weight=sample_weight)


def cv_lgb_shift_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    p_test_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    n_splits: int = 10,
    use_shift_groups: bool = True,
    label: str = "model",
):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 8000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 80,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "random_state": seed,
        "verbose": -1,
    }

    if use_shift_groups:
        groups = make_shift_groups(p_test_train, n_bins=50)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_top = []

    w_proxy = density_ratio_weights(p_test_train, clip=(0.2, 5.0))

    for tr, va in splits:
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        fit_w = sample_weight[tr] if sample_weight is not None else None

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr,
            y_tr,
            sample_weight=fit_w,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        p_va = m.predict_proba(X_va)[:, 1]
        oof[va] = p_va
        test_pred += m.predict_proba(X_test)[:, 1] / n_splits

        thr = np.quantile(p_test_train[va], 0.70)
        idx_top = va[p_test_train[va] >= thr]
        if len(idx_top) > 50:
            fold_top.append(roc_auc_score(y.iloc[idx_top], oof[idx_top]))
        else:
            fold_top.append(np.nan)

    overall_std = roc_auc_score(y, oof)
    overall_w = weighted_auc(y, oof, w_proxy)
    overall_top = np.nanmean(fold_top)

    print(
        f"[{label}] CV AUC std: {overall_std:.5f} | weighted: {overall_w:.5f} | val-top30%: {overall_top:.5f}"
    )
    return oof, test_pred


def quantile_bin_joint(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list[str],
    n_bins: int = 50,
    drop_original: bool = True,
    suffix: str = "__qbin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xt = X_train.copy()
    Xs = X_test.copy()

    for c in cols:
        if c not in Xt.columns or c not in Xs.columns:
            continue
        if not pd.api.types.is_numeric_dtype(Xt[c]):
            continue

        s_all = pd.concat([Xt[c], Xs[c]], axis=0, ignore_index=True)
        try:
            b_all = pd.qcut(s_all, q=n_bins, duplicates="drop")
        except Exception:
            continue

        # Avoid pandas Interval categoricals (can trigger LightGBM JSON circular refs).
        # Use stable integer codes as the coarsened representation.
        codes = b_all.cat.codes.astype("int16")
        Xt[c + suffix] = codes.iloc[: len(Xt)].reset_index(drop=True)
        Xs[c + suffix] = codes.iloc[len(Xt) :].reset_index(drop=True)

        if drop_original:
            Xt = Xt.drop(columns=[c])
            Xs = Xs.drop(columns=[c])

    return Xt, Xs


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    test_ids = test["id"]
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    train = to_category(train)
    test = to_category(test)

    y = train[TARGET].astype(int)
    X = train.drop(columns=[TARGET])
    X_test = test.copy()

    p_test_train, p_test_test, _ = fit_domain_classifier_get_p_test(
        X, X_test, seed=SEED
    )

    shift_candidates = [
        "triglycerides",
        "cholesterol_total",
        "cholesterol_hdl",
        "cholesterol_ldl",
        "glucose",
        "hba1c",
        "insulin",
        "bmi",
        "waist_circumference",
        "hip_circumference",
        "waist_hip_ratio",
        "systolic_bp",
        "diastolic_bp",
        "age",
        "income",
        "education_years",
        "physical_activity",
        "sleep_hours",
        "smoking_pack_years",
        "alcohol_units",
        "family_history_diabetes",
        "diet_score",
        "stress_level",
        "heart_rate",
        "creatinine",
        "egfr",
        "alt",
        "ast",
        "crp",
    ]

    X_soft, X_test_soft = quantile_bin_joint(
        X, X_test, shift_candidates, n_bins=50, drop_original=True
    )
    print("Soft-binned shapes:", X_soft.shape, X_test_soft.shape)

    _, pred_soft = cv_lgb_shift_metrics(
        X_soft,
        y,
        X_test_soft,
        p_test_train,
        sample_weight=None,
        seed=SEED,
        n_splits=N_SPLITS,
        use_shift_groups=True,
        label="V8_SOFT_BIN",
    )

    out_soft = pd.DataFrame({"id": test_ids, TARGET: pred_soft})
    out_soft.to_csv("submission_v8_soft_bin.csv", index=False)
    print("Saved: submission_v8_soft_bin.csv")

    base_path = "submission_v8_base_shiftcv.csv"
    if os.path.exists(base_path):
        base = pd.read_csv(base_path)
        pred_base = base[TARGET].to_numpy()
        pred_blend = 0.6 * pred_base + 0.4 * pred_soft
        out_blend = pd.DataFrame({"id": test_ids, TARGET: pred_blend})
        out_blend.to_csv("submission_v8_blend_base_softbin_60_40.csv", index=False)
        print("Saved: submission_v8_blend_base_softbin_60_40.csv")
    else:
        print(f"Note: {base_path} not found; skipping base+soft blend")


if __name__ == "__main__":
    main()
