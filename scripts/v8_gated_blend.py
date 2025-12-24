import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")

SEED = 42
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
    return p_train, p_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    test_ids = test["id"]
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    train = to_category(train)
    test = to_category(test)

    X = train.drop(columns=[TARGET])

    _, p_test_test = fit_domain_classifier_get_p_test(X, test, seed=SEED)
    q = np.quantile(p_test_test, [0.5, 0.7, 0.85])
    print("p_test(test) quantiles q50/q70/q85:", q)

    base = pd.read_csv("submission_v8_base_shiftcv.csv")
    sub = pd.read_csv("submission_v8_subset_top50.csv")

    pred_base = base[TARGET].to_numpy()
    pred_sub = sub[TARGET].to_numpy()

    # Gate: sigmoid around q70 with temperature picked to be reasonably sharp.
    t0 = float(q[1])
    temp = 0.06
    alpha = sigmoid((p_test_test - t0) / temp)

    pred = (1 - alpha) * pred_base + alpha * pred_sub

    out = pd.DataFrame({"id": test_ids, TARGET: pred})
    out_path = "submission_v8_gated_base_sub50_sigmoid.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Optional: slightly more aggressive gate (sharper transition)
    temp2 = 0.04
    alpha2 = sigmoid((p_test_test - t0) / temp2)
    pred2 = (1 - alpha2) * pred_base + alpha2 * pred_sub
    out2_path = "submission_v8_gated_base_sub50_sigmoid_sharp.csv"
    pd.DataFrame({"id": test_ids, TARGET: pred2}).to_csv(out2_path, index=False)
    print("Saved:", out2_path)


if __name__ == "__main__":
    main()
