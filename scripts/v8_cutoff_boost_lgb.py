import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    target: str = "diagnosed_diabetes"
    cutoff_id: int = 678260
    post_weight: float = 5.0  # how much to upweight post-cutoff rows in final refit
    seed: int = 42
    num_boost_round: int = 8000
    early_stopping_rounds: int = 200


def to_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category")
    return out


def add_orig_mean_count_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    *,
    target_col: str,
    max_unique: int,
    smooth_m: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add orig-based mean/count encoding features.

    We compute, for each shared feature column c, two maps from the external
    ("orig") dataset:
      - orig_mean_c: E[y | c=value] in orig
      - orig_count_c: count(c=value) in orig

    This avoids concatenating orig rows (which can worsen covariate shift), but
    still transfers signal about feature-value risk.
    """

    if target_col not in orig_df.columns:
        raise ValueError(f"orig_df is missing target column: {target_col!r}")

    # Work on copies to avoid surprising callers.
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()

    global_mean = float(orig[target_col].mean())

    shared_cols = [
        c
        for c in train.columns
        if c in orig.columns and c != target_col
    ]

    for c in shared_cols:
        # Ensure type alignment for categorical/object columns.
        if train[c].dtype == "object" or orig[c].dtype == "object":
            orig[c] = orig[c].astype(str)
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)

        nunique = int(orig[c].nunique(dropna=False))
        if nunique > int(max_unique):
            continue

        mean_map = orig.groupby(c, dropna=False)[target_col].mean()
        count_map = orig.groupby(c, dropna=False).size()

        m_col = f"orig_mean_{c}"
        n_col = f"orig_count_{c}"

        # Raw maps
        t_mean = train[c].map(mean_map)
        te_mean = test[c].map(mean_map)
        t_cnt = train[c].map(count_map).fillna(0)
        te_cnt = test[c].map(count_map).fillna(0)

        # Optional smoothing: (cnt*mean + m*global) / (cnt + m)
        if smooth_m and float(smooth_m) > 0:
            m = float(smooth_m)
            t_mean = (t_cnt * t_mean.fillna(global_mean) + m * global_mean) / (t_cnt + m)
            te_mean = (te_cnt * te_mean.fillna(global_mean) + m * global_mean) / (te_cnt + m)
        else:
            t_mean = t_mean.fillna(global_mean)
            te_mean = te_mean.fillna(global_mean)

        train[m_col] = t_mean.astype("float32")
        test[m_col] = te_mean.astype("float32")
        train[n_col] = t_cnt.astype("float32")
        test[n_col] = te_cnt.astype("float32")

    return train, test


def main() -> None:
    p = argparse.ArgumentParser(description="Train cutoff-boosted LightGBM and write a submission CSV.")
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--target", default=Config.target)
    p.add_argument("--cutoff-id", type=int, default=Config.cutoff_id)
    p.add_argument("--post-weight", type=float, default=Config.post_weight)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num-boost-round", type=int, default=Config.num_boost_round)
    p.add_argument("--early-stopping-rounds", type=int, default=Config.early_stopping_rounds)
    p.add_argument(
        "--orig-csv",
        default="",
        help=(
            "Optional path to an external/original dataset CSV with the same feature columns plus the target. "
            "If provided, adds orig_mean_* and orig_count_* features."
        ),
    )
    p.add_argument(
        "--orig-target",
        default=Config.target,
        help="Target column name in --orig-csv (defaults to --target).",
    )
    p.add_argument(
        "--orig-max-unique",
        type=int,
        default=1000,
        help="Skip orig-encoding for columns with > this many unique values in orig.",
    )
    p.add_argument(
        "--orig-smooth-m",
        type=float,
        default=0.0,
        help="Smoothing strength m for orig_mean encodings; 0 disables smoothing.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output submission path. If omitted, a name is generated from post_weight.",
    )
    args = p.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        target=args.target,
        cutoff_id=args.cutoff_id,
        post_weight=args.post_weight,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    train = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))

    test_ids = test["id"].copy()
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    if args.orig_csv:
        orig = pd.read_csv(args.orig_csv)
        train, test = add_orig_mean_count_features(
            train,
            test,
            orig,
            target_col=args.orig_target,
            max_unique=args.orig_max_unique,
            smooth_m=args.orig_smooth_m,
        )

    train = to_category(train)
    test = to_category(test)

    y = train[cfg.target].astype(int)
    X = train.drop(columns=[cfg.target])

    # Use an ID-ordered split as in the referenced notebook: pre-cutoff train, post-cutoff as validation.
    # NOTE: this assumes the original train.csv is ordered by id; if not, we still have the cutoff row index behavior.
    # We rebuild the split using the original order by merging the id back in via the saved file ordering.
    train_raw = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"), usecols=["id"])
    is_post = train_raw["id"].values >= cfg.cutoff_id

    X_tr, y_tr = X.loc[~is_post], y.loc[~is_post]
    X_va, y_va = X.loc[is_post], y.loc[is_post]

    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        seed=cfg.seed,
        verbosity=-1,
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=int(cfg.num_boost_round),
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(int(cfg.early_stopping_rounds), verbose=True)],
    )

    val_pred = model.predict(X_va)
    val_auc = roc_auc_score(y_va, val_pred)
    print(f"Holdout(post-cutoff) AUC: {val_auc:.5f} | best_iter={model.best_iteration}")

    # Refit on full data with post-cutoff rows upweighted.
    w = np.ones(len(X), dtype=np.float32)
    w[is_post] *= float(cfg.post_weight)

    dfull = lgb.Dataset(X, label=y, weight=w, categorical_feature=cat_cols, free_raw_data=False)
    final = lgb.train(
        params,
        dfull,
        num_boost_round=int(model.best_iteration),
    )

    test_pred = final.predict(test)

    if args.out:
        out_path = args.out
    else:
        # Keep filenames stable/readable for weight sweeps.
        w_str = f"{cfg.post_weight:g}".replace(".", "p")
        if args.orig_csv:
            out_path = f"submission_v8_cutoff_boost_w{w_str}_origenc.csv"
        else:
            out_path = f"submission_v8_cutoff_boost_w{w_str}.csv"
    pd.DataFrame({"id": test_ids, cfg.target: test_pred}).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
