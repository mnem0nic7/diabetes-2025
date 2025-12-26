"""Train a lightweight model and write a daily submission CSV.

This script is designed to run in constrained environments (limited RAM/CPU).
It trains a small LightGBM model on a sampled subset of the training data,
then predicts the full test set and writes a Kaggle-ready submission CSV.

Usage:
    python scripts/daily_quick_submit.py --date 2025-12-24

Outputs:
    submission_daily_<date>.csv
"""

from __future__ import annotations

import argparse
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "diagnosed_diabetes"


def _sample_train_csv(
    path: Path,
    *,
    target: str,
    n_target: int,
    seed: int,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    got = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if target not in chunk.columns:
            raise SystemExit(f"Missing target column {target!r} in {path}")

        remaining = n_target - got
        if remaining <= 0:
            break

        # Sample a fraction that should, on average, hit the target size.
        frac = min(1.0, max(0.01, remaining / max(1, len(chunk))))
        # Add a bit of jitter so we don't repeatedly pick the same exact rows.
        frac = min(1.0, frac * float(rng.uniform(1.05, 1.25)))

        take = chunk.sample(frac=frac, random_state=int(rng.integers(0, 2**31 - 1)))
        parts.append(take)
        got += len(take)

    if not parts:
        raise SystemExit(f"No data read from {path}")

    df = pd.concat(parts, axis=0, ignore_index=True)
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=seed).reset_index(drop=True)

    return df


def _encode_object_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    obj_cols = [c for c in train.columns if train[c].dtype == "object"]
    for c in obj_cols:
        combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0, ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        train[c] = codes[: len(train)].astype(np.int32)
        test[c] = codes[len(train) :].astype(np.int32)

    # Downcast numerics to save memory.
    for df in (train, test):
        for c in df.columns:
            if c == TARGET:
                continue
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype(np.float32)
            elif pd.api.types.is_integer_dtype(df[c]):
                # Keep ids separately; features can be smaller.
                if c != "id":
                    df[c] = df[c].astype(np.int32)

    return train, test


def _density_ratio_weights(p_test: np.ndarray, *, clip: tuple[float, float] = (0.2, 5.0)) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p_test, eps, 1.0 - eps)
    w = p / (1.0 - p)
    w = np.clip(w, float(clip[0]), float(clip[1]))
    w = w / float(np.mean(w))
    return w.astype(np.float32)


def _fit_adversarial_weights(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    seed: int,
    adv_test_sample: int = 200_000,
) -> np.ndarray:
    """Fit a lightweight discriminator to estimate p(test|x) and return train weights."""

    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LightGBM is required for adversarial weights. Install it (pip install lightgbm).\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    rng = np.random.default_rng(int(seed))
    if adv_test_sample and len(X_test) > int(adv_test_sample):
        X_test_fit = X_test.sample(n=int(adv_test_sample), random_state=int(rng.integers(0, 2**31 - 1)))
    else:
        X_test_fit = X_test

    X_all = pd.concat([X_train, X_test_fit], axis=0, ignore_index=True)
    y_dom = np.concatenate([
        np.zeros(len(X_train), dtype=np.int8),
        np.ones(len(X_test_fit), dtype=np.int8),
    ])

    adv = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )
    adv.fit(X_all, y_dom)

    p_test = adv.predict_proba(X_train)[:, 1]
    return _density_ratio_weights(p_test)


def generate_submission(
    *,
    data_dir: str = "data",
    target: str = TARGET,
    train_sample: int = 200_000,
    seed: int = 42,
    adv_weights: bool = False,
    adv_test_sample: int = 200_000,
    out_path: str,
) -> Path:
    """Generate a Kaggle-ready submission CSV and return the output path."""

    data_dir_path = Path(data_dir)
    train_path = data_dir_path / "train.csv"
    test_path = data_dir_path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing {train_path} or {test_path}")

    # Import LightGBM lazily so the script can still show a nice error.
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LightGBM is required. Install it (pip install lightgbm) in your environment.\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    train_sample_df = _sample_train_csv(
        train_path,
        target=target,
        n_target=int(train_sample),
        seed=int(seed),
    )
    test = pd.read_csv(test_path)

    if "id" not in train_sample_df.columns or "id" not in test.columns:
        raise SystemExit("Missing 'id' column in train/test")

    test_ids = test["id"].astype(np.int64)

    train_sample_df, test = _encode_object_columns(train_sample_df, test)

    y = train_sample_df[target].astype(np.int8)
    X = train_sample_df.drop(columns=[target, "id"])
    X_test = test.drop(columns=["id"])

    sample_weight = None
    if adv_weights:
        sample_weight = _fit_adversarial_weights(
            X_train=X,
            X_test=X_test,
            seed=int(seed),
            adv_test_sample=int(adv_test_sample),
        )

    # Simple stratified split
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(X))
    idx0 = idx[y.to_numpy() == 0]
    idx1 = idx[y.to_numpy() == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n_val0 = max(1, int(0.2 * len(idx0)))
    n_val1 = max(1, int(0.2 * len(idx1)))

    val_idx = np.concatenate([idx0[:n_val0], idx1[:n_val1]])
    tr_idx = np.setdiff1d(idx, val_idx, assume_unique=False)

    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[val_idx]
    y_va = y.iloc[val_idx]

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=800,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight[tr_idx] if sample_weight is not None else None,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict_proba(X_test)[:, 1]
    out_path_p = Path(out_path)
    pd.DataFrame({"id": test_ids, target: pred}).to_csv(out_path_p, index=False)
    return out_path_p


def generate_submission_with_metrics(
    *,
    data_dir: str = "data",
    target: str = TARGET,
    train_sample: int = 200_000,
    seed: int = 42,
    adv_weights: bool = False,
    adv_test_sample: int = 200_000,
    out_path: str,
) -> tuple[Path, dict[str, float]]:
    """Generate a submission CSV and return (path, metrics).

    Metrics are computed on the held-out validation split used during training.
    """

    # Reuse the same implementation but compute and return validation AUC.
    # This stays lightweight and avoids refactoring the whole pipeline into classes.
    from sklearn.metrics import roc_auc_score  # local import to keep startup cheap

    data_dir_path = Path(data_dir)
    train_path = data_dir_path / "train.csv"
    test_path = data_dir_path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing {train_path} or {test_path}")

    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LightGBM is required. Install it (pip install lightgbm) in your environment.\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    train_sample_df = _sample_train_csv(
        train_path,
        target=target,
        n_target=int(train_sample),
        seed=int(seed),
    )
    test = pd.read_csv(test_path)
    if "id" not in train_sample_df.columns or "id" not in test.columns:
        raise SystemExit("Missing 'id' column in train/test")

    test_ids = test["id"].astype(np.int64)
    train_sample_df, test = _encode_object_columns(train_sample_df, test)

    y = train_sample_df[target].astype(np.int8)
    X = train_sample_df.drop(columns=[target, "id"])
    X_test = test.drop(columns=["id"])

    sample_weight = None
    if adv_weights:
        sample_weight = _fit_adversarial_weights(
            X_train=X,
            X_test=X_test,
            seed=int(seed),
            adv_test_sample=int(adv_test_sample),
        )

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(X))
    idx0 = idx[y.to_numpy() == 0]
    idx1 = idx[y.to_numpy() == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n_val0 = max(1, int(0.2 * len(idx0)))
    n_val1 = max(1, int(0.2 * len(idx1)))
    val_idx = np.concatenate([idx0[:n_val0], idx1[:n_val1]])
    tr_idx = np.setdiff1d(idx, val_idx, assume_unique=False)

    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[val_idx]
    y_va = y.iloc[val_idx]

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=800,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight[tr_idx] if sample_weight is not None else None,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    val_auc = float(roc_auc_score(y_va.to_numpy(), va_pred))

    pred = model.predict_proba(X_test)[:, 1]
    out_path_p = Path(out_path)
    pd.DataFrame({"id": test_ids, target: pred}).to_csv(out_path_p, index=False)

    metrics = {
        "val_auc": val_auc,
        "best_iteration": float(getattr(model, "best_iteration_", 0) or 0),
    }
    return out_path_p, metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train a small model and write a daily submission CSV.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--train-sample", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adv-weights", action="store_true", help="Use adversarial train-vs-test density-ratio weights")
    p.add_argument(
        "--adv-test-sample",
        type=int,
        default=200_000,
        help="How many test rows to sample when fitting the adversarial discriminator (0=all).",
    )
    p.add_argument("--date", default=str(_date.today()))
    p.add_argument("--out", default="")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else Path(f"submission_daily_{args.date}.csv")
    out_path = generate_submission(
        data_dir=args.data_dir,
        target=args.target,
        train_sample=int(args.train_sample),
        seed=int(args.seed),
        adv_weights=bool(args.adv_weights),
        adv_test_sample=int(args.adv_test_sample),
        out_path=str(out_path),
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
