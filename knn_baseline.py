import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Defaults:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    target: str = "diagnosed_diabetes"
    id_col: str = "id"

    n_splits: int = 5
    seed: int = 42

    # KNN hyperparams (distance-weighted tends to be safer)
    n_neighbors: int = 75
    k_values: str = "31,51,75,101"  # optional sweep
    weights: str = "distance"  # uniform|distance
    metric: str = "minkowski"  # minkowski (p=2 -> euclidean)
    p: int = 2

    # Speed knob: CV and fit can be very slow on full 700k rows.
    # Use a sample by default to get a quick read on KNN viability.
    sample_n: int = 200_000

    submission_path: str = "scratch/submission_knn_baseline.csv"


def _abs_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Pipeline,
    splits: int,
    seed: int,
    label: str,
) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=int(splits), shuffle=True, random_state=int(seed))

    oof = np.zeros(len(X), dtype=float)
    fold_aucs: list[float] = []

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tf = time.time()
        print(f"{label} fold {fold}/{splits}: fitting on {len(tr_idx):,} rows...")
        model.fit(X_tr, y_tr)
        print(f"{label} fold {fold}/{splits}: fit done in {time.time() - tf:.1f}s; predicting...")

        tp = time.time()
        p_va = model.predict_proba(X_va)[:, 1]
        print(f"{label} fold {fold}/{splits}: predict done in {time.time() - tp:.1f}s")
        oof[va_idx] = p_va

        auc = roc_auc_score(y_va, p_va)
        fold_aucs.append(float(auc))
        print(f"{label} fold {fold}: AUC={auc:.6f}")

    overall = float(roc_auc_score(y, oof))
    mean_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"{label} OOF AUC: {overall:.6f}")
    print(f"{label} fold AUC mean±std: {mean_std[0]:.6f} ± {mean_std[1]:.6f}")
    print(f"{label} CV total time: {time.time() - t0:.1f}s")
    return overall, mean_std[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="KNN baseline with preprocessing + CV AUC (+ optional k sweep).")
    ap.add_argument("--train", default=Defaults.train_path)
    ap.add_argument("--test", default=Defaults.test_path)
    ap.add_argument("--target", default=Defaults.target)
    ap.add_argument("--id-col", default=Defaults.id_col)
    ap.add_argument("--splits", type=int, default=Defaults.n_splits)
    ap.add_argument("--seed", type=int, default=Defaults.seed)
    ap.add_argument("--n-neighbors", type=int, default=Defaults.n_neighbors)
    ap.add_argument(
        "--k-values",
        default=Defaults.k_values,
        help="Comma-separated k values to sweep; set empty to disable sweep",
    )
    ap.add_argument("--weights", default=Defaults.weights)
    ap.add_argument("--metric", default=Defaults.metric)
    ap.add_argument("--p", type=int, default=Defaults.p)
    ap.add_argument(
        "--sample-n",
        type=int,
        default=Defaults.sample_n,
        help="Use at most N rows (stratified) for CV+fit. Set 0 for full data (may be very slow).",
    )
    ap.add_argument(
        "--cv-only",
        action="store_true",
        help="Run CV only; do not fit full and write submission",
    )
    ap.add_argument("--out", default=Defaults.submission_path)
    args = ap.parse_args()

    train_path = _abs_path(args.train)
    test_path = _abs_path(args.test)
    out_path = _abs_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading train: {train_path.as_posix()}")
    train = pd.read_csv(train_path)
    print(f"Loading test:  {test_path.as_posix()}")
    test = pd.read_csv(test_path)
    print(f"train shape={train.shape} test shape={test.shape}")

    if args.id_col not in train.columns or args.id_col not in test.columns:
        raise SystemExit(f"Missing id column '{args.id_col}'")
    if args.target not in train.columns:
        raise SystemExit(f"Missing target column '{args.target}'")

    y_full = train[args.target].astype(int).to_numpy()
    X_full = train.drop(columns=[args.target])
    X_test = test.copy()

    # Optional stratified sampling for speed.
    if int(args.sample_n) and int(args.sample_n) > 0 and int(args.sample_n) < len(X_full):
        n = int(args.sample_n)
        rng = np.random.RandomState(int(args.seed))
        y0 = np.where(y_full == 0)[0]
        y1 = np.where(y_full == 1)[0]
        n1 = max(1, int(round(n * (len(y1) / len(y_full)))))
        n0 = max(1, n - n1)
        idx = np.concatenate([
            rng.choice(y0, size=min(n0, len(y0)), replace=False),
            rng.choice(y1, size=min(n1, len(y1)), replace=False),
        ])
        rng.shuffle(idx)
        X = X_full.iloc[idx].reset_index(drop=True)
        y = y_full[idx]
        sample_note = f"sampled {len(X):,}/{len(X_full):,} rows"
    else:
        X = X_full
        y = y_full
        sample_note = f"full {len(X):,} rows"

    print(f"Using {sample_note} for CV/fit")

    def make_model(k: int) -> Pipeline:
        pre = _build_preprocessor(X)
        knn = KNeighborsClassifier(
            n_neighbors=int(k),
            weights=args.weights,
            metric=args.metric,
            p=int(args.p),
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("knn", knn)])

    # Optional sweep to get a quick answer on what KNN yields.
    best_k = int(args.n_neighbors)
    if str(args.k_values).strip():
        k_values = [int(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
        k_values = [k for k in k_values if k >= 3]
        if k_values:
            best_auc = -1.0
            print(f"Sweeping k values: {k_values}")
            for k in k_values:
                label = f"[k={k}]"
                model = make_model(k)
                overall_auc, _ = _run_cv(X, y, model, splits=int(args.splits), seed=int(args.seed), label=label)
                if overall_auc > best_auc:
                    best_auc = overall_auc
                    best_k = k
            print(f"Best from sweep: k={best_k}")
        else:
            print("No valid k-values in --k-values; using --n-neighbors")
    else:
        print("k sweep disabled; using --n-neighbors")

    # Run CV once for selected k if sweep disabled.
    if not str(args.k_values).strip():
        model = make_model(best_k)
        _run_cv(X, y, model, splits=int(args.splits), seed=int(args.seed), label=f"[k={best_k}]")

    if args.cv_only:
        print("--cv-only set; skipping submission write")
        return

    # Fit on the same data we evaluated (sample by default) and write submission.
    print(f"Fitting final model (k={best_k}) on {sample_note}...")
    model = make_model(best_k)
    t_fit = time.time()
    model.fit(X, y)
    print(f"Final fit done in {time.time() - t_fit:.1f}s; predicting test...")
    t_pred = time.time()
    p_test = model.predict_proba(X_test)[:, 1]
    print(f"Test predict done in {time.time() - t_pred:.1f}s")

    # If output path doesn't encode k/sample, make it informative.
    out_path_final = out_path
    if out_path_final.name == Path(Defaults.submission_path).name:
        suffix = f"knn_k{best_k}_n{len(X):d}.csv"
        out_path_final = out_path_final.with_name(suffix)

    sub = pd.DataFrame({args.id_col: test[args.id_col].astype(int), args.target: p_test.astype(float)})
    sub.to_csv(out_path_final, index=False)
    print(f"Wrote submission: {out_path_final.as_posix()}  (n={len(sub)})")


if __name__ == "__main__":
    main()
