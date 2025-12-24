import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD


REPO_ROOT = Path(__file__).resolve().parent.parent


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

    # Optional dimensionality reduction for sparse one-hot data.
    # Using SVD can significantly speed up KNN and often improves generalization.
    # Set to 0 to disable.
    svd_components: int = 256

    submission_path: str = "submissions/submission_knn_baseline.csv"


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


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.effective_components_: int | None = None
        self._svd: TruncatedSVD | None = None
        self._passthrough: bool = False

    def fit(self, X, y=None):
        n_features = int(getattr(X, "shape", [0, 0])[1])
        req = int(self.n_components)

        # Disable if requested 0 or if SVD cannot be applied meaningfully.
        if req <= 0 or n_features <= 1:
            self._passthrough = True
            self.effective_components_ = None
            self._svd = None
            return self

        # TruncatedSVD requires n_components <= n_features - 1.
        eff = min(req, max(1, n_features - 1))
        self.effective_components_ = int(eff)
        self._passthrough = False
        self._svd = TruncatedSVD(n_components=int(eff), random_state=int(self.random_state))
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._passthrough or self._svd is None:
            return X
        return self._svd.transform(X)


def _build_knn_pipeline(
    X_ref: pd.DataFrame,
    *,
    k: int,
    weights: str,
    metric: str,
    p: int,
    n_jobs: int,
    seed: int,
    svd_components: int,
    normalize: bool,
    algorithm: str,
) -> Pipeline:
    pre = _build_preprocessor(X_ref)

    steps: list[tuple[str, object]] = [("pre", pre)]

    if int(svd_components) and int(svd_components) > 0:
        steps.append(("svd", SafeTruncatedSVD(n_components=int(svd_components), random_state=int(seed))))

    # Cosine distance is much more stable when each row is L2-normalized.
    if normalize or str(metric).lower() == "cosine":
        steps.append(("norm", Normalizer(norm="l2")))

    knn = KNeighborsClassifier(
        n_neighbors=int(k),
        weights=weights,
        metric=metric,
        p=int(p),
        n_jobs=int(n_jobs),
        algorithm=algorithm,
    )
    steps.append(("knn", knn))
    return Pipeline(steps=steps)


def _metric_for_gpu(metric: str, p: int) -> str:
    m = str(metric).lower().strip()
    if m in {"cosine", "euclidean", "l2"}:
        return "cosine" if m == "cosine" else "euclidean"
    if m == "minkowski":
        if int(p) == 2:
            return "euclidean"
        raise ValueError("GPU KNN only supports minkowski with p=2 (euclidean)")
    raise ValueError(f"Unsupported GPU metric: {metric}")


def _build_transformer_pipeline(
    X_ref: pd.DataFrame,
    *,
    seed: int,
    svd_components: int,
    normalize: bool,
    metric: str,
) -> Pipeline:
    pre = _build_preprocessor(X_ref)
    steps: list[tuple[str, object]] = [("pre", pre)]

    if int(svd_components) and int(svd_components) > 0:
        steps.append(("svd", SafeTruncatedSVD(n_components=int(svd_components), random_state=int(seed))))

    if normalize or str(metric).lower() == "cosine":
        steps.append(("norm", Normalizer(norm="l2")))
    return Pipeline(steps=steps)


def _run_cv_gpu(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    k: int,
    weights: str,
    metric: str,
    p: int,
    splits: int,
    seed: int,
    svd_components: int,
    normalize: bool,
    label: str,
    quiet: bool,
) -> tuple[float, float]:
    try:
        import cupy as cp
        from cuml.neighbors import KNeighborsClassifier as cuKNN
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "GPU mode requires cuML + CuPy. Activate the rapids env and run via micromamba."
        ) from e

    if int(svd_components) <= 0:
        raise ValueError("GPU mode requires --svd-components > 0 (to avoid densifying huge OHE matrices)")

    gpu_metric = _metric_for_gpu(metric, p)
    skf = StratifiedKFold(n_splits=int(splits), shuffle=True, random_state=int(seed))

    oof = np.zeros(len(X), dtype=float)
    fold_aucs: list[float] = []
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tf = time.time()
        if not quiet:
            print(f"{label} fold {fold}/{splits}: preprocessing...")
        transformer = _build_transformer_pipeline(
            X_tr,
            seed=int(seed),
            svd_components=int(svd_components),
            normalize=bool(normalize),
            metric=str(metric),
        )
        Xtr = transformer.fit_transform(X_tr)
        Xva = transformer.transform(X_va)
        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xva = np.asarray(Xva, dtype=np.float32)
        if not quiet:
            print(
                f"{label} fold {fold}/{splits}: preproc done in {time.time() - tf:.1f}s; "
                "moving to GPU..."
            )

        Xtr_g = cp.asarray(Xtr)
        Xva_g = cp.asarray(Xva)
        ytr_g = cp.asarray(y_tr.astype(np.int32))

        tg = time.time()
        model = cuKNN(
            n_neighbors=int(k),
            weights=str(weights),
            metric=str(gpu_metric),
        )
        model.fit(Xtr_g, ytr_g)
        p_va = model.predict_proba(Xva_g)[:, 1]
        p_va = cp.asnumpy(p_va).astype(float)
        if not quiet:
            print(f"{label} fold {fold}/{splits}: GPU fit+predict in {time.time() - tg:.1f}s")

        oof[va_idx] = p_va
        auc = roc_auc_score(y_va, p_va)
        fold_aucs.append(float(auc))
        if not quiet:
            print(f"{label} fold {fold}: AUC={auc:.6f}")

    overall = float(roc_auc_score(y, oof))
    mean_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"{label} OOF AUC: {overall:.6f}")
    print(f"{label} fold AUC mean±std: {mean_std[0]:.6f} ± {mean_std[1]:.6f}")
    if not quiet:
        print(f"{label} CV total time: {time.time() - t0:.1f}s")
    return overall, mean_std[0]


def _fit_predict_gpu(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_test: pd.DataFrame,
    *,
    k: int,
    weights: str,
    metric: str,
    p: int,
    seed: int,
    svd_components: int,
    normalize: bool,
) -> np.ndarray:
    try:
        import cupy as cp
        from cuml.neighbors import KNeighborsClassifier as cuKNN
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "GPU mode requires cuML + CuPy. Activate the rapids env and run via micromamba."
        ) from e

    if int(svd_components) <= 0:
        raise ValueError("GPU mode requires --svd-components > 0")
    gpu_metric = _metric_for_gpu(metric, p)

    transformer = _build_transformer_pipeline(
        X_fit,
        seed=int(seed),
        svd_components=int(svd_components),
        normalize=bool(normalize),
        metric=str(metric),
    )
    Xtr = transformer.fit_transform(X_fit)
    Xte = transformer.transform(X_test)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    Xtr_g = cp.asarray(Xtr)
    Xte_g = cp.asarray(Xte)
    ytr_g = cp.asarray(y_fit.astype(np.int32))

    model = cuKNN(
        n_neighbors=int(k),
        weights=str(weights),
        metric=str(gpu_metric),
    )
    model.fit(Xtr_g, ytr_g)
    p_test = model.predict_proba(Xte_g)[:, 1]
    return cp.asnumpy(p_test).astype(float)


def _run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Pipeline,
    splits: int,
    seed: int,
    label: str,
    quiet: bool,
) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=int(splits), shuffle=True, random_state=int(seed))

    oof = np.zeros(len(X), dtype=float)
    fold_aucs: list[float] = []

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tf = time.time()
        if not quiet:
            print(f"{label} fold {fold}/{splits}: fitting on {len(tr_idx):,} rows...")
        model.fit(X_tr, y_tr)
        if not quiet:
            print(f"{label} fold {fold}/{splits}: fit done in {time.time() - tf:.1f}s; predicting...")

        tp = time.time()
        p_va = model.predict_proba(X_va)[:, 1]
        if not quiet:
            print(f"{label} fold {fold}/{splits}: predict done in {time.time() - tp:.1f}s")
        oof[va_idx] = p_va

        auc = roc_auc_score(y_va, p_va)
        fold_aucs.append(float(auc))
        if not quiet:
            print(f"{label} fold {fold}: AUC={auc:.6f}")

    overall = float(roc_auc_score(y, oof))
    mean_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"{label} OOF AUC: {overall:.6f}")
    print(f"{label} fold AUC mean±std: {mean_std[0]:.6f} ± {mean_std[1]:.6f}")
    if not quiet:
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
        "--svd-components",
        type=int,
        default=Defaults.svd_components,
        help="TruncatedSVD components applied after OHE (0 disables).",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Apply L2 normalization after preprocessing (recommended for cosine metric).",
    )
    ap.add_argument(
        "--algorithm",
        default="auto",
        help="KNN backend: auto|brute|kd_tree|ball_tree. For sparse/cosine, brute is often best.",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Use GPU KNN via cuML (requires running under the rapids-knn micromamba env). "
            "Note: requires --svd-components > 0."
        ),
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=Defaults.sample_n,
        help="Use at most N rows (stratified) for CV+fit. Set 0 for full data (may be very slow).",
    )
    ap.add_argument(
        "--fit-full-train",
        action="store_true",
        help=(
            "Fit the final submission model on the full training set even if CV ran on a sample. "
            "Warning: may be very slow for KNN."
        ),
    )
    ap.add_argument(
        "--cv-only",
        action="store_true",
        help="Run CV only; do not fit full and write submission",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging (no per-fold timing/AUC lines).",
    )
    ap.add_argument("--out", default=Defaults.submission_path)
    args = ap.parse_args()

    train_path = _abs_path(args.train)
    test_path = _abs_path(args.test)
    out_path = _abs_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Loading train: {train_path.as_posix()}")
    train = pd.read_csv(train_path)
    if not args.quiet:
        print(f"Loading test:  {test_path.as_posix()}")
    test = pd.read_csv(test_path)
    if not args.quiet:
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

    if not args.quiet:
        print(f"Using {sample_note} for CV/fit")

    def make_model(k: int) -> Pipeline:
        return _build_knn_pipeline(
            X,
            k=int(k),
            weights=str(args.weights),
            metric=str(args.metric),
            p=int(args.p),
            n_jobs=-1,
            seed=int(args.seed),
            svd_components=int(args.svd_components),
            normalize=bool(args.normalize),
            algorithm=str(args.algorithm),
        )

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
                if args.gpu:
                    overall_auc, _ = _run_cv_gpu(
                        X,
                        y,
                        k=int(k),
                        weights=str(args.weights),
                        metric=str(args.metric),
                        p=int(args.p),
                        splits=int(args.splits),
                        seed=int(args.seed),
                        svd_components=int(args.svd_components),
                        normalize=bool(args.normalize),
                        label=label,
                        quiet=bool(args.quiet),
                    )
                else:
                    model = make_model(k)
                    overall_auc, _ = _run_cv(
                        X,
                        y,
                        model,
                        splits=int(args.splits),
                        seed=int(args.seed),
                        label=label,
                        quiet=bool(args.quiet),
                    )
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
        if args.gpu:
            _run_cv_gpu(
                X,
                y,
                k=int(best_k),
                weights=str(args.weights),
                metric=str(args.metric),
                p=int(args.p),
                splits=int(args.splits),
                seed=int(args.seed),
                svd_components=int(args.svd_components),
                normalize=bool(args.normalize),
                label=f"[k={best_k}]",
                quiet=bool(args.quiet),
            )
        else:
            model = make_model(best_k)
            _run_cv(
                X,
                y,
                model,
                splits=int(args.splits),
                seed=int(args.seed),
                label=f"[k={best_k}]",
                quiet=bool(args.quiet),
            )

    if args.cv_only:
        print("--cv-only set; skipping submission write")
        return

    # Fit on either the sampled data (default) or full training.
    if args.fit_full_train:
        X_fit = X_full
        y_fit = y_full
        fit_note = f"full {len(X_fit):,} rows"
    else:
        X_fit = X
        y_fit = y
        fit_note = sample_note

    print(f"Fitting final model (k={best_k}) on {fit_note}...")

    # If fitting on full, rebuild the pipeline with the full column view.
    model = _build_knn_pipeline(
        X_fit,
        k=int(best_k),
        weights=str(args.weights),
        metric=str(args.metric),
        p=int(args.p),
        n_jobs=-1,
        seed=int(args.seed),
        svd_components=int(args.svd_components),
        normalize=bool(args.normalize),
        algorithm=str(args.algorithm),
    )
    if args.gpu:
        t_pred = time.time()
        p_test = _fit_predict_gpu(
            X_fit,
            y_fit,
            X_test,
            k=int(best_k),
            weights=str(args.weights),
            metric=str(args.metric),
            p=int(args.p),
            seed=int(args.seed),
            svd_components=int(args.svd_components),
            normalize=bool(args.normalize),
        )
        print(f"GPU fit+predict done in {time.time() - t_pred:.1f}s")
    else:
        t_fit = time.time()
        model.fit(X_fit, y_fit)
        print(f"Final fit done in {time.time() - t_fit:.1f}s; predicting test...")
        t_pred = time.time()
        p_test = model.predict_proba(X_test)[:, 1]
        print(f"Test predict done in {time.time() - t_pred:.1f}s")

    # If output path doesn't encode k/sample, make it informative.
    out_path_final = out_path
    if out_path_final.name == Path(Defaults.submission_path).name:
        gpu_tag = "_gpu" if args.gpu else ""
        suffix = (
            f"knn{gpu_tag}_k{best_k}_n{len(X_fit):d}_svd{int(args.svd_components):d}_"
            f"{str(args.metric)}.csv"
        )
        out_path_final = out_path_final.with_name(suffix)

    sub = pd.DataFrame({args.id_col: test[args.id_col].astype(int), args.target: p_test.astype(float)})
    sub.to_csv(out_path_final, index=False)
    print(f"Wrote submission: {out_path_final.as_posix()}  (n={len(sub)})")


if __name__ == "__main__":
    main()
