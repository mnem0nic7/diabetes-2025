from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _rank01(p: np.ndarray) -> np.ndarray:
    # Stable rank -> (0,1). Average ranks for ties.
    # We avoid scipy dependency by using argsort twice; ties get arbitrary ordering,
    # so for exact tie handling we'd need pandas. For model preds ties are rare.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    return ranks / (len(p) + 1.0)


@dataclass(frozen=True)
class PredArtifact:
    path: str
    name: str
    train_id: np.ndarray
    y: np.ndarray
    oof: np.ndarray
    test_id: np.ndarray
    test: np.ndarray


def load_pred_artifact(path: str) -> PredArtifact:
    z = np.load(path, allow_pickle=True)

    def _req(k: str) -> np.ndarray:
        if k not in z:
            raise ValueError(f"Missing key {k!r} in {path}")
        return z[k]

    name = str(z["name"].item()) if "name" in z else os.path.basename(path)

    train_id = _req("train_id").astype(np.int64)
    y = _req("y").astype(np.int64)
    oof = _req("oof").astype(np.float64)
    test_id = _req("test_id").astype(np.int64)
    test = _req("test").astype(np.float64)

    if len(train_id) != len(y) or len(y) != len(oof):
        raise ValueError(f"Train arrays length mismatch in {path}")
    if len(test_id) != len(test):
        raise ValueError(f"Test arrays length mismatch in {path}")

    return PredArtifact(
        path=path,
        name=name,
        train_id=train_id,
        y=y,
        oof=oof,
        test_id=test_id,
        test=test,
    )


def _align_artifacts(arts: list[PredArtifact]) -> list[PredArtifact]:
    if not arts:
        return arts

    base_train = arts[0].train_id
    base_test = arts[0].test_id

    out: list[PredArtifact] = []
    for a in arts:
        if len(a.train_id) != len(base_train) or not np.array_equal(a.train_id, base_train):
            raise ValueError(
                "Train id mismatch across artifacts. Ensure all models used the same training rows/order. "
                f"Mismatch at: {a.path}"
            )
        if len(a.test_id) != len(base_test) or not np.array_equal(a.test_id, base_test):
            raise ValueError(
                "Test id mismatch across artifacts. Ensure all models predicted the same test rows/order. "
                f"Mismatch at: {a.path}"
            )
        if len(a.y) != len(arts[0].y) or not np.array_equal(a.y, arts[0].y):
            raise ValueError(f"Target y mismatch across artifacts: {a.path}")
        out.append(a)

    return out


def transform_preds(p: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "raw").lower()
    if mode == "raw":
        return p
    if mode == "logit":
        return _logit(p)
    if mode == "rank":
        return _rank01(p)
    raise ValueError(f"Unknown mode: {mode!r} (expected raw|logit|rank)")


def inverse_transform_blend(z: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "raw").lower()
    if mode == "raw":
        return np.clip(z, 0.0, 1.0)
    if mode == "logit":
        return _sigmoid(z)
    if mode == "rank":
        # already in (0,1)
        return np.clip(z, 0.0, 1.0)
    raise ValueError(f"Unknown mode: {mode!r} (expected raw|logit|rank)")


def hillclimb_weights(
    preds: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 30,
    step: float = 0.05,
    positive: bool = True,
    start_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Optimize blend weights to maximize ROC-AUC on OOF.

    preds: shape (n_models, n_samples)
    """

    if preds.ndim != 2:
        raise ValueError("preds must be 2D (n_models, n_samples)")

    n_models, n_samples = preds.shape
    if len(y) != n_samples:
        raise ValueError("y length mismatch")

    rng = np.random.default_rng(seed)

    if start_weights is None:
        w = np.ones(n_models, dtype=np.float64) / n_models
    else:
        w = np.asarray(start_weights, dtype=np.float64)
        if w.shape != (n_models,):
            raise ValueError(f"start_weights must have shape ({n_models},)")
        if positive:
            w = np.clip(w, 0.0, None)
        s = float(w.sum())
        w = (w / s) if s > 0 else (np.ones(n_models, dtype=np.float64) / n_models)

    best_pred = np.dot(w, preds)
    best_auc = float(roc_auc_score(y, best_pred))

    # Coordinate hill-climb with decreasing step.
    for e in range(max(1, int(epochs))):
        frac = 1.0 - (e / max(1.0, float(epochs - 1)))
        cur_step = float(step) * (0.2 + 0.8 * frac)

        n_trials = 200 * n_models
        for _ in range(n_trials):
            i = int(rng.integers(0, n_models))
            delta = float(rng.choice([-1.0, 1.0])) * cur_step

            w2 = w.copy()
            w2[i] += delta

            if positive:
                if w2[i] < 0:
                    continue

            s = float(w2.sum())
            if s <= 0:
                continue
            w2 /= s

            pred2 = np.dot(w2, preds)
            auc2 = float(roc_auc_score(y, pred2))
            if auc2 > best_auc:
                w = w2
                best_auc = auc2
                best_pred = pred2

    return w, best_auc


def format_weights(names: Iterable[str], weights: np.ndarray) -> str:
    items = [{"name": n, "w": float(w)} for n, w in zip(names, weights)]
    return json.dumps(items, sort_keys=False)


def oof_optimize_and_blend(
    artifacts: list[PredArtifact],
    *,
    mode: str = "raw",
    seed: int = 42,
    epochs: int = 30,
    step: float = 0.05,
    positive: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (weights, blended_test_preds, oof_auc)."""

    artifacts = _align_artifacts(artifacts)
    names = [a.name for a in artifacts]

    oof_m = np.stack([transform_preds(a.oof, mode) for a in artifacts], axis=0)
    test_m = np.stack([transform_preds(a.test, mode) for a in artifacts], axis=0)
    y = artifacts[0].y

    w, best_auc = hillclimb_weights(oof_m, y, seed=seed, epochs=epochs, step=step, positive=positive)

    blended_test = np.dot(w, test_m)
    blended_test = inverse_transform_blend(blended_test, mode)

    return w, blended_test, best_auc
