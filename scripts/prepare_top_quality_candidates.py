import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Defaults:
    target: str = "diagnosed_diabetes"
    out_dir: str = "submissions"
    manifest: str = "submissions/tomorrow_manifest_top_quality.csv"

    # Local AUC priors sources (higher is better).
    priors: str = "submissions/unsubmitted_candidates.csv,submissions/shortlist_rank.csv,scratch/unsubmitted_candidates.csv,scratch/shortlist_rank.csv"

    # How many base models to consider.
    top_k: int = 8

    # Blend configs
    pair_weights: str = "0.93,0.90,0.87"


def _abs_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _read_submission(path: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format (need id,{target}): {path}")
    out = df[["id", target]].copy()
    out["id"] = out["id"].astype(int)
    out[target] = out[target].astype(float)
    return out


def _assert_same_ids(frames: list[pd.DataFrame]) -> None:
    base = frames[0]["id"].to_numpy()
    for i, df in enumerate(frames[1:], start=2):
        ids = df["id"].to_numpy()
        if len(ids) != len(base) or not np.array_equal(ids, base):
            raise ValueError(f"ID mismatch between submission[1] and submission[{i}]")


def _safe_slug(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    for ch in [" ", "\t", "\n", "\r"]:
        name = name.replace(ch, "")
    return name


def _load_auc_priors(paths: list[Path]) -> dict[str, float]:
    """Return filename -> best metric_value (AUC)."""
    priors: dict[str, float] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        if "path" not in df.columns or "metric_value" not in df.columns:
            continue

        for _, row in df.iterrows():
            raw = str(row.get("path") or "")
            if not raw:
                continue
            try:
                val = float(row.get("metric_value"))
            except Exception:
                continue
            base = Path(raw).name
            if not base:
                continue
            if base not in priors:
                priors[base] = val
            else:
                priors[base] = max(priors[base], val)

    return priors


def _resolve_candidate_paths(priors: dict[str, float]) -> list[Path]:
    # Prefer submissions/, then root-level file, then scratch/.
    resolved: list[tuple[float, Path]] = []
    for base, auc in priors.items():
        if not base.lower().endswith(".csv"):
            continue

        p0 = (REPO_ROOT / "submissions" / base)
        p1 = (REPO_ROOT / base)
        p2 = (REPO_ROOT / "scratch" / base)
        if p0.exists():
            resolved.append((auc, p0))
        elif p1.exists():
            resolved.append((auc, p1))
        elif p2.exists():
            resolved.append((auc, p2))

    resolved.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in resolved]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "recipe"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate top-quality candidate submissions from local AUC priors.")
    p.add_argument("--target", default=Defaults.target)
    p.add_argument("--priors", default=Defaults.priors, help="Comma-separated CSVs with columns path,metric_value")
    p.add_argument("--top-k", type=int, default=Defaults.top_k)
    p.add_argument("--pair-weights", default=Defaults.pair_weights)
    p.add_argument("--out-dir", default=Defaults.out_dir)
    p.add_argument("--manifest", default=Defaults.manifest)
    args = p.parse_args()

    priors_paths = [_abs_path(s.strip()) for s in (args.priors or "").split(",") if s.strip()]
    priors = _load_auc_priors(priors_paths)
    if not priors:
        raise SystemExit(
            "No AUC priors found. Expected scratch/unsubmitted_candidates.csv or scratch/shortlist_rank.csv with columns path,metric_value."
        )

    candidates = _resolve_candidate_paths(priors)
    if not candidates:
        raise SystemExit("No submission CSVs referenced by priors exist on disk.")

    top_k = max(2, int(args.top_k))
    bases = candidates[:top_k]

    missing = [str(p) for p in bases if not p.exists()]
    if missing:
        raise SystemExit("Missing base submission files:\n- " + "\n- ".join(missing))

    os.makedirs(args.out_dir, exist_ok=True)

    frames = [_read_submission(pth, args.target) for pth in bases]
    _assert_same_ids(frames)

    ids = frames[0]["id"].to_numpy(dtype=int)
    preds = [df[args.target].to_numpy(dtype=float) for df in frames]
    slugs = [_safe_slug(pth) for pth in bases]

    anchor_slug = slugs[0]
    anchor_pred = preds[0]

    weights = [float(x.strip()) for x in (args.pair_weights or "").split(",") if x.strip()]
    if not weights:
        weights = [0.90]

    rows: list[dict[str, str]] = []

    # 1) Rank-average over top_k
    ranks = [pd.Series(pv).rank(method="average").to_numpy(dtype=float) for pv in preds]
    rankavg = np.mean(np.stack(ranks, axis=0), axis=0)
    rankavg = rankavg / (len(rankavg) + 1.0)
    out = Path(args.out_dir) / f"topq_rankavg_{anchor_slug}_plus_{len(bases)-1}.csv"
    pd.DataFrame({"id": ids, args.target: rankavg}).to_csv(out, index=False)
    rows.append({"file": out.as_posix(), "recipe": f"rankavg({', '.join(slugs)})"})

    # 2) Trimmed mean (drop min/max across top_k) mapped to (0,1)
    stack = np.stack(preds, axis=0)
    if stack.shape[0] >= 4:
        sorted_stack = np.sort(stack, axis=0)
        trimmed = np.mean(sorted_stack[1:-1, :], axis=0)
    else:
        trimmed = np.mean(stack, axis=0)
    trimmed = np.clip(trimmed, 1e-6, 1.0 - 1e-6)
    out = Path(args.out_dir) / f"topq_trimmean_{anchor_slug}_k{len(bases)}.csv"
    pd.DataFrame({"id": ids, args.target: trimmed}).to_csv(out, index=False)
    rows.append({"file": out.as_posix(), "recipe": f"trimmean({', '.join(slugs)})"})

    # 3) Pairwise blends vs anchor (linear + logitavg + pmean)
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def logit(pv: np.ndarray) -> np.ndarray:
        pv = np.clip(pv, 1e-6, 1.0 - 1e-6)
        return np.log(pv / (1.0 - pv))

    for other_slug, other_pred in zip(slugs[1:], preds[1:]):
        for w in weights:
            w = float(w)
            w2 = 1.0 - w
            out = Path(args.out_dir) / f"topq_lin_{anchor_slug}_{other_slug}_{int(w*100):02d}_{int(w2*100):02d}.csv"
            y = w * anchor_pred + w2 * other_pred
            pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
            rows.append({"file": out.as_posix(), "recipe": f"{w:.2f}*{anchor_slug} + {w2:.2f}*{other_slug}"})

        out = Path(args.out_dir) / f"topq_logitavg_{anchor_slug}_{other_slug}.csv"
        y = sigmoid(0.5 * (logit(anchor_pred) + logit(other_pred)))
        pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
        rows.append({"file": out.as_posix(), "recipe": f"logitavg({anchor_slug}, {other_slug})"})

    # 4) Small 3-way blends for diversity if we have >=3
    if len(preds) >= 3:
        a, b, c = preds[0], preds[1], preds[2]
        sA, sB, sC = slugs[0], slugs[1], slugs[2]

        out = Path(args.out_dir) / f"topq_3way_{sA}_{sB}_{sC}_050_030_020.csv"
        y = 0.50 * a + 0.30 * b + 0.20 * c
        pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
        rows.append({"file": out.as_posix(), "recipe": f"0.50*{sA} + 0.30*{sB} + 0.20*{sC}"})

        out = Path(args.out_dir) / f"topq_3way_{sA}_{sB}_{sC}_060_020_020.csv"
        y = 0.60 * a + 0.20 * b + 0.20 * c
        pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
        rows.append({"file": out.as_posix(), "recipe": f"0.60*{sA} + 0.20*{sB} + 0.20*{sC}"})

    manifest_path = _abs_path(args.manifest)
    _write_manifest(manifest_path, rows)

    print(f"Bases (top {len(bases)} by local AUC priors):")
    for pth in bases:
        print(f"- {pth.as_posix()}")
    print(f"Wrote {len(rows)} candidates -> {manifest_path.as_posix()}")


if __name__ == "__main__":
    main()
