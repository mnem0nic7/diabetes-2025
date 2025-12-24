import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Defaults:
    target: str = "diagnosed_diabetes"
    out_dir: str = "submissions"
    manifest: str = "tomorrow_manifest_v2.csv"
    # By convention: first is "best" / anchor.
    subs: str = ",".join(
        [
            "submissions/blend_cutoff80_v16orig19.csv",
            "submissions/submission_v23_stack_advanced.csv",
            "submissions/submission_v21_autogluon_full.csv",
            "submissions/submission_v18_multislice.csv",
            "submissions/submission_v16_drop6_pl05_heavy3.csv",
            "submissions/submission_v17_lgb_orig_mix.csv",
            "submissions/submission_v22_recon_weighted_lgb.csv",
            "submissions/submission_knn_gpu_k201_full.csv",
        ]
    )
    weights: str = "0.90,0.85,0.80"
    power: int = 10


def _read_sub(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    df = df[["id", target]].copy()
    df["id"] = df["id"].astype(int)
    return df


def _assert_same_ids(frames: list[pd.DataFrame]) -> None:
    base_ids = frames[0]["id"].to_numpy()
    for i, df in enumerate(frames[1:], start=2):
        ids = df["id"].to_numpy()
        if len(ids) != len(base_ids) or not np.array_equal(ids, base_ids):
            raise ValueError(f"ID mismatch between submission[1] and submission[{i}]")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _safe_slug(path: str) -> str:
    name = os.path.basename(path)
    name = name.replace(".csv", "")
    for ch in [" ", "\t", "\n", "\r"]:
        name = name.replace(ch, "")
    return name


def _write_manifest(path: str, rows: list[dict[str, str]], append: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existing: dict[str, dict[str, str]] = {}
    if append and os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("file"):
                    existing[r["file"]] = r

    for r in rows:
        existing[r["file"]] = r

    merged_rows = list(existing.values())
    merged_rows.sort(key=lambda r: r["file"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "recipe"])
        writer.writeheader()
        writer.writerows(merged_rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate a pack of blend submissions (linear/logitavg/rankavg/power-mean) "
            "and write a manifest for tomorrow's submits."
        )
    )
    p.add_argument("--target", default=Defaults.target)
    p.add_argument("--subs", default=Defaults.subs, help="Comma-separated submission CSV paths; first is the anchor")
    p.add_argument("--weights", default=Defaults.weights, help="Comma-separated anchor weights for linear blends")
    p.add_argument("--power", type=int, default=Defaults.power, help="Exponent for power-mean blend")
    p.add_argument("--out-dir", default=Defaults.out_dir)
    p.add_argument("--manifest", default=Defaults.manifest)
    p.add_argument("--append", action="store_true", help="Append/merge into an existing manifest instead of overwriting")
    args = p.parse_args()

    subs = [s.strip() for s in (args.subs or "").split(",") if s.strip()]
    if len(subs) < 2:
        raise SystemExit("Provide at least 2 submissions via --subs")

    weights = [float(w.strip()) for w in (args.weights or "").split(",") if w.strip()]
    if not weights:
        raise SystemExit("Provide at least 1 weight via --weights")

    missing = [s for s in subs if not os.path.exists(s)]
    if missing:
        raise SystemExit("Missing submission files:\n- " + "\n- ".join(missing))

    os.makedirs(args.out_dir, exist_ok=True)

    frames = [_read_sub(s, args.target) for s in subs]
    _assert_same_ids(frames)

    ids = frames[0]["id"].to_numpy(dtype=int)
    preds = [df[args.target].to_numpy(dtype=float) for df in frames]

    anchor_path = subs[0]
    anchor_slug = _safe_slug(anchor_path)

    rows: list[dict[str, str]] = []

    # 1) Rank-average across all submissions
    ranks = []
    for pvec in preds:
        # average ranks for ties
        ranks.append(pd.Series(pvec).rank(method="average").to_numpy(dtype=float))
    rankavg = np.mean(np.stack(ranks, axis=0), axis=0)
    rankavg = rankavg / (len(rankavg) + 1.0)  # map to (0,1)
    out = os.path.join(args.out_dir, f"tomorrow_rankavg_{anchor_slug}_plus_{len(subs)-1}.csv")
    pd.DataFrame({"id": ids, args.target: rankavg}).to_csv(out, index=False)
    rows.append({"file": out.replace("\\", "/"), "recipe": f"rankavg({', '.join([_safe_slug(s) for s in subs])})"})

    # 2) Pairwise blends versus anchor
    a = preds[0]
    for other_path, b in zip(subs[1:], preds[1:]):
        other_slug = _safe_slug(other_path)

        # Linear blends
        for w in weights:
            w_other = 1.0 - w
            out = os.path.join(args.out_dir, f"tomorrow_lin_{anchor_slug}_{other_slug}_{int(w*100):02d}_{int(w_other*100):02d}.csv")
            y = w * a + w_other * b
            pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
            rows.append({"file": out.replace("\\", "/"), "recipe": f"{w:.2f}*{anchor_slug} + {w_other:.2f}*{other_slug}"})

        # Logit-average
        out = os.path.join(args.out_dir, f"tomorrow_logitavg_{anchor_slug}_{other_slug}.csv")
        y = _sigmoid(0.5 * (_logit(a) + _logit(b)))
        pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
        rows.append({"file": out.replace("\\", "/"), "recipe": f"logitavg({anchor_slug}, {other_slug})"})

        # Power-mean (p-norm mean) blend (tends to upweight confident preds)
        out = os.path.join(args.out_dir, f"tomorrow_pmean_p{args.power}_{anchor_slug}_{other_slug}.csv")
        p = float(args.power)
        y = ((np.power(a, p) + np.power(b, p)) / 2.0) ** (1.0 / p)
        pd.DataFrame({"id": ids, args.target: y}).to_csv(out, index=False)
        rows.append({"file": out.replace("\\", "/"), "recipe": f"pmean(p={args.power}; {anchor_slug}, {other_slug})"})

    _write_manifest(args.manifest, rows, append=bool(args.append))
    print(f"Wrote {len(rows)} candidates")
    print("Manifest:", args.manifest)


if __name__ == "__main__":
    main()
