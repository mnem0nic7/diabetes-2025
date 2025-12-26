import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oof_blend_utils import load_pred_artifact, oof_optimize_and_blend
from segment_rank_blend import nina_delta_matrix_m4, parse_segments, segment_rank_blend


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
    oof_artifacts: str = ""
    oof_mode: str = "raw"
    oof_seed: int = 42
    oof_epochs: int = 30
    oof_step: float = 0.05

    segment_rank: bool = False
    segment_rank_k: int = 4
    segment_rank_segments: str = "0.00003:0.00177,0.00180:0.00293,0.00300:0.00474"
    segment_rank_different: str = "0,1,2,3"
    segment_rank_asc: float = 0.30
    segment_rank_desc: float = 0.70


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

    p.add_argument(
        "--oof-artifacts",
        default=Defaults.oof_artifacts,
        help=(
            "Optional comma-separated list of .npz prediction artifacts to create an OOF-optimized blend. "
            "Each artifact must contain train_id,y,oof,test_id,test,name."
        ),
    )
    p.add_argument("--oof-mode", default=Defaults.oof_mode, choices=["raw", "logit", "rank"], help="Blend space")
    p.add_argument("--oof-seed", type=int, default=Defaults.oof_seed)
    p.add_argument("--oof-epochs", type=int, default=Defaults.oof_epochs)
    p.add_argument("--oof-step", type=float, default=Defaults.oof_step)

    p.add_argument(
        "--segment-rank",
        action="store_true",
        help=(
            "Optional: write a Nina-style rank-conditioned blend candidate using the first K submissions "
            "(starting from the anchor)."
        ),
    )
    p.add_argument("--segment-rank-k", type=int, default=Defaults.segment_rank_k)
    p.add_argument("--segment-rank-segments", default=Defaults.segment_rank_segments)
    p.add_argument("--segment-rank-different", default=Defaults.segment_rank_different)
    p.add_argument("--segment-rank-asc", type=float, default=Defaults.segment_rank_asc)
    p.add_argument("--segment-rank-desc", type=float, default=Defaults.segment_rank_desc)
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

    # 3) OOF-optimized blend across prediction artifacts (optional)
    if args.oof_artifacts:
        paths = [s.strip() for s in (args.oof_artifacts or "").split(",") if s.strip()]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit("Missing OOF artifact files:\n- " + "\n- ".join(missing))
        if len(paths) < 2:
            raise SystemExit("Provide at least 2 artifacts via --oof-artifacts")

        arts = [load_pred_artifact(pth) for pth in paths]
        w, blended_test, best_auc = oof_optimize_and_blend(
            arts,
            mode=args.oof_mode,
            seed=int(args.oof_seed),
            epochs=int(args.oof_epochs),
            step=float(args.oof_step),
            positive=True,
        )

        # Write as another tomorrow candidate.
        art_slugs = [_safe_slug(a.name) for a in arts]
        name = f"tomorrow_oofw_{args.oof_mode}_{len(arts)}m_seed{int(args.oof_seed)}.csv"
        out = os.path.join(args.out_dir, name)
        pd.DataFrame({"id": ids, args.target: blended_test}).to_csv(out, index=False)

        w_parts = [f"{wi:.3f}*{slug}" for wi, slug in zip(w, art_slugs)]
        recipe = f"oofw_auc={best_auc:.6f} mode={args.oof_mode} :: " + " + ".join(w_parts)
        rows.append({"file": out.replace("\\", "/"), "recipe": recipe})

    # 4) Nina-style rank-conditioned blend (optional)
    if bool(args.segment_rank):
        k = int(args.segment_rank_k)
        if k < 2:
            raise SystemExit("--segment-rank-k must be >= 2")
        if len(subs) < k:
            raise SystemExit(f"Need at least {k} submissions in --subs for --segment-rank")

        if k != 4:
            raise SystemExit("Currently, --segment-rank requires K=4 (matches Nina's delta matrix)")

        seg_subs = subs[:k]
        seg_frames = [_read_sub(s, args.target) for s in seg_subs]
        _assert_same_ids(seg_frames)
        seg_preds = np.stack([df[args.target].to_numpy(dtype=float) for df in seg_frames], axis=0)

        base_w = np.full(k, 1.0 / k, dtype=float)
        delta = nina_delta_matrix_m4()
        segments = parse_segments(str(args.segment_rank_segments))
        different = [int(x.strip()) for x in str(args.segment_rank_different).split(",") if x.strip()]
        if len(different) != 4:
            raise SystemExit("--segment-rank-different must be 4 ints")

        asc = float(args.segment_rank_asc)
        desc = float(args.segment_rank_desc)
        if abs((asc + desc) - 1.0) > 1e-6:
            raise SystemExit("--segment-rank-asc + --segment-rank-desc must sum to 1.0")

        y = segment_rank_blend(
            preds=seg_preds,
            base_weights=base_w,
            delta_matrix=delta,
            segments=segments,
            different=different,
            asc_weight=asc,
            desc_weight=desc,
        )

        out = os.path.join(args.out_dir, f"tomorrow_segrank_{anchor_slug}_plus_{k-1}.csv")
        pd.DataFrame({"id": ids, args.target: np.clip(y, 0.0, 1.0)}).to_csv(out, index=False)

        recipe = (
            f"segrank(k={k}; asc={asc:.2f} desc={desc:.2f}; segments={args.segment_rank_segments}; "
            f"different={args.segment_rank_different}; base=uniform; deltas=nina_m4; "
            f"inputs={', '.join([_safe_slug(s) for s in seg_subs])})"
        )
        rows.append({"file": out.replace("\\", "/"), "recipe": recipe})

    _write_manifest(args.manifest, rows, append=bool(args.append))
    print(f"Wrote {len(rows)} candidates")
    print("Manifest:", args.manifest)


if __name__ == "__main__":
    main()
