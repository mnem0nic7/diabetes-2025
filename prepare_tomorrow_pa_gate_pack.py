import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Defaults:
    data_dir: str = "data"
    target: str = "diagnosed_diabetes"

    # Anchor submissions
    base_sub: str = "scratch/blend_cutoff80_v16orig19.csv"
    gated_sub: str = "submission_v20_weighted_gated.csv"

    # Candidates to gate/blend (comma-separated)
    candidates: str = ",".join(
        [
            "submission_v23_stack_advanced.csv",
            "submission_v22_recon_weighted_lgb.csv",
            "submission_v18_multislice.csv",
            "submission_v16_drop6_pl05_heavy3.csv",
            "submission_v17_lgb_orig_mix.csv",
            "submission_v21_autogluon_full.csv",
        ]
    )

    # Optional diversity submission
    xgb_sub: str = ""

    out_dir: str = "scratch"
    manifest: str = "tomorrow_manifest_pa.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _read_sub(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    df = df[["id", target]].copy()
    df["id"] = df["id"].astype(int)
    return df


def _safe_slug(path: str) -> str:
    name = os.path.basename(path)
    if name.lower().endswith(".csv"):
        name = name[:-4]
    for ch in [" ", "\t", "\n", "\r"]:
        name = name.replace(ch, "")
    return name


def _write_manifest_row(writer: csv.DictWriter, file: str, recipe: str) -> None:
    writer.writerow({"file": file.replace("\\", "/"), "recipe": recipe})


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a PA-gated candidate pack (constant blends + PA-sigmoid gating) "
            "against a base and a gated anchor."
        )
    )
    p.add_argument("--data-dir", default=Defaults.data_dir)
    p.add_argument("--target", default=Defaults.target)
    p.add_argument("--base-sub", default=Defaults.base_sub)
    p.add_argument("--gated-sub", default=Defaults.gated_sub)
    p.add_argument(
        "--candidates",
        default=Defaults.candidates,
        help="Comma-separated candidate submission CSVs to blend/gate.",
    )
    p.add_argument(
        "--xgb-sub",
        default=Defaults.xgb_sub,
        help="Optional extra submission to use in 3-way blends if provided and exists.",
    )
    p.add_argument("--out-dir", default=Defaults.out_dir)
    p.add_argument("--manifest", default=Defaults.manifest)
    args = p.parse_args()

    candidates = [s.strip() for s in (args.candidates or "").split(",") if s.strip()]
    if not candidates:
        raise SystemExit("No candidates provided")

    required = [args.base_sub, args.gated_sub]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    for c in candidates:
        if not os.path.exists(c):
            raise SystemExit(f"Missing candidate file: {c}")

    test_path = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise SystemExit(f"Missing: {test_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    base = _read_sub(args.base_sub, args.target)
    gated = _read_sub(args.gated_sub, args.target)

    xgb = None
    if args.xgb_sub and os.path.exists(args.xgb_sub):
        xgb = _read_sub(args.xgb_sub, args.target)

    test_pa = pd.read_csv(test_path, usecols=["id", "physical_activity_minutes_per_week"]).rename(
        columns={"physical_activity_minutes_per_week": "pa"}
    )
    test_pa["id"] = test_pa["id"].astype(int)

    # Keep grids modest by default.
    slopes = [4.0, 5.0, 6.0]
    q_his = [0.85, 0.90]

    pa = test_pa["pa"].to_numpy(dtype=float)
    q50 = float(np.nanquantile(pa, 0.50))

    base_slug = _safe_slug(args.base_sub)
    gated_slug = _safe_slug(args.gated_sub)

    # Manifest
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "recipe"])
        writer.writeheader()

        for cand_path in candidates:
            cand = _read_sub(cand_path, args.target)
            cand_slug = _safe_slug(cand_path)

            merged = cand.merge(base, on="id", suffixes=("", "_base"))
            merged = merged.merge(gated, on="id", suffixes=("", "_gated"))
            merged = merged.merge(test_pa, on="id")
            if xgb is not None:
                merged = merged.merge(xgb, on="id", suffixes=("", "_xgb"))

            c = merged[args.target].to_numpy(dtype=float)
            b = merged[f"{args.target}_base"].to_numpy(dtype=float)
            g = merged[f"{args.target}_gated"].to_numpy(dtype=float)
            x = None
            if xgb is not None:
                x = merged[f"{args.target}_xgb"].to_numpy(dtype=float)

            # Constant blends (keep small)
            for w in [0.80, 0.85, 0.90]:
                w2 = 1.0 - w

                out = os.path.join(args.out_dir, f"tomorrow_pa_lin_{cand_slug}_vs_{base_slug}_{int(w*100)}_{int(w2*100)}.csv")
                pred = w * c + w2 * b
                pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
                _write_manifest_row(writer, out, f"{w:.2f}*{cand_slug} + {w2:.2f}*{base_slug}")

                out = os.path.join(args.out_dir, f"tomorrow_pa_lin_{cand_slug}_vs_{gated_slug}_{int(w*100)}_{int(w2*100)}.csv")
                pred = w * c + w2 * g
                pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
                _write_manifest_row(writer, out, f"{w:.2f}*{cand_slug} + {w2:.2f}*{gated_slug}")

            # PA-gated blends (sweep slope + quantile anchor)
            for slope in slopes:
                for q_hi in q_his:
                    qh = float(np.nanquantile(pa, q_hi))
                    scale = max(1e-6, qh - q50)
                    z = (pa - q50) / scale
                    gate = _sigmoid(float(slope) * z)
                    gate_name = f"pa_gate_s{slope:g}_q{int(q_hi*100)}"

                    out = os.path.join(args.out_dir, f"tomorrow_{gate_name}_{cand_slug}_vs_{base_slug}.csv")
                    pred = gate * c + (1.0 - gate) * b
                    pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
                    _write_manifest_row(writer, out, f"gate(PA,{gate_name})*{cand_slug} + (1-gate)*{base_slug}")

                    out = os.path.join(args.out_dir, f"tomorrow_{gate_name}_{cand_slug}_vs_{gated_slug}.csv")
                    pred = gate * c + (1.0 - gate) * g
                    pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
                    _write_manifest_row(writer, out, f"gate(PA,{gate_name})*{cand_slug} + (1-gate)*{gated_slug}")

            # Simple 3-way constant blend
            out = os.path.join(args.out_dir, f"tomorrow_pa_3way_{cand_slug}_055_030_015.csv")
            pred = 0.55 * c + 0.30 * g + 0.15 * b
            pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
            _write_manifest_row(writer, out, f"0.55*{cand_slug} + 0.30*{gated_slug} + 0.15*{base_slug}")

            if x is not None:
                out = os.path.join(args.out_dir, f"tomorrow_pa_3way_{cand_slug}_055_030_015_xgb.csv")
                pred = 0.55 * c + 0.30 * g + 0.15 * x
                pd.DataFrame({"id": merged["id"].astype(int), args.target: pred}).to_csv(out, index=False)
                _write_manifest_row(writer, out, f"0.55*{cand_slug} + 0.30*{gated_slug} + 0.15*xgb")

    print("Wrote manifest:", args.manifest)


if __name__ == "__main__":
    main()
