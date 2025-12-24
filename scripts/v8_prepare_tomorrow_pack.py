import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Inputs:
    data_dir: str = "data"
    target: str = "diagnosed_diabetes"

    # Baselines to blend with
    base_sub: str = "submission_v8_base_shiftcv.csv"
    gated_sub: str = "submission_v8_gated_base_sub50_sigmoid.csv"

    # Optional: include an XGB model to increase diversity
    xgb_sub: str = "submission_xgb.csv"

    # One or more cutoff-boost submissions (comma-separated)
    cutoff_subs: tuple[str, ...] = (
        "submission_v8_cutoff_boost_w5.csv",
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _read_sub(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    return df[["id", target]].rename(columns={target: os.path.basename(path)})


def _write_manifest_row(writer: csv.DictWriter, file: str, recipe: str) -> None:
    writer.writerow({"file": file, "recipe": recipe})


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare a set of candidate submissions for tomorrow.")
    p.add_argument("--data-dir", default=Inputs.data_dir)
    p.add_argument("--target", default=Inputs.target)
    p.add_argument("--base-sub", default=Inputs.base_sub)
    p.add_argument("--gated-sub", default=Inputs.gated_sub)
    p.add_argument(
        "--xgb-sub",
        default=Inputs.xgb_sub,
        help="Optional XGB submission CSV to include in 3-way blends.",
    )
    p.add_argument(
        "--cutoff-subs",
        default=",".join(Inputs.cutoff_subs),
        help="Comma-separated list of cutoff-boost submission CSVs to blend.",
    )
    p.add_argument(
        "--manifest",
        default="tomorrow_manifest.csv",
        help="CSV manifest file describing each produced candidate.",
    )
    args = p.parse_args()

    cutoff_subs = tuple([s.strip() for s in args.cutoff_subs.split(",") if s.strip()])
    if not cutoff_subs:
        raise SystemExit("No cutoff submissions provided")

    inputs = Inputs(
        data_dir=args.data_dir,
        target=args.target,
        base_sub=args.base_sub,
        gated_sub=args.gated_sub,
        xgb_sub=args.xgb_sub,
        cutoff_subs=cutoff_subs,
    )

    base = _read_sub(inputs.base_sub, inputs.target)
    gated = _read_sub(inputs.gated_sub, inputs.target)

    xgb = None
    if inputs.xgb_sub and os.path.exists(inputs.xgb_sub):
        xgb = _read_sub(inputs.xgb_sub, inputs.target)

    test_pa = pd.read_csv(
        os.path.join(inputs.data_dir, "test.csv"),
        usecols=["id", "physical_activity_minutes_per_week"],
    ).rename(columns={"physical_activity_minutes_per_week": "pa"})

    # Gate derived only from PA (cheap, stable).
    # We sweep a small grid of (slope, q_hi) because this is what moved LB.
    pa = test_pa["pa"].to_numpy(dtype=float)
    q50 = float(np.nanquantile(pa, 0.50))

    gate_specs: list[tuple[str, float, float]] = []
    for slope in [3.0, 4.0, 5.0, 6.0]:
        for q_hi in [0.80, 0.85, 0.90]:
            gate_specs.append((f"pa_gate_s{slope:g}_q{int(q_hi*100)}", slope, q_hi))

    # Manifest
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "recipe"])
        writer.writeheader()

        for cutoff_sub in inputs.cutoff_subs:
            cutoff = _read_sub(cutoff_sub, inputs.target)

            merged = cutoff.merge(base, on="id").merge(gated, on="id").merge(test_pa, on="id")
            if xgb is not None:
                merged = merged.merge(xgb, on="id")

            c = merged[os.path.basename(cutoff_sub)].to_numpy()
            b = merged[os.path.basename(inputs.base_sub)].to_numpy()
            g = merged[os.path.basename(inputs.gated_sub)].to_numpy()
            x = None
            if xgb is not None:
                x = merged[os.path.basename(inputs.xgb_sub)].to_numpy()

            # Constant blends (small set)
            for w in [0.6, 0.7, 0.8]:
                name = f"submission_tomorrow_cutoff_{os.path.basename(cutoff_sub).replace('.csv','')}_base_{int(w*100)}_{int((1-w)*100)}.csv"
                pred = w * c + (1 - w) * b
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, f"{w}*cutoff + {1-w}*base")

                name = f"submission_tomorrow_cutoff_{os.path.basename(cutoff_sub).replace('.csv','')}_gated_{int(w*100)}_{int((1-w)*100)}.csv"
                pred = w * c + (1 - w) * g
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, f"{w}*cutoff + {1-w}*gated")

            # PA-gated blends (sweep slope + quantile anchor)
            for gate_name, slope, q_hi in gate_specs:
                qh = float(np.nanquantile(pa, q_hi))
                scale = max(1e-6, qh - q50)
                z = (pa - q50) / scale
                gate = _sigmoid(slope * z)

                name = f"submission_tomorrow_{gate_name}_{os.path.basename(cutoff_sub).replace('.csv','')}_vs_base.csv"
                pred = gate * c + (1 - gate) * b
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, f"gate(PA,{gate_name})*cutoff + (1-gate)*base")

                name = f"submission_tomorrow_{gate_name}_{os.path.basename(cutoff_sub).replace('.csv','')}_vs_gated.csv"
                pred = gate * c + (1 - gate) * g
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, f"gate(PA,{gate_name})*cutoff + (1-gate)*gated")

            # Simple 3-way constant blend (keep very small)
            # (cutoff, gated, base) = (0.55, 0.30, 0.15)
            name = f"submission_tomorrow_3way_{os.path.basename(cutoff_sub).replace('.csv','')}_055_030_015.csv"
            pred = 0.55 * c + 0.30 * g + 0.15 * b
            pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
            _write_manifest_row(writer, name, "0.55*cutoff + 0.30*gated + 0.15*base")

            # XGB-including 3-way blends (for diversity)
            if x is not None:
                name = f"submission_tomorrow_3way_{os.path.basename(cutoff_sub).replace('.csv','')}_055_030_015_xgb.csv"
                pred = 0.55 * c + 0.30 * g + 0.15 * x
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, "0.55*cutoff + 0.30*gated + 0.15*xgb")

                name = f"submission_tomorrow_3way_{os.path.basename(cutoff_sub).replace('.csv','')}_050_030_020_xgb.csv"
                pred = 0.50 * c + 0.30 * g + 0.20 * x
                pd.DataFrame({"id": merged["id"].astype(int), inputs.target: pred}).to_csv(name, index=False)
                _write_manifest_row(writer, name, "0.50*cutoff + 0.30*gated + 0.20*xgb")

    print("Wrote manifest:", args.manifest)


if __name__ == "__main__":
    main()
