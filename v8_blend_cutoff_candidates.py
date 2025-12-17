import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    target: str = "diagnosed_diabetes"
    cutoff_sub: str = "submission_v8_cutoff_boost_w5.csv"
    base_sub: str = "submission_v8_base_shiftcv.csv"
    gated_sub: str = "submission_v8_gated_base_sub50_sigmoid.csv"
    out_prefix: str = "submission_v8_"


def _read_sub(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    return df[["id", target]].rename(columns={target: os.path.basename(path)})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    p = argparse.ArgumentParser(description="Blend cutoff/base/gated submissions (including optional PA-gated blends).")
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--target", default=Config.target)
    p.add_argument("--cutoff-sub", default=Config.cutoff_sub)
    p.add_argument("--base-sub", default=Config.base_sub)
    p.add_argument("--gated-sub", default=Config.gated_sub)
    p.add_argument("--out-prefix", default=Config.out_prefix)
    args = p.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        target=args.target,
        cutoff_sub=args.cutoff_sub,
        base_sub=args.base_sub,
        gated_sub=args.gated_sub,
        out_prefix=args.out_prefix,
    )

    cutoff = _read_sub(cfg.cutoff_sub, cfg.target)
    base = _read_sub(cfg.base_sub, cfg.target)
    gated = _read_sub(cfg.gated_sub, cfg.target)

    merged = cutoff.merge(base, on="id").merge(gated, on="id")

    c = merged[cfg.cutoff_sub].to_numpy()
    b = merged[cfg.base_sub].to_numpy()
    g = merged[cfg.gated_sub].to_numpy()

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    print("Correlations:")
    print(" cutoff vs base :", corr(c, b))
    print(" cutoff vs gated:", corr(c, g))
    print(" base   vs gated:", corr(b, g))

    outs = {}

    # Simple constant blends
    for w in [0.5, 0.6, 0.7, 0.8]:
        outs[f"blend_cutoff_base_{int(w*100)}_{int((1-w)*100)}"] = w * c + (1 - w) * b
        outs[f"blend_cutoff_gated_{int(w*100)}_{int((1-w)*100)}"] = w * c + (1 - w) * g

    # Gate using physical_activity_minutes_per_week (the feature used to find the shift cutoff).
    test = pd.read_csv(os.path.join(cfg.data_dir, "test.csv"), usecols=["id", "physical_activity_minutes_per_week"]) \
        .rename(columns={"physical_activity_minutes_per_week": "pa"})

    merged2 = merged.merge(test, on="id", how="left")
    pa = merged2["pa"].to_numpy(dtype=float)

    q50 = float(np.nanquantile(pa, 0.50))
    q85 = float(np.nanquantile(pa, 0.85))
    scale = max(1e-6, q85 - q50)
    z = (pa - q50) / scale

    # Higher PA is more test-like, so increase cutoff model weight as PA rises.
    gate_soft = _sigmoid(2.0 * z)  # gentle
    gate_sharp = _sigmoid(5.0 * z)  # sharper

    outs["gate_pa_cutoff_base_soft"] = gate_soft * c + (1 - gate_soft) * b
    outs["gate_pa_cutoff_base_sharp"] = gate_sharp * c + (1 - gate_sharp) * b

    outs["gate_pa_cutoff_gated_soft"] = gate_soft * c + (1 - gate_soft) * g
    outs["gate_pa_cutoff_gated_sharp"] = gate_sharp * c + (1 - gate_sharp) * g

    # Write
    for name, pred in outs.items():
        out_path = f"{cfg.out_prefix}{name}.csv"
        pd.DataFrame({"id": merged2["id"].astype(int), cfg.target: pred}).to_csv(out_path, index=False)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
