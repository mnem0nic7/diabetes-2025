from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oof_blend_utils import (
    load_pred_artifact,
    oof_optimize_and_blend,
    transform_preds,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Learn blend weights from OOF predictions to maximize ROC-AUC, then write a blended submission. "
            "Inputs are .npz artifacts containing train_id,y,oof,test_id,test (see --help in producer scripts)."
        )
    )
    p.add_argument(
        "--artifacts",
        required=True,
        help="Comma-separated .npz artifact paths (must share identical train_id/test_id).",
    )
    p.add_argument("--target", default="diagnosed_diabetes")
    p.add_argument("--out", required=True, help="Output submission CSV path")
    p.add_argument("--mode", default="raw", choices=["raw", "logit", "rank"], help="Space to blend in")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--allow-negative", action="store_true", help="Allow negative weights (not recommended)")
    args = p.parse_args()

    paths = [s.strip() for s in (args.artifacts or "").split(",") if s.strip()]
    if len(paths) < 2:
        raise SystemExit("Provide at least 2 artifacts via --artifacts")

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing artifact files:\n- " + "\n- ".join(missing))

    artifacts = [load_pred_artifact(p) for p in paths]

    # Report individual OOF AUCs (in the same transform space).
    y = artifacts[0].y
    for a in artifacts:
        auc = float(roc_auc_score(y, transform_preds(a.oof, args.mode)))
        print(f"OOF AUC | {a.name}: {auc:.6f}")

    w, blended_test, best_auc = oof_optimize_and_blend(
        artifacts,
        mode=args.mode,
        seed=int(args.seed),
        epochs=int(args.epochs),
        step=float(args.step),
        positive=(not args.allow_negative),
    )

    print("=" * 60)
    print(f"Best blended OOF AUC ({args.mode}): {best_auc:.6f}")
    for name, wi in sorted(zip([a.name for a in artifacts], w), key=lambda x: -x[1]):
        print(f"  {wi: .5f}  {name}")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    test_id = artifacts[0].test_id
    df = pd.DataFrame({"id": test_id.astype(int), args.target: blended_test.astype(float)})
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
