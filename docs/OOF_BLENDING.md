# OOF-weighted blending

OOF-weighted blending learns ensemble weights using **out-of-fold** predictions, typically giving a more reliable ensemble than hand-picked weights.

## Artifact format

The OOF blending utilities operate on `.npz` files with these keys:

- `name`: string identifier
- `train_id`: int array (length = number of training rows)
- `y`: int array (0/1 labels aligned to `train_id`)
- `oof`: float array (OOF predictions aligned to `train_id`)
- `test_id`: int array (length = number of test rows)
- `test`: float array (test predictions aligned to `test_id`)

All artifacts in a blend must have identical `train_id`, `test_id`, and `y`.

## Producing artifacts

Two scripts can dump artifacts today:

- `scripts/v15_shift_stack.py --preds-dir <dir>`
- `scripts/v16_shift_stack_piecewise_meta.py --preds-dir <dir>`

Each will write multiple `.npz` files (experts + meta) for the run.

## Blending modes

The blender supports three blend spaces:

- `raw`: blend probabilities directly.
- `logit`: blend in logit space, then apply sigmoid.
- `rank`: blend rank-transformed predictions (AUC often improves robustness).

## CLI usage

```bash
python scripts/oof_weight_blend.py \
  --artifacts preds/v16/submission_v16_shift_piecewise__meta.npz,preds/v15/submission_v15_shift_stack__meta.npz \
  --mode rank \
  --out submissions/submission_oofblend.csv
```

Controls:
- `--epochs`, `--step`: hill-climb search budget.
- `--allow-negative`: allow negative weights (usually not recommended).

## Integrating with tomorrow packs

`prepare_tomorrow_blend_pack.py` can add an OOF-optimized candidate directly:

```bash
python scripts/prepare_tomorrow_blend_pack.py \
  --subs "submissions/<anchor>.csv,submissions/<other>.csv" \
  --oof-artifacts preds/v16/...__meta.npz,preds/v15/...__meta.npz \
  --append
```

The generated manifest row will include the learned weights and the OOF AUC.
