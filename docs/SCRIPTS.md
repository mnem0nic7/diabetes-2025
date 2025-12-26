# Scripts

This repo primarily runs via Python scripts in `scripts/`.

## Submission automation

- `scripts/daily_best_pack.py`
  - Builds a small grid of “daily” candidates, ranks by quick validation AUC, writes `daily_manifest.csv`, then submits up to remaining quota.
- `scripts/daily_submit.py`
  - One-command “build + (optional) upload” for a single daily candidate.
- `scripts/submit_from_manifest.py`
  - Submits rows listed in a `file,recipe` manifest.

## Blend / candidate generation

- `scripts/prepare_tomorrow_blend_pack.py`
  - Generates a pack of blends from existing submission CSVs:
    - `rankavg` across all
    - pairwise linear blends vs anchor
    - pairwise `logitavg`
    - pairwise power-mean
  - Optionally adds an OOF-optimized blend via `--oof-artifacts`.

- `scripts/make_best_submit_manifest.py`
  - Creates a curated manifest by selecting key candidate filenames from other manifests.

## Shift-aware / stacking models

- `scripts/v15_shift_stack.py`
  - Shift-aware mixture/stacking using `p_test` (train-vs-test probability) + multiple experts.
  - Can dump standardized OOF/test artifacts with `--preds-dir`.

- `scripts/v16_shift_stack_piecewise_meta.py`
  - Similar to v15 but uses piecewise (binned) meta features and tunes LR regularization.
  - Can dump standardized OOF/test artifacts with `--preds-dir`.

## Feature engineering / baseline models

- `scripts/v8_cutoff_boost_lgb.py`
  - LightGBM with “cutoff” split and optional external/original mean/count encodings.

- `scripts/v14_catboost_te.py`
  - CatBoost + target encoding and optional external/original dataset augmentation.

- `scripts/train_lgbm_weighted.py`
  - LightGBM baseline with adversarial validation weights.

## OOF-weighted blending utilities

- `scripts/oof_blend_utils.py`
  - Implements loading of `.npz` artifacts, ID alignment, weight hill-climbing on OOF ROC-AUC, and blending in `raw|logit|rank` spaces.

- `scripts/oof_weight_blend.py`
  - CLI wrapper that prints per-model OOF AUCs, learns weights, and writes a blended submission CSV.

## Kaggle kernel metadata

- `kaggle.yml` and `kernel-metadata.json` describe an optional private Kaggle kernel configuration.
- In practice the scripts default to competition `playground-series-s5e12`.
