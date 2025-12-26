# Workflows

This repo is organized around a few repeatable workflows:

## 1) Daily best-of submissions

Goal: generate multiple lightweight candidates quickly, rank them by a quick validation AUC, and submit the top ones (up to the daily quota).

Command:

```bash
source .venv/bin/activate
python scripts/daily_best_pack.py --kaggle-config-dir . --max-candidates 10
```

Outputs:
- `daily_manifest.csv` with columns `file,recipe`
- multiple `submissions/daily_<YYYY-MM-DD>_seed*_n*(_adv).csv`

Notes:
- “Quick validation” is a small holdout split used for ranking; it is not a full CV.
- Candidates can include `adv=1` density-ratio weights (train-vs-test discrimination).

## 2) Tomorrow blend pack (submission-space blends)

Goal: build multiple blends from existing submission CSVs.

Command:

```bash
python scripts/prepare_tomorrow_blend_pack.py \
  --subs "submissions/<anchor>.csv,submissions/<other1>.csv,submissions/<other2>.csv" \
  --weights "0.90,0.85,0.80" \
  --manifest tomorrow_manifest_v2.csv
```

Blend types:
- rank-average across all submissions
- pairwise linear blends vs the anchor
- pairwise logit-average vs the anchor
- pairwise power-mean vs the anchor

## 3) Curate a small “best” manifest

Goal: select a short list of the strongest candidates (based on filename patterns) from one or more manifests.

Command:

```bash
python scripts/make_best_submit_manifest.py \
  --from tomorrow_manifest_v2.csv,tomorrow_manifest_pa.csv \
  --out tomorrow_manifest_best.csv
```

## 4) Submit from a manifest

Goal: submit the first N rows from a manifest.

Command:

```bash
python scripts/submit_from_manifest.py --manifest tomorrow_manifest_best.csv --top 5
```

Use `--dry-run` to print what would be uploaded.

## 5) OOF-weighted blending (learned weights)

Goal: learn blend weights using **out-of-fold** predictions to maximize ROC-AUC, then apply those weights to test predictions.

Steps:

1) Run a model script that dumps `.npz` artifacts:

```bash
python scripts/v16_shift_stack_piecewise_meta.py \
  --out submissions/submission_v16_shift_piecewise.csv \
  --preds-dir preds/v16
```

2) Blend artifacts:

```bash
python scripts/oof_weight_blend.py \
  --artifacts preds/v16/submission_v16_shift_piecewise__meta.npz,preds/v15/submission_v15_shift_stack__meta.npz \
  --mode rank \
  --out submissions/submission_oofblend.csv
```

3) (Optional) include OOF-optimized candidate in the tomorrow pack generator:

```bash
python scripts/prepare_tomorrow_blend_pack.py \
  --oof-artifacts preds/v16/...__meta.npz,preds/v15/...__meta.npz \
  --append
```
