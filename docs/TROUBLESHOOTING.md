# Troubleshooting

## Environment / dependencies

### `ModuleNotFoundError: numpy` / `pandas` / `sklearn`

Activate the venv before running scripts:

```bash
source .venv/bin/activate
```

If packages are missing:

```bash
pip install -r requirements.txt
```

### `python` not found / wrong Python

Use `python3` outside the venv. Inside the venv, `python` should work.

## Kaggle auth / uploads

### Auth failures

Ensure `kaggle.json` exists and is readable by your user:

- Put it in the repo root (it is gitignored), then run:
  - `python scripts/daily_best_pack.py --kaggle-config-dir .`

Or set:

```bash
export KAGGLE_CONFIG_DIR="/path/to/dir/with/kaggle.json"
```

### Quota exhausted

Scripts that submit (e.g. `daily_best_pack.py`) check today’s UTC submission count and will skip uploads if you have no remaining quota.

## Data / file size

- `data/train.csv` can be large; some scripts use chunked reads.
- `data/` is gitignored. If scripts complain about missing data files, download the competition data into `data/`.

## OOF blending errors

### “Train id mismatch across artifacts”

You are mixing artifacts generated from different training row sets or orderings.

Fix:
- generate all artifacts from the same run setup (same train file, same filtering/augmentation, same ordering), or
- only blend models that share the same `train_id`/`test_id`.

### Output looks uncalibrated

Try blending in `rank` mode for AUC-focused stability:

```bash
python scripts/oof_weight_blend.py --mode rank ...
```
