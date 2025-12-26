# diabetes-2025 (Kaggle)

Local workflow for the Kaggle Playground Series S5E12 competition (target: `diagnosed_diabetes`).

This repo is optimized for fast daily iteration:
- train small / shift-aware models
- generate multiple candidate submissions
- build “tomorrow” blend packs
- submit the top-N automatically (respecting Kaggle daily quota)

## Project layout

- `data/`
  - `train.csv`, `test.csv`, `sample_submission.csv`
  - (Optional) external/original datasets under `data/orig*/...`
- `scripts/` — training, blending, manifests, and submission automation
- `submissions/` — generated submission CSVs
- `notebooks/` — exploration and model experiments
- `scratch/` — local experiments and pulled public kernels

## Setup

### 1) Python environment

Use the repo virtualenv:

```bash
cd /path/to/diabetes-2025
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Kaggle authentication

This repo expects `kaggle.json` to be available via one of:
- `KAGGLE_CONFIG_DIR` pointing to a folder that contains `kaggle.json`
- passing `--kaggle-config-dir .` to scripts that upload

Example:

```bash
export KAGGLE_CONFIG_DIR="$(pwd)"
```

Notes:
- `kaggle.json` is gitignored by default.
- Uploading uses the Kaggle API via `kaggle`/`kagglesdk`.

## Daily workflow (best-of pack)

### Build + submit the best 5 for today

This generates a small grid of candidates (seed/sample size, optionally adversarial weights), ranks them by a quick validation AUC, writes `daily_manifest.csv`, then submits up to the remaining daily quota.

```bash
source .venv/bin/activate
python scripts/daily_best_pack.py --kaggle-config-dir . --max-candidates 10
```

To build only (no upload):

```bash
python scripts/daily_best_pack.py --no-upload --max-candidates 10
```

Artifacts:
- `daily_manifest.csv` (file, recipe)
- `submissions/daily_<YYYY-MM-DD>_seed*_n*.csv`

## “Tomorrow” blend packs

The repo uses a manifest-based pipeline:
1) generate a pack of blend candidates
2) write a manifest CSV
3) submit top rows from the manifest

### Build a blend pack

```bash
python scripts/prepare_tomorrow_blend_pack.py \
  --subs "submissions/<anchor>.csv,submissions/<other1>.csv,submissions/<other2>.csv" \
  --manifest tomorrow_manifest_v2.csv
```

This produces candidates like:
- `tomorrow_rankavg_*`
- `tomorrow_lin_*`
- `tomorrow_logitavg_*`
- `tomorrow_pmean_*`

### Curate a short “best to submit” manifest

```bash
python scripts/make_best_submit_manifest.py --out tomorrow_manifest_best.csv
```

### Submit from manifest

```bash
python scripts/submit_from_manifest.py --manifest tomorrow_manifest_best.csv --top 5
```

Use `--dry-run` to print without uploading.

## OOF-weighted blending (OOF AUC optimized)

For blends that are learned from out-of-fold predictions (rather than hand-picked weights), see:
- `scripts/oof_weight_blend.py` (CLI)
- `scripts/oof_blend_utils.py` (implementation)
- `scripts/v15_shift_stack.py` / `scripts/v16_shift_stack_piecewise_meta.py` (can dump `.npz` prediction artifacts via `--preds-dir`)

Quick start:

1) Run a model and dump artifacts:
```bash
python scripts/v16_shift_stack_piecewise_meta.py --out submissions/submission_v16_shift_piecewise.csv --preds-dir preds/v16
```

2) Learn weights on OOF AUC and output a blended submission:
```bash
python scripts/oof_weight_blend.py \
  --artifacts preds/v16/submission_v16_shift_piecewise__meta.npz,preds/v15/submission_v15_shift_stack__meta.npz \
  --mode rank \
  --out submissions/submission_oofblend.csv
```

You can also include an OOF-optimized candidate inside the “tomorrow pack” generator with `--oof-artifacts`.

## Troubleshooting

- If a script fails with `ModuleNotFoundError: numpy` etc, ensure the venv is activated.
- If uploads fail, check `kaggle.json` permissions and `KAGGLE_CONFIG_DIR`.

More detailed docs:
- `docs/WORKFLOWS.md`
- `docs/SCRIPTS.md`
- `docs/OOF_BLENDING.md`
- `docs/TROUBLESHOOTING.md`
- `AGENTS.md` (instructions for coding agents)
