# Kaggle “Code” ideas mined (2025-12-29)

Competition: `playground-series-s5e12`

This note summarizes *ideas/patterns* seen in a handful of high-signal public kernels, downloaded locally under:

- `scratch/kaggle_notebooks/2025-12-29/`

## Kernels pulled

- `anthonytherrien/ps-s5e12-blend` → `scratch/kaggle_notebooks/2025-12-29/anthonytherrien__ps-s5e12-blend/`
- `kdmitrie/pgs512-xgb-lgb-ydf-tabm-ag-fe-cv-optuna` → `scratch/kaggle_notebooks/2025-12-29/kdmitrie__pgs512-xgb-lgb-ydf-tabm-ag-fe-cv-optuna/`
- `zhukovoleksiy/s5e12-exploring-fe-optuna-ensemble` → `scratch/kaggle_notebooks/2025-12-29/zhukovoleksiy__s5e12-exploring-fe-optuna-ensemble/`
- `jessewaite/s5e12-catboost-te-5fold-0-70477` → `scratch/kaggle_notebooks/2025-12-29/jessewaite__s5e12-catboost-te-5fold-0-70477/`
- `nina2025/ps-s5e12-segment-setup` → `scratch/kaggle_notebooks/2025-12-29/nina2025__ps-s5e12-segment-setup/`
- `mariusborel/lgbm-predicts-diabetes` → `scratch/kaggle_notebooks/2025-12-29/mariusborel__lgbm-predicts-diabetes/`

## High-signal patterns

### 1) Target encoding done *OOF* to prevent leakage

Common in several kernels: encode categorical features with target mean (and smoothing) **inside CV folds**, then apply learned mapping to validation/test.

Why it matters here:
- This is often a fast, reliable lift for tabular binary classification when categoricals are present.

Where it fits in this repo:
- You already have CatBoost TE work in `scripts/v14_catboost_te.py`.
- If your LGBM/XGB scripts still treat categoricals as ordinal labels, consider swapping in OOF target encoding (or CatBoost’s native handling) for those pipelines.

### 2) CV: prefer StratifiedKFold / RepeatedStratifiedKFold

Multiple kernels use `StratifiedKFold` (sometimes repeated) for more stable OOF estimates on a binary target.

Where it fits:
- Any script producing OOF artifacts for blending (see `scripts/oof_blend_utils.py`, `scripts/oof_weight_blend.py`).

### 3) Outlier handling via quantile clipping (or removing extremes)

Example approaches:
- Clip each numeric feature to `[Q1-1.5*IQR, Q3+1.5*IQR]`.
- Remove/clip values outside low/high quantiles (e.g. 1% / 99%).

Notes:
- This can help some linear-ish models and stabilize trees when there are extreme values.
- Risk: can hurt if “extremes” are actually signal.

Where it fits:
- Easy to A/B in `scripts/train_lgbm_weighted.py` or any baseline script.

### 4) Class imbalance knobs: `scale_pos_weight` / `class_weight`

Several kernels search or set imbalance parameters:
- LightGBM: `scale_pos_weight` or `is_unbalance`.
- Sklearn-style models: `class_weight`.

Where it fits:
- Any LGBM/XGB training script; especially if your CV AUC is sensitive to prevalence shifts.

### 5) Simple weighted blending of submissions

A very common “last mile” improvement:
- Blend multiple submission files by `weighted_avg(pred)`.

Where it fits:
- Your repo already has manifest-based submission flows (`scripts/submit_from_manifest.py`, blend pack scripts, etc.).
- Takeaway: keep a *lightweight* “blend two files quickly” path around for fast iteration.

### 6) Stacking / meta models

A top kernel includes multi-model training + stacking-style ideas (incl. AutoGluon / tabular nets). Even if you don’t adopt those exact libraries, the recurring pattern is:

- Generate diverse base predictors → persist OOF + test preds → train a small meta model or do learned weighting.

Where it fits:
- Your existing OOF artifact + blend tooling (see `docs/OOF_BLENDING.md` and `scripts/oof_weight_blend.py`).

### 7) “Submission distribution” sanity checks

The “segment setup” style kernels do a lot of:
- KDE / histogram comparisons across candidate submissions
- checking column-wise mass / correlations

Where it fits:
- You already have distribution/shift tooling (`scripts/detect_shift.py`, `scripts/poll_competition_submissions.py`).
- Takeaway: compare candidate submission distributions before trusting a blend.

## Suggested quick experiments (low effort)

1) Add an OOF target encoding step to one non-CatBoost pipeline and compare CV AUC.
2) A/B quantile clipping vs none on your strongest LGBM.
3) Tune `scale_pos_weight` per CV (or set it from class ratio) and re-check stability.
4) Ensure every “candidate” model can optionally dump standardized `.npz` OOF artifacts for learned blending.

