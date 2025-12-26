"""Generate and submit a small pack of strong daily submissions.

What it does:
1) Builds multiple daily candidates (different seeds/sample sizes)
2) Ranks them by quick validation AUC
3) Writes daily_manifest.csv (file,recipe)
4) Submits the top candidates up to today's remaining Kaggle quota

This is intentionally lightweight and designed for constrained environments.

Examples:
  # Build candidates + submit as many as quota allows
  python scripts/daily_best_pack.py --kaggle-config-dir .

  # Build only, no upload
  python scripts/daily_best_pack.py --no-upload

  # Generate fewer candidates
  python scripts/daily_best_pack.py --max-candidates 3
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy

from daily_quick_submit import generate_submission_with_metrics


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CandidateSpec:
    seed: int
    train_sample: int
    adv_weights: bool = False


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_kaggle_config_dir(config_dir: str) -> None:
    if config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = config_dir


def _fetch_recent_submissions(api: KaggleApi, competition: str, limit: int) -> list[object]:
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(int(limit), 100)),
    )
    return subs or []


def _count_submissions_for_utc_date(subs: list[object], utc_date: str) -> int:
    n = 0
    for s in subs:
        dt = getattr(s, "date", None)
        if dt is None:
            continue
        if isinstance(dt, str):
            if dt.startswith(utc_date):
                n += 1
        else:
            try:
                if dt.strftime("%Y-%m-%d") == utc_date:
                    n += 1
            except Exception:
                pass
    return n


def _already_uploaded_file_names(subs: list[object]) -> set[str]:
    out: set[str] = set()
    for s in subs:
        fn = str(getattr(s, "file_name", "") or "")
        if fn:
            out.add(fn)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate and submit a best-of daily pack.")
    p.add_argument("--competition", default="playground-series-s5e12")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--target", default="diagnosed_diabetes")
    p.add_argument("--out-dir", default="submissions")
    p.add_argument("--manifest", default="daily_manifest.csv")

    p.add_argument("--daily-limit", type=int, default=5)
    p.add_argument("--submission-lookback", type=int, default=80)
    p.add_argument("--kaggle-config-dir", default="")

    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument(
        "--date",
        default="",
        help="Override the date tag used in filenames/messages (YYYY-MM-DD). Defaults to current UTC day.",
    )
    p.add_argument(
        "--adv-weights",
        default="0,1",
        help="Comma-separated list of 0/1 toggles to include adversarial-weighted candidates.",
    )
    p.add_argument(
        "--adv-test-sample",
        type=int,
        default=200_000,
        help="How many test rows to sample when fitting adversarial weights (0=all).",
    )
    args = p.parse_args()

    utc_day = args.date.strip() or _utc_day()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adv_flags = [bool(int(x)) for x in str(args.adv_weights).split(",") if x.strip()]

    # A small grid of candidates; we rank by quick val AUC.
    # Keep this intentionally small to avoid long runtimes.
    base_grid = [
        (42, 200_000),
        (43, 200_000),
        (44, 200_000),
        (42, 250_000),
        (43, 250_000),
    ]

    grid: list[CandidateSpec] = [
        CandidateSpec(seed=s, train_sample=n, adv_weights=a) for (s, n) in base_grid for a in adv_flags
    ]
    grid = grid[: max(0, int(args.max_candidates))]

    built: list[dict[str, object]] = []
    for spec in grid:
        adv_tag = "_adv" if spec.adv_weights else ""
        out_path = out_dir / f"daily_{utc_day}_seed{spec.seed}_n{spec.train_sample}{adv_tag}.csv"
        path, metrics = generate_submission_with_metrics(
            data_dir=args.data_dir,
            target=args.target,
            train_sample=int(spec.train_sample),
            seed=int(spec.seed),
            adv_weights=bool(spec.adv_weights),
            adv_test_sample=int(args.adv_test_sample),
            out_path=str(out_path),
        )
        built.append(
            {
                "file": str(path).replace("\\", "/"),
                "seed": spec.seed,
                "train_sample": spec.train_sample,
                "adv_weights": int(bool(spec.adv_weights)),
                "val_auc": float(metrics.get("val_auc", 0.0)),
            }
        )
        adv_note = " adv=1" if spec.adv_weights else ""
        print(f"Built {path} | val_auc={metrics.get('val_auc'):.6f}{adv_note}")

    # Rank by val AUC descending
    built.sort(key=lambda r: float(r["val_auc"]), reverse=True)

    # Write manifest (file,recipe)
    manifest_path = Path(args.manifest)
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "recipe"])
        w.writeheader()
        for r in built:
            adv_note = " adv=1" if int(r.get("adv_weights", 0)) else ""
            recipe = (
                f"daily_quick_submit seed={int(r['seed'])} n={int(r['train_sample'])}{adv_note} "
                f"val_auc={float(r['val_auc']):.6f}"
            )
            w.writerow({"file": r["file"], "recipe": recipe})

    print(f"Wrote manifest: {manifest_path} ({len(built)} rows)")

    if args.no_upload:
        print("No-upload mode; skipping Kaggle submission.")
        return

    _ensure_kaggle_config_dir(args.kaggle_config_dir)
    api = KaggleApi()
    api.authenticate()

    subs = _fetch_recent_submissions(api, args.competition, limit=int(args.submission_lookback))
    used_today = _count_submissions_for_utc_date(subs, utc_day)
    remaining = max(0, int(args.daily_limit) - used_today)

    if remaining <= 0:
        print(f"Quota exhausted for {utc_day} (UTC): {used_today}/{args.daily_limit}. Nothing to submit.")
        return

    uploaded_files = _already_uploaded_file_names(subs)

    # Submit top-N that aren't already uploaded by filename
    to_submit = []
    for r in built:
        base = Path(str(r["file"])).name
        if base in uploaded_files:
            continue
        to_submit.append(r)
        if len(to_submit) >= remaining:
            break

    if not to_submit:
        print(f"No new files to submit (remaining quota {remaining}).")
        return

    for i, r in enumerate(to_submit, start=1):
        file_path = str(r["file"])
        adv_note = " adv=1" if int(r.get("adv_weights", 0)) else ""
        msg = (
            f"AUTO {utc_day}: seed={int(r['seed'])} "
            f"n={int(r['train_sample'])} "
            f"{adv_note.strip()} "
            f"val_auc={float(r['val_auc']):.6f}"
        )
        msg = " ".join(msg.split())
        print(f"[{i}/{len(to_submit)}] Submitting {file_path} -> {args.competition} | {msg}")
        api.competition_submit(file_name=file_path, message=msg, competition=args.competition)


if __name__ == "__main__":
    main()
