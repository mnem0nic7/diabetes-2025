"""One-command daily submit: build + quota-check + (optional) upload.

This wraps the lightweight generator in daily_quick_submit and only uploads if
there is remaining submission quota for the current UTC day.

Examples:
  # Build + submit (skips upload if quota exhausted)
  python scripts/daily_submit.py

  # Just build the CSV (never uploads)
  python scripts/daily_submit.py --no-upload

  # Use a different sample size
  python scripts/daily_submit.py --train-sample 300000

Notes:
- Uses Kaggle API auth via either $KAGGLE_CONFIG_DIR or an explicit --kaggle-config-dir.
- Counts today's submissions (UTC) from recent submissions for the competition.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy

from daily_quick_submit import generate_submission


def _utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fetch_recent_submissions(api: KaggleApi, competition: str, limit: int) -> list[object]:
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(int(limit), 100)),
    )
    return subs or []


def _count_submissions_for_utc_date(subs: list[object], utc_date: str) -> int:
    # kagglesdk returns date either as datetime or string.
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


def main() -> None:
    p = argparse.ArgumentParser(description="Daily Kaggle submission (build + quota-check + upload).")
    p.add_argument("--competition", default="playground-series-s5e12")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--target", default="diagnosed_diabetes")
    p.add_argument("--train-sample", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--daily-limit", type=int, default=5, help="Max submissions per UTC day to assume.")
    p.add_argument("--submission-lookback", type=int, default=50, help="How many recent submissions to inspect.")

    p.add_argument(
        "--kaggle-config-dir",
        default="",
        help="Folder containing kaggle.json. If omitted, uses $KAGGLE_CONFIG_DIR if set.",
    )
    p.add_argument("--no-upload", action="store_true", help="Only build the CSV; do not upload.")

    p.add_argument(
        "--tag",
        default="",
        help="Submission tag used in message. Default: UTC date YYYY-MM-DD.",
    )
    p.add_argument("--out", default="", help="Output CSV path. Default: submission_daily_<UTC date>.csv")
    args = p.parse_args()

    utc_tag = args.tag or _utc_today_str()
    quota_day = _utc_today_str()
    out_path = Path(args.out) if args.out else Path(f"submission_daily_{utc_tag}.csv")

    # Build submission file first (always), so it's ready even if we skip upload.
    out_path = generate_submission(
        data_dir=args.data_dir,
        target=args.target,
        train_sample=int(args.train_sample),
        seed=int(args.seed),
        out_path=str(out_path),
    )

    if args.no_upload:
        print(f"Built only (no upload): {out_path}")
        return

    # Point Kaggle API at the desired config dir.
    if args.kaggle_config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = args.kaggle_config_dir

    api = KaggleApi()
    api.authenticate()

    subs = _fetch_recent_submissions(api, args.competition, limit=int(args.submission_lookback))
    today_count = _count_submissions_for_utc_date(subs, quota_day)

    if today_count >= int(args.daily_limit):
        print(
            f"Quota exhausted for {quota_day} (UTC): {today_count}/{args.daily_limit} submissions already. "
            f"Skipping upload; file is ready: {out_path}"
        )
        return

    msg = f"AUTO {utc_tag}: daily_quick_submit (train_sample={int(args.train_sample)})"
    print(f"Submitting {out_path} -> {args.competition} | {msg}")
    api.competition_submit(file_name=str(out_path), message=msg, competition=args.competition)


if __name__ == "__main__":
    main()
