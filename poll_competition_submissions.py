import argparse
import os
import time
from datetime import datetime

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    # dt can be a datetime or a string depending on SDK
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def _score_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.5f}"
    except Exception:
        return str(x)


def fetch_submissions(api: KaggleApi, competition: str, limit: int):
    # This endpoint works even when the kaggle CLI is broken (CLI passes page_number).
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(limit, 100)),
    )
    return subs or []


def print_submissions(subs, contains: str | None, limit: int):
    rows = []
    for s in subs:
        desc = getattr(s, "description", "") or ""
        if contains and contains not in desc:
            continue

        rows.append(
            {
                "date": _fmt_dt(getattr(s, "date", None)),
                "status": str(getattr(s, "status", "")),
                "public": _score_str(getattr(s, "public_score", None)),
                "ref": str(getattr(s, "ref", "")),
                "file": str(getattr(s, "file_name", "")),
                "desc": desc,
            }
        )

    rows = rows[:limit]

    # Header
    print("date\t\t\tstatus\t\tpublic\tref\t\tfile\tdescription")
    for r in rows:
        print(
            f"{r['date']}\t{r['status']}\t{r['public']}\t{r['ref']}\t{r['file']}\t{r['desc']}"
        )

    return rows


def all_scored(rows) -> bool:
    if not rows:
        return False
    for r in rows:
        if "COMPLETE" not in r["status"]:
            return False
        if not r["public"]:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Poll Kaggle competition submissions + public score.")
    p.add_argument("-c", "--competition", default="playground-series-s5e12")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--contains", default="", help="Only show submissions whose description contains this substring")
    p.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If >0, refresh every N seconds until all shown rows are COMPLETE and have a public score.",
    )
    args = p.parse_args()

    # Helpful hint if user wants to keep kaggle.json in this repo.
    if not os.environ.get("KAGGLE_CONFIG_DIR"):
        # No side effects, just guidance.
        pass

    api = KaggleApi()
    api.authenticate()

    contains = args.contains or None
    while True:
        subs = fetch_submissions(api, args.competition, args.limit)
        rows = print_submissions(subs, contains=contains, limit=args.limit)

        if args.watch <= 0:
            return

        if all_scored(rows):
            return

        print(f"\nWaiting {args.watch}s for scores to populate...\n")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
