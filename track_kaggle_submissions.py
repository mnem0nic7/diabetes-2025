import argparse
import csv
import os
import time
from datetime import datetime

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
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
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def fetch_submissions(api: KaggleApi, competition: str, limit: int):
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(limit, 100)),
    )
    return subs or []


def upsert_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existing: dict[str, dict[str, str]] = {}
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ref = (r.get("ref") or "").strip()
                if ref:
                    existing[ref] = r

    for r in rows:
        existing[r["ref"]] = r

    merged = list(existing.values())
    merged.sort(key=lambda r: (r.get("date") or ""), reverse=True)

    fieldnames = ["date", "status", "public", "ref", "file", "desc"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def main() -> None:
    p = argparse.ArgumentParser(description="Track Kaggle submissions into a local CSV (status + public score).")
    p.add_argument("-c", "--competition", default="playground-series-s5e12")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--contains", default="AUTO", help="Only keep rows whose description contains this substring")
    p.add_argument("--out", default="scratch/submission_tracking.csv")
    p.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If >0, refresh every N seconds until all filtered rows are COMPLETE and scored.",
    )
    args = p.parse_args()

    if not os.environ.get("KAGGLE_CONFIG_DIR"):
        repo_hint = os.path.abspath(".")
        print(
            "Hint: set KAGGLE_CONFIG_DIR to point to the folder containing kaggle.json.\n"
            f"  Example (PowerShell): $env:KAGGLE_CONFIG_DIR=\"{repo_hint}\"\n"
        )

    api = KaggleApi()
    api.authenticate()

    while True:
        subs = fetch_submissions(api, args.competition, args.limit)
        rows: list[dict[str, str]] = []

        for s in subs:
            desc = getattr(s, "description", "") or ""
            if args.contains and args.contains not in desc:
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

        upsert_csv(args.out, rows)
        print(f"Wrote {len(rows)} rows -> {args.out}")

        if args.watch <= 0:
            return

        all_scored = True
        for r in rows:
            if "COMPLETE" not in (r.get("status") or ""):
                all_scored = False
                break
            if not (r.get("public") or "").strip():
                all_scored = False
                break

        if all_scored and rows:
            return

        print(f"Waiting {args.watch}s...")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
