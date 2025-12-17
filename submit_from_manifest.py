import argparse
import csv
import os
from datetime import datetime, timezone

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Submit Kaggle competition submissions listed in a manifest CSV. "
            "Manifest must have columns: file,recipe"
        )
    )
    p.add_argument("--manifest", default="tomorrow_manifest.csv")
    p.add_argument("--competition", default="playground-series-s5e12")
    p.add_argument("--top", type=int, default=5, help="Submit at most N rows from the manifest")
    p.add_argument(
        "--contains",
        default="",
        help="Only submit rows whose recipe contains this substring (case-sensitive)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without uploading",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to the submission message. Default: UTC date YYYY-MM-DD.",
    )
    args = p.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows: list[dict[str, str]] = []
    with open(args.manifest, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "file" not in reader.fieldnames or "recipe" not in reader.fieldnames:
            raise SystemExit("Manifest must have columns: file,recipe")
        for r in reader:
            rows.append({"file": (r.get("file") or "").strip(), "recipe": (r.get("recipe") or "").strip()})

    if args.contains:
        rows = [r for r in rows if args.contains in r["recipe"]]

    rows = [r for r in rows if r["file"]]
    if not rows:
        raise SystemExit("No matching rows to submit")

    rows = rows[: max(0, args.top)]

    for i, r in enumerate(rows, start=1):
        path = os.path.abspath(r["file"])
        if not os.path.exists(path):
            raise SystemExit(f"Missing file: {path}")
        msg = f"AUTO {tag}: {r['recipe']}"
        print(f"[{i}/{len(rows)}] {args.competition} <= {path} | {msg}")

    if args.dry_run:
        print("\nDry-run only; no submissions were uploaded.")
        return

    api = KaggleApi()
    api.authenticate()

    for r in rows:
        path = os.path.abspath(r["file"])
        msg = f"AUTO {tag}: {r['recipe']}"
        api.competition_submit(file_name=path, message=msg, competition=args.competition)


if __name__ == "__main__":
    main()
