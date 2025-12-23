import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Defaults:
    competition: str = "playground-series-s5e12"
    out: str = "tomorrow_manifest_best.csv"
    manifests: str = ",".join(
        [
            "tomorrow_manifest_v2.csv",
            "tomorrow_manifest_pa.csv",
            "daily_manifest.csv",
            "scratch/tomorrow_manifest.csv",
            "tomorrow_manifest.csv",
        ]
    )
    top: int = 50
    priors_limit: int = 100


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def _default_kaggle_config_dir() -> str:
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return os.environ["KAGGLE_CONFIG_DIR"]
    if (REPO_ROOT / "kaggle.json").exists():
        return str(REPO_ROOT)
    home_token = Path.home() / ".kaggle" / "kaggle.json"
    if home_token.exists():
        return str(home_token.parent)
    return ""


def _ensure_kaggle_config_dir(config_dir: str) -> None:
    if config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = config_dir


def _authenticate_kaggle() -> KaggleApi:
    _ensure_kaggle_config_dir(os.environ.get("KAGGLE_CONFIG_DIR", "") or _default_kaggle_config_dir())
    api = KaggleApi()
    api.authenticate()
    return api


def _read_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "file" not in reader.fieldnames or "recipe" not in reader.fieldnames:
            raise ValueError(f"Manifest must have columns file,recipe: {path}")
        for r in reader:
            rows.append({"file": (r.get("file") or "").strip(), "recipe": (r.get("recipe") or "").strip()})
    return [r for r in rows if r["file"]]


def _abs_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def fetch_score_priors(api: KaggleApi, competition: str, limit: int, higher_is_better: bool) -> dict[str, float]:
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(limit, 100)),
    )

    scores: dict[str, float] = {}
    for s in (subs or []):
        status = str(getattr(s, "status", "") or "")
        if "COMPLETE" not in status:
            continue

        file_name = str(getattr(s, "file_name", "") or "")
        if not file_name:
            continue

        public = getattr(s, "public_score", None)
        try:
            val = float(public)
        except Exception:
            continue

        if file_name not in scores:
            scores[file_name] = val
        else:
            scores[file_name] = max(scores[file_name], val) if higher_is_better else min(scores[file_name], val)

    return scores


def score_index(score_by_file: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in score_by_file.items():
        out[k] = v
        if k.lower().endswith(".csv"):
            out[k[:-4]] = v
    return out


def extract_components(recipe: str) -> list[str]:
    tokens = _TOKEN_RE.findall(recipe or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def estimate_score(recipe: str, score_idx: dict[str, float]) -> float | None:
    comps = extract_components(recipe)
    comp_scores: list[tuple[str, float]] = [(c, score_idx[c]) for c in comps if c in score_idx]
    if not comp_scores:
        return None

    # Weighted patterns: 0.80*foo
    weights = re.findall(r"(0\.[0-9]+)\*([A-Za-z][A-Za-z0-9_]{2,})", recipe or "")
    wlist: list[tuple[float, float]] = []
    for w_str, name in weights:
        try:
            w = float(w_str)
        except Exception:
            continue
        if name in score_idx:
            wlist.append((w, score_idx[name]))

    if wlist:
        wsum = sum(w for w, _ in wlist)
        if wsum > 0:
            return sum(w * s for w, s in wlist) / wsum

    return sum(s for _, s in comp_scores) / len(comp_scores)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a curated manifest of best candidate submissions to upload.")
    p.add_argument("-c", "--competition", default=Defaults.competition)
    p.add_argument("--manifests", default=Defaults.manifests, help="Comma-separated manifest paths")
    p.add_argument("--out", default=Defaults.out)
    p.add_argument("--top", type=int, default=Defaults.top)
    p.add_argument("--priors-limit", type=int, default=Defaults.priors_limit)
    p.add_argument("--higher-is-better", action="store_true", default=True)
    p.add_argument("--lower-is-better", action="store_true", help="If set, treat lower public score as better")
    p.add_argument("--include-unknown", action="store_true", help="Include rows with no estimated score")
    p.add_argument("--exclude-submitted", action="store_true", default=True)
    p.add_argument("--allow-submitted", action="store_true", help="If set, do not exclude already-uploaded filenames")
    args = p.parse_args()

    higher_is_better = bool(args.higher_is_better)
    if args.lower_is_better:
        higher_is_better = False

    if args.allow_submitted:
        exclude_submitted = False
    else:
        exclude_submitted = bool(args.exclude_submitted)

    manifest_paths = [s.strip() for s in (args.manifests or "").split(",") if s.strip()]
    resolved: list[Path] = []
    for m in manifest_paths:
        pth = _abs_path(m)
        if pth.exists():
            resolved.append(pth)

    if not resolved:
        raise SystemExit("No manifest files found from --manifests")

    api = _authenticate_kaggle()
    priors = fetch_score_priors(api, args.competition, limit=int(args.priors_limit), higher_is_better=higher_is_better)
    idx = score_index(priors)

    uploaded_names = set(priors.keys())

    candidates: dict[str, dict[str, str]] = {}
    meta: dict[str, dict[str, object]] = {}

    for mp in resolved:
        for r in _read_manifest(mp):
            abs_file = _abs_path(r["file"])
            if not abs_file.exists():
                continue

            base_name = abs_file.name
            if exclude_submitted and base_name in uploaded_names:
                continue

            key = str(abs_file)
            if key not in candidates:
                candidates[key] = {"file": r["file"].replace("\\", "/"), "recipe": r["recipe"]}

            est = estimate_score(r["recipe"], idx)
            meta[key] = {
                "est": est,
                "submitted": base_name in uploaded_names,
            }

    rows = []
    for key, r in candidates.items():
        est = meta.get(key, {}).get("est")
        if est is None and not args.include_unknown:
            continue
        rows.append((r, est))

    def sort_key(item):
        _, est = item
        # Unknown last
        if est is None:
            return (1, 0.0)
        return (0, -est if higher_is_better else est)

    rows.sort(key=sort_key)

    out_rows = [r for r, _ in rows][: max(0, int(args.top))]

    out_path = _abs_path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "recipe"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows -> {out_path}")
    if out_rows:
        print("Top 5:")
        for i, r in enumerate(out_rows[:5], start=1):
            print(f"  [{i}] {r['file']} | {r['recipe']}")


if __name__ == "__main__":
    main()
