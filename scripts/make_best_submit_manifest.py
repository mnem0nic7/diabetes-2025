import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _load_rows(paths: list[Path]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in paths:
        if p.exists():
            out.extend(_read_manifest(p))
    return out


def _pick(rows: list[dict[str, str]], pattern: str) -> dict[str, str] | None:
    rx = re.compile(pattern)
    candidates: list[tuple[float, dict[str, str]]] = []
    for r in rows:
        if not rx.search(r["file"]):
            continue
        p = _abs_path(r["file"])
        if not p.exists():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, r))

    if not candidates:
        return None

    # Prefer the newest on-disk artifact when multiple candidates match.
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    p = argparse.ArgumentParser(description="Create a curated 'best to submit' manifest from existing candidate manifests.")
    p.add_argument("--from", dest="from_paths", default="tomorrow_manifest_v2.csv,tomorrow_manifest_pa.csv", help="Comma-separated input manifests")
    p.add_argument("--out", default="tomorrow_manifest_best.csv")
    args = p.parse_args()

    in_paths = [s.strip() for s in (args.from_paths or "").split(",") if s.strip()]
    manifests = [_abs_path(s) for s in in_paths]

    rows = _load_rows(manifests)
    if not rows:
        raise SystemExit("No rows loaded from input manifests")

    # Ordered shortlist (kept intentionally small). Patterns match the 'file' column.
    #
    # NOTE: Historically this repo used very specific anchor filenames (e.g. v16orig19).
    # When those anchors aren't available locally, we still want to curate a useful
    # shortlist from *whatever* candidates exist in the input manifests.
    wanted_patterns = [
        # OOF-optimized blend candidates (when available)
        r"tomorrow_oofw_.*\.csv$",

        # Nina-style rank-conditioned segment blend (when available)
        r"tomorrow_segrank_.*\.csv$",

        # Rank-average across many models (diversity)
        r"tomorrow_rankavg_.*\.csv$",

        # Anchor-heavy linear blends (prefer 90/10) vs current strong models
        r"tomorrow_lin_.*_weighted_oofblend_rank_90_09\.csv$",
        r"tomorrow_lin_.*_weighted_lgbm_le_90_09\.csv$",
        r"tomorrow_lin_.*_weighted_lgbm_te_90_09\.csv$",

        # A couple slightly less anchored
        r"tomorrow_lin_.*_weighted_oofblend_rank_85_15\.csv$",
        r"tomorrow_lin_.*_weighted_lgbm_le_85_15\.csv$",
        r"tomorrow_lin_.*_weighted_lgbm_te_85_15\.csv$",

        # Nonlinear combos that sometimes beat linear
        r"tomorrow_logitavg_.*_weighted_oofblend_rank\.csv$",
        r"tomorrow_logitavg_.*_weighted_lgbm_le\.csv$",
        r"tomorrow_logitavg_.*_weighted_lgbm_te\.csv$",
        r"tomorrow_pmean_.*_weighted_oofblend_rank\.csv$",
        r"tomorrow_pmean_.*_weighted_lgbm_le\.csv$",
        r"tomorrow_pmean_.*_weighted_lgbm_te\.csv$",

        # PA gates (only a few)
        r"tomorrow_pa_gate_.*\.csv$",
    ]

    picked: list[dict[str, str]] = []
    missing: list[str] = []
    seen_files: set[str] = set()

    for pat in wanted_patterns:
        r = _pick(rows, pat)
        if not r:
            missing.append(pat)
            continue
        if r["file"] in seen_files:
            continue
        picked.append(r)
        seen_files.add(r["file"])

    out_path = _abs_path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "recipe"])
        w.writeheader()
        w.writerows(picked)

    print(f"Wrote {len(picked)} rows -> {out_path}")
    if missing:
        print(f"Missing {len(missing)} expected candidates (skipped).")


if __name__ == "__main__":
    main()
