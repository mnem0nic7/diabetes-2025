import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


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
    for r in rows:
        if rx.search(r["file"]):
            if _abs_path(r["file"]).exists():
                return r
    return None


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
    wanted_patterns = [
        # Rank-average across many strong models (diversity)
        r"tomorrow_rankavg_.*\.csv$",

        # High-weight blends vs anchor best blend (0.90)
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v23_stack_advanced_90_09\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v16_drop6_pl05_heavy3_90_09\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v18_multislice_90_09\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v21_autogluon_full_90_09\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v22_recon_weighted_lgb_90_09\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v17_lgb_orig_mix_90_09\.csv$",

        # A couple slightly less anchored (0.85)
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v23_stack_advanced_85_15\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v16_drop6_pl05_heavy3_85_15\.csv$",
        r"tomorrow_lin_blend_cutoff80_v16orig19_submission_v18_multislice_85_15\.csv$",

        # Nonlinear combos that sometimes beat linear
        r"tomorrow_logitavg_blend_cutoff80_v16orig19_submission_v23_stack_advanced\.csv$",
        r"tomorrow_pmean_p10_blend_cutoff80_v16orig19_submission_v23_stack_advanced\.csv$",

        # PA gates (only a few), for the strongest candidate vs base/gated anchors
        r"tomorrow_pa_gate_s5_q90_submission_v23_stack_advanced_vs_blend_cutoff80_v16orig19\.csv$",
        r"tomorrow_pa_gate_s6_q90_submission_v23_stack_advanced_vs_blend_cutoff80_v16orig19\.csv$",
        r"tomorrow_pa_gate_s5_q90_submission_v23_stack_advanced_vs_submission_v20_weighted_gated\.csv$",
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
