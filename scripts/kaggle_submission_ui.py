import csv
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMITTED_LEDGER = REPO_ROOT / "scratch" / "submitted_candidates.csv"


def _default_kaggle_config_dir() -> str:
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return os.environ["KAGGLE_CONFIG_DIR"]

    repo_token = REPO_ROOT / "kaggle.json"
    if repo_token.exists():
        return str(REPO_ROOT)

    home_token = Path.home() / ".kaggle" / "kaggle.json"
    if home_token.exists():
        return str(home_token.parent)

    return ""


def _ensure_kaggle_config_dir(config_dir: str) -> None:
    if config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = config_dir


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
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


def _list_manifest_paths() -> list[Path]:
    patterns = [
        "*manifest*.csv",
        "tomorrow_manifest*.csv",
        "daily_manifest.csv",
    ]

    candidates: set[Path] = set()
    for pat in patterns:
        candidates.update(REPO_ROOT.glob(pat))
        candidates.update((REPO_ROOT / "scratch").glob(pat))

    out = [p for p in candidates if p.is_file()]
    out.sort(key=lambda p: (p.name.lower(), str(p.parent).lower()))
    return out


def _read_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "file" not in reader.fieldnames or "recipe" not in reader.fieldnames:
            raise ValueError("Manifest must have columns: file,recipe")
        for r in reader:
            rows.append({"file": (r.get("file") or "").strip(), "recipe": (r.get("recipe") or "").strip()})

    rows = [r for r in rows if r["file"]]
    return rows


def _abs_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _manifest_signature(manifest_paths: list[Path]) -> str:
    parts: list[str] = []
    for p in manifest_paths:
        try:
            stt = p.stat()
            parts.append(f"{p.as_posix()}:{int(stt.st_mtime)}:{int(stt.st_size)}")
        except Exception:
            parts.append(p.as_posix())
    return str(hash("|".join(parts)))


def _load_submitted_index() -> set[str]:
    """Return a set of normalized identifiers for already-submitted candidates."""

    out: set[str] = set()

    # UI ledger of submitted candidates.
    if SUBMITTED_LEDGER.exists():
        try:
            df = pd.read_csv(SUBMITTED_LEDGER)
            if not df.empty and "file" in df.columns:
                for v in df["file"].astype(str).tolist():
                    if not v or v == "nan":
                        continue
                    try:
                        ap = str(_abs_path(v))
                    except Exception:
                        ap = v
                    out.add(ap)
                    base = Path(v).name
                    if base:
                        out.add(base)
                        if base.lower().endswith(".csv"):
                            out.add(base[:-4])
        except Exception:
            pass

    # Optional: if a local Kaggle tracking cache exists, treat its file_name entries as submitted.
    tracking = REPO_ROOT / "scratch" / "submission_tracking.csv"
    if tracking.exists():
        try:
            df = pd.read_csv(tracking)
            if not df.empty and "file" in df.columns:
                for v in df["file"].astype(str).tolist():
                    if not v or v == "nan":
                        continue
                    base = Path(v).name
                    if base:
                        out.add(base)
                        if base.lower().endswith(".csv"):
                            out.add(base[:-4])
        except Exception:
            pass

    return out


def _append_submitted_ledger(competition: str, rows: list[dict[str, str]]) -> None:
    SUBMITTED_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    existing: list[dict[str, str]] = []
    if SUBMITTED_LEDGER.exists():
        try:
            existing_df = pd.read_csv(SUBMITTED_LEDGER)
            if not existing_df.empty:
                existing = existing_df.fillna("").to_dict(orient="records")
        except Exception:
            existing = []

    for r in rows:
        existing.append(
            {
                "timestamp": now,
                "competition": competition,
                "file": str(r.get("file") or ""),
                "recipe": str(r.get("recipe") or ""),
            }
        )

    df = pd.DataFrame(existing)
    # De-dupe by file (keep earliest entry for stability)
    if not df.empty and "file" in df.columns:
        df = df.drop_duplicates(subset=["file"], keep="first")
    df.to_csv(SUBMITTED_LEDGER, index=False)


def _authenticate_kaggle() -> KaggleApi:
    # Ensure env var is set before authenticate() reads config.
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR", "") or _default_kaggle_config_dir()
    _ensure_kaggle_config_dir(config_dir)
    api = KaggleApi()
    api.authenticate()
    return api


def _load_local_score_priors(higher_is_better: bool) -> dict[str, float]:
    """Load public score priors from locally cached tracking CSVs.

    Expected columns: date,status,public,ref,file,desc
    """

    candidate_paths = [
        REPO_ROOT / "scratch" / "submission_tracking.csv",
        REPO_ROOT / "submission_tracking.csv",
    ]

    scores: dict[str, float] = {}
    for path in candidate_paths:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            status = str(row.get("status", "") or "")
            if "COMPLETE" not in status:
                continue

            file_name = str(row.get("file", "") or "")
            if not file_name:
                continue

            public = row.get("public", None)
            try:
                val = float(public)
            except Exception:
                continue

            if file_name not in scores:
                scores[file_name] = val
            else:
                if higher_is_better:
                    scores[file_name] = max(scores[file_name], val)
                else:
                    scores[file_name] = min(scores[file_name], val)

    return scores


def _load_local_metric_priors() -> dict[str, float]:
    """Load local validation metric priors from scratch artifacts.

    These are *not* Kaggle public scores; they are internal metrics like holdout AUC.
    Used as a fallback to populate predicted scores when Kaggle listing is unavailable.
    """

    candidate_paths = [
        REPO_ROOT / "scratch" / "unsubmitted_candidates.csv",
        REPO_ROOT / "scratch" / "shortlist_rank.csv",
    ]

    scores: dict[str, float] = {}
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue

        # unsubmitted_candidates.csv: path,metric_value
        # shortlist_rank.csv: path,metric_value
        if "path" not in df.columns or "metric_value" not in df.columns:
            continue

        for _, row in df.iterrows():
            p = str(row.get("path", "") or "")
            if not p:
                continue
            try:
                val = float(row.get("metric_value"))
            except Exception:
                continue

            base = Path(p).name
            if not base:
                continue
            if base not in scores:
                scores[base] = val
            else:
                # For metric priors, keep the best observed value.
                scores[base] = max(scores[base], val)

    return scores


@st.cache_data(ttl=30)
def _fetch_submissions_cached(competition: str, limit: int) -> pd.DataFrame:
    api = _authenticate_kaggle()
    subs = api.competition_submissions(
        competition=competition,
        group=SubmissionGroup.SUBMISSION_GROUP_ALL,
        sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
        page_size=max(1, min(limit, 100)),
    )

    rows = []
    for s in (subs or []):
        rows.append(
            {
                "date": _fmt_dt(getattr(s, "date", None)),
                "status": str(getattr(s, "status", "")),
                "public": _score_str(getattr(s, "public_score", None)),
                "ref": str(getattr(s, "ref", "")),
                "file": str(getattr(s, "file_name", "")),
                "desc": str(getattr(s, "description", "") or ""),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def _fetch_score_priors_cached(competition: str, limit: int, higher_is_better: bool) -> dict[str, float]:
    """Return a mapping of uploaded file_name -> best known public_score (heuristic priors)."""
    api = _authenticate_kaggle()
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
            if higher_is_better:
                scores[file_name] = max(scores[file_name], val)
            else:
                scores[file_name] = min(scores[file_name], val)

    return scores


def _score_index(score_by_file: dict[str, float]) -> dict[str, float]:
    """Index by both filename and stem for easier matching."""
    out: dict[str, float] = {}
    for k, v in score_by_file.items():
        out[k] = v
        if k.lower().endswith(".csv"):
            out[k[:-4]] = v
    return out


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def _extract_components(recipe: str) -> list[str]:
    # Heuristic extraction: look for tokens that resemble our submission stems.
    tokens = _TOKEN_RE.findall(recipe or "")
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _match_score_token(token: str, score_idx: dict[str, float], higher_is_better: bool) -> tuple[str, float] | None:
    """Match a component token to a score index.

    Tries exact match first, then a conservative substring match for alias tokens
    (e.g. 'autogluon' -> 'submission_v21_autogluon_full').
    """

    if not token:
        return None

    if token in score_idx:
        return token, score_idx[token]

    tok = token.lower()
    if tok in {"best_public", "best"} and score_idx:
        items = list(score_idx.items())
        if higher_is_better:
            k, v = max(items, key=lambda kv: kv[1])
        else:
            k, v = min(items, key=lambda kv: kv[1])
        return k, v

    # Avoid overly broad fuzzy matching for very short tokens.
    if len(token) < 5:
        return None
    candidates: list[tuple[str, float]] = []
    for k, v in score_idx.items():
        kl = k.lower()
        if tok in kl:
            candidates.append((k, v))

    if not candidates:
        return None

    # Choose best score among substring matches.
    if higher_is_better:
        best = max(candidates, key=lambda kv: kv[1])
    else:
        best = min(candidates, key=lambda kv: kv[1])
    return best


def _estimate_score(recipe: str, score_idx: dict[str, float], higher_is_better: bool) -> tuple[float | None, str]:
    """Return (estimated_score, evidence_str). Purely heuristic.

    In this UI, the score is intended to be AUC-based (local validation metric),
    not Kaggle public score.
    """
    comps = _extract_components(recipe)
    comp_scores: list[tuple[str, float]] = []
    for c in comps:
        m = _match_score_token(c, score_idx, higher_is_better=higher_is_better)
        if m is not None:
            comp_scores.append(m)

    if not comp_scores:
        return None, ""

    # Weighted blend patterns like 0.80*foo + 0.20*bar
    weight_pairs = re.findall(r"(0\.[0-9]+)\*([A-Za-z][A-Za-z0-9_]{2,})", recipe or "")
    weights: list[tuple[float, float]] = []
    for w_str, name in weight_pairs:
        try:
            w = float(w_str)
        except Exception:
            continue
        m = _match_score_token(name, score_idx, higher_is_better=higher_is_better)
        if m is not None:
            weights.append((w, m[1]))

    if weights:
        wsum = sum(w for w, _ in weights)
        if wsum > 0:
            est = sum(w * s for w, s in weights) / wsum
        else:
            est = sum(s for _, s in weights) / len(weights)
        ev = ", ".join([f"{n}:{s:.6f}" for n, s in comp_scores[:4]])
        return float(est), ev

    # Otherwise average known component scores.
    est = sum(s for _, s in comp_scores) / len(comp_scores)
    ev = ", ".join([f"{n}:{s:.6f}" for n, s in comp_scores[:4]])
    return float(est), ev


def _build_all_candidates_table(manifest_paths: list[Path], score_idx: dict[str, float]) -> pd.DataFrame:
    """Build a de-duped table of all candidate rows across manifests.

    Dedupe key is absolute file path.
    """

    out_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    submitted_idx = _load_submitted_index()

    for mp in manifest_paths:
        try:
            rows = _read_manifest(mp)
        except Exception:
            continue

        for r in rows:
            abs_file = str(_abs_path(r["file"]))

            # Hide anything we've already submitted (local ledger or tracking cache).
            base_name = Path(abs_file).name
            base_stem = base_name[:-4] if base_name.lower().endswith(".csv") else base_name
            if abs_file in submitted_idx or base_name in submitted_idx or base_stem in submitted_idx:
                continue

            if abs_file in seen:
                continue
            seen.add(abs_file)

            p = Path(abs_file)
            est, ev = _estimate_score(r["recipe"], score_idx, higher_is_better=True)
            # Streamlit renders None as the string "None" in some cases; use NaN for blank.
            est_cell = float("nan") if est is None else float(est)
            out_rows.append(
                {
                    "submit": False,
                    "file": r["file"],
                    "exists": "yes" if p.exists() else "no",
                    "size_kb": f"{(p.stat().st_size / 1024.0):.1f}" if p.exists() else "",
                    "predicted_score": est_cell,
                    "predicted_score_evidence": ev,
                    "recipe": r["recipe"],
                    "manifest": str(mp.relative_to(REPO_ROOT)).replace("\\", "/"),
                }
            )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    # Sort best-first when we have predicted_score.
    def _sort_key(row):
        pred = row.get("predicted_score")
        try:
            if pred is None or pred == "" or pd.isna(pred):
                return (1, 0.0)
        except Exception:
            return (1, 0.0)
        try:
            return (0, -float(pred))
        except Exception:
            return (0, 0.0)

    df = df.sort_values(by=list(df.columns), key=lambda col: col)  # stable baseline
    df = df.assign(_k=df.apply(_sort_key, axis=1)).sort_values(by="_k").drop(columns=["_k"])
    return df


def _run_script(script: str, args: list[str]) -> tuple[int, str]:
    # Scripts live under scripts/ in this repo.
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (cp.stdout or "") + ("\n" if cp.stdout and cp.stderr else "") + (cp.stderr or "")
        return cp.returncode, out.strip()
    except Exception as e:
        return 1, f"Failed to run {script}: {type(e).__name__}: {e}"


def _submit_manifest_rows(competition: str, rows: list[dict[str, str]], top_n: int, contains: str, tag: str) -> list[str]:
    to_submit = rows
    if contains:
        to_submit = [r for r in to_submit if contains in r["recipe"]]

    to_submit = to_submit[: max(0, int(top_n))]
    if not to_submit:
        raise ValueError("No matching rows to submit")

    api = _authenticate_kaggle()
    # Optional sanity check (some environments get 401 on ListSubmissions even though submit works).
    try:
        api.competition_submissions(competition=competition, page_size=1)
    except Exception:
        pass

    msgs = []
    for r in to_submit:
        path = _abs_path(r["file"])  # supports scratch/... etc
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        msg = f"AUTO {tag}: {r['recipe']}"
        api.competition_submit(file_name=str(path), message=msg, competition=competition)
        msgs.append(f"Submitted: {path.name} | {msg}")

    return msgs


def main() -> None:
    st.set_page_config(page_title="Kaggle Submissions", layout="wide")

    st.title("Kaggle Submission Console")

    with st.sidebar:
        st.header("Settings")
        competition = st.text_input("Competition", value="playground-series-s5e12")
        show_pending = st.checkbox("Show pending submissions", value=False)
        pending_limit = st.number_input("Pending fetch limit", min_value=1, max_value=100, value=30, step=5)
        pending_desc_contains = st.text_input("Pending desc contains", value="AUTO")

        st.divider()
        st.caption("Predicted score (AUC)")
        st.caption("Uses local validation metrics from submissions/*.csv (and scratch/*.csv for legacy)")
        # AUC is higher-is-better; keep as fixed behavior for scoring.

        st.divider()
        st.caption("Auth")
        default_dir = _default_kaggle_config_dir()
        if not os.environ.get("KAGGLE_CONFIG_DIR") and default_dir:
            _ensure_kaggle_config_dir(default_dir)
        config_dir = st.text_input("KAGGLE_CONFIG_DIR", value=os.environ.get("KAGGLE_CONFIG_DIR", default_dir))
        apply_auth = st.button("Apply credentials path")
        if apply_auth:
            _ensure_kaggle_config_dir(config_dir)
            _fetch_submissions_cached.clear()

        repo_token = REPO_ROOT / "kaggle.json"
        home_token = Path.home() / ".kaggle" / "kaggle.json"
        st.text(f"Repo token: {'found' if repo_token.exists() else 'missing'}")
        st.text(f"Home token: {'found' if home_token.exists() else 'missing'}")
        st.text(f"Active KAGGLE_CONFIG_DIR={os.environ.get('KAGGLE_CONFIG_DIR', '')}")
        st.text(f"Repo={REPO_ROOT}")
        if not (repo_token.exists() or home_token.exists() or config_dir):
            st.warning(
                "Kaggle credentials not detected. Put kaggle.json in this repo or in ~/.kaggle, "
                "or set KAGGLE_CONFIG_DIR to the folder containing kaggle.json."
            )

    metric_priors = _load_local_metric_priors()
    score_idx = _score_index(metric_priors)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Potential submissions")
        st.caption("All candidates across manifests. Select rows to submit.")

        manifest_paths = _list_manifest_paths()
        if not manifest_paths:
            st.info("No manifest files found yet.")
        else:
            base_df = _build_all_candidates_table(manifest_paths, score_idx)
            if base_df.empty:
                st.info("No candidate rows found in manifests.")
            else:
                sig = _manifest_signature(manifest_paths)
                editor_key = f"candidates_editor_{sig}"
                edited = st.data_editor(
                    base_df,
                    width="stretch",
                    hide_index=True,
                    disabled=[
                        "file",
                        "exists",
                        "size_kb",
                        "predicted_score",
                        "predicted_score_evidence",
                        "recipe",
                        "manifest",
                    ],
                    column_config={
                        "submit": st.column_config.CheckboxColumn(required=False),
                        "predicted_score": st.column_config.NumberColumn(format="%.6f"),
                    },
                    key=editor_key,
                )
                missing_count = int((edited["exists"] == "no").sum()) if not edited.empty else 0
                if missing_count:
                    st.warning(f"{missing_count} candidates are missing on disk")
                selected_count = int(edited["submit"].sum()) if not edited.empty else 0
                st.caption(f"Selected: {selected_count}")
                if metric_priors and not edited.empty:
                    st.caption(
                        "‘predicted_score’ is a heuristic estimate from component AUCs (local validation metrics)."
                    )

        if show_pending:
            st.divider()
            st.subheader("Pending submissions")
            refresh = st.button("Refresh pending")
            if refresh:
                _fetch_submissions_cached.clear()

            try:
                df = _fetch_submissions_cached(competition=competition, limit=int(pending_limit))
            except Exception as e:
                st.error(f"Failed to fetch pending: {type(e).__name__}: {e}")
                st.info(
                    "If you see 401 Unauthorized, your kaggle.json token is missing/invalid for this process. "
                    "Set KAGGLE_CONFIG_DIR in the sidebar to the folder containing kaggle.json, then Refresh."
                )
                df = pd.DataFrame(columns=["date", "status", "public", "ref", "file", "desc"])

            if not df.empty:
                df = df[~df["status"].str.contains("COMPLETE", na=False)]
                if pending_desc_contains:
                    df = df[df["desc"].str.contains(pending_desc_contains, na=False)]

            st.dataframe(df, width="stretch", hide_index=True)

    with col_right:
        st.subheader("Next steps")

        st.markdown("**Generate candidates**")
        gen_blends = st.button("Generate blend pack")
        if gen_blends:
            code, out = _run_script("prepare_tomorrow_blend_pack.py", ["--manifest", "tomorrow_manifest_v2.csv"])
            if code == 0:
                st.success("Blend pack generated")
            else:
                st.error("Blend pack failed")
            if out:
                st.code(out)

        gen_pa = st.button("Generate PA-gate pack")
        if gen_pa:
            code, out = _run_script("prepare_tomorrow_pa_gate_pack.py", ["--manifest", "tomorrow_manifest_pa.csv"])
            if code == 0:
                st.success("PA-gate pack generated")
            else:
                st.error("PA-gate pack failed")
            if out:
                st.code(out)

        st.divider()

        st.markdown("**Submit selected**")
        tag_default = datetime.now(UTC).strftime("%Y-%m-%d")
        tag = st.text_input("Tag", value=tag_default)

        confirm = st.checkbox("I understand this submits to Kaggle and uses daily quota", value=False)
        submit = st.button("Submit selected")
        if submit:
            if not confirm:
                st.error("Confirmation required")
            else:
                try:
                    # Locate the active editor key (latest candidates table)
                    manifest_paths = _list_manifest_paths()
                    sig = _manifest_signature(manifest_paths)
                    editor_key = f"candidates_editor_{sig}"
                    edited_df = st.session_state.get(editor_key)
                    if edited_df is None or not isinstance(edited_df, pd.DataFrame) or edited_df.empty:
                        raise ValueError("No candidates table available")

                    picked = edited_df[edited_df["submit"] == True]  # noqa: E712
                    if picked.empty:
                        raise ValueError("No rows selected")

                    rows = [
                        {"file": str(r["file"]), "recipe": str(r["recipe"])}
                        for _, r in picked.iterrows()
                    ]

                    # Submit only selected rows.
                    msgs = _submit_manifest_rows(competition=competition, rows=rows, top_n=len(rows), contains="", tag=tag)
                    for m in msgs:
                        st.success(m)

                    # Record locally so submitted rows disappear from the list.
                    _append_submitted_ledger(competition=competition, rows=rows)

                    _fetch_submissions_cached.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Submit failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
