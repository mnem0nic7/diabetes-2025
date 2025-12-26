import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Defaults:
    target: str = "diagnosed_diabetes"
    out: str = "submission_segment_rank_blend.csv"
    # Nina's example mixing (asc_weight, desc_weight)
    asc_weight: float = 0.30
    desc_weight: float = 0.70


def _read_sub(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or target not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    out = df[["id", target]].copy()
    out["id"] = out["id"].astype(int)
    out[target] = out[target].astype(float)
    return out


def _assert_same_ids(frames: list[pd.DataFrame]) -> None:
    base_ids = frames[0]["id"].to_numpy()
    for i, df in enumerate(frames[1:], start=2):
        ids = df["id"].to_numpy()
        if len(ids) != len(base_ids) or not np.array_equal(ids, base_ids):
            raise ValueError(f"ID mismatch between submission[1] and submission[{i}]")


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in (s or "").split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in (s or "").split(",") if x.strip()]


def parse_segments(s: str) -> list[tuple[float, float]]:
    """Parse a segment string like '0.00003:0.00177,0.00180:0.00293,0.00300:0.00474'."""
    segs: list[tuple[float, float]] = []
    if not (s or "").strip():
        return segs
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad segment '{p}'. Expected lo:hi")
        lo_s, hi_s = p.split(":", 1)
        lo = float(lo_s.strip())
        hi = float(hi_s.strip())
        if not (lo < hi):
            raise ValueError(f"Bad segment '{p}': require lo < hi")
        segs.append((lo, hi))
    return segs


def parse_delta_matrix(s: str) -> np.ndarray:
    """Parse semicolon-separated vectors: '0.11,-0.03; -0.03,0.11'."""
    rows: list[list[float]] = []
    for row_s in [r.strip() for r in (s or "").split(";") if r.strip()]:
        rows.append(_parse_float_list(row_s))
    if not rows:
        raise ValueError("Empty delta-matrix")
    width = len(rows[0])
    if width == 0:
        raise ValueError("Empty delta row")
    for r in rows:
        if len(r) != width:
            raise ValueError("All delta rows must have the same length")
    return np.asarray(rows, dtype=float)


def nina_delta_matrix_m4() -> np.ndarray:
    """Nina's example deltas for M=4 (percentages / 100)."""
    return np.asarray(
        [
            [11, -3, -1, -7],
            [-3, 11, -7, -1],
            [-1, -7, 11, -3],
            [-7, -1, -3, 11],
        ],
        dtype=float,
    ) / 100.0


def segment_rank_blend(
    preds: np.ndarray,
    base_weights: np.ndarray,
    delta_matrix: np.ndarray,
    segments: list[tuple[float, float]],
    different: list[int],
    asc_weight: float,
    desc_weight: float,
) -> np.ndarray:
    """Nina-style rank-conditioned blend.

    preds: shape (m, n)
    base_weights: shape (m,)
    delta_matrix: shape (k, m) where each row is a per-rank adjustment vector
    segments: list of 3 ranges; else bucket handled implicitly
    different: list of 4 indices into delta_matrix for (seg1, seg2, seg3, else)
    """
    if preds.ndim != 2:
        raise ValueError("preds must be 2D (m, n)")
    m, n = preds.shape

    base_weights = np.asarray(base_weights, dtype=float)
    if base_weights.shape != (m,):
        raise ValueError(f"base_weights must have shape ({m},)")

    delta_matrix = np.asarray(delta_matrix, dtype=float)
    if delta_matrix.ndim != 2 or delta_matrix.shape[1] != m:
        raise ValueError(f"delta_matrix must have shape (k, {m})")

    if len(different) != 4:
        raise ValueError("different must have 4 ints: seg1,seg2,seg3,else")
    if any((d < 0 or d >= delta_matrix.shape[0]) for d in different):
        raise ValueError("different indices out of range for delta_matrix")

    if not np.isfinite([asc_weight, desc_weight]).all():
        raise ValueError("asc/desc weights must be finite")
    if abs((asc_weight + desc_weight) - 1.0) > 1e-6:
        raise ValueError("asc_weight + desc_weight must sum to 1.0")

    # Disagreement score per row
    mxm = preds.max(axis=0) - preds.min(axis=0)

    # Determine which delta-vector index to use per row
    if segments and len(segments) != 3:
        raise ValueError("segments must have exactly 3 ranges (else is implicit)")

    sel = np.full(n, different[3], dtype=int)
    if segments:
        (s1_lo, s1_hi), (s2_lo, s2_hi), (s3_lo, s3_hi) = segments
        sel[(mxm > s1_lo) & (mxm <= s1_hi)] = different[0]
        sel[(mxm > s2_lo) & (mxm <= s2_hi)] = different[1]
        sel[(mxm > s3_lo) & (mxm <= s3_hi)] = different[2]

    # Rank positions per row for desc/asc
    order_desc = np.argsort(-preds, axis=0)
    order_asc = np.argsort(preds, axis=0)

    # rank_pos[j, i] = position (0..m-1) of model j in row i's ordering
    rank_pos_desc = np.empty((m, n), dtype=int)
    rank_pos_asc = np.empty((m, n), dtype=int)
    cols = np.arange(n)
    for pos in range(m):
        rank_pos_desc[order_desc[pos, :], cols] = pos
        rank_pos_asc[order_asc[pos, :], cols] = pos

    # Compute per-row coefficients = base_weights[model] + delta[rank]
    # We need delta vector per row (from sel), then index by rank_pos.
    deltas_for_rows = delta_matrix[sel, :]  # (n, m) per-rank

    # Build coefficient matrices (m, n)
    coef_desc = base_weights[:, None] + deltas_for_rows.T[rank_pos_desc, cols]
    coef_asc = base_weights[:, None] + deltas_for_rows.T[rank_pos_asc, cols]

    y_desc = np.sum(preds * coef_desc, axis=0)
    y_asc = np.sum(preds * coef_asc, axis=0)

    return desc_weight * y_desc + asc_weight * y_asc


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Nina-style rank-conditioned blending over existing submissions. "
            "Optionally uses mx-m segmentation to choose different per-rank correction tables."
        )
    )
    p.add_argument("--inputs", required=True, help="Comma-separated submission CSV paths")
    p.add_argument("--target", default=Defaults.target)
    p.add_argument("--out", default=Defaults.out)

    p.add_argument(
        "--weights",
        default="",
        help="Comma-separated base weights per input; default is uniform 1/M",
    )

    p.add_argument(
        "--delta-matrix",
        default="",
        help=(
            "Semicolon-separated per-rank adjustment vectors. Example (M=4): "
            "'0.11,-0.03,-0.01,-0.07; -0.03,0.11,-0.07,-0.01; -0.01,-0.07,0.11,-0.03; -0.07,-0.01,-0.03,0.11'"
        ),
    )
    p.add_argument("--nina-deltas", action="store_true", help="Use Nina's M=4 delta matrix")

    p.add_argument(
        "--segments",
        default="",
        help="3 mx-m ranges as 'lo:hi,lo:hi,lo:hi' (else is implicit). If omitted, no segmentation is applied.",
    )
    p.add_argument(
        "--different",
        default="0,1,2,3",
        help="4 indices into delta-matrix for seg1,seg2,seg3,else",
    )

    p.add_argument("--asc", type=float, default=Defaults.asc_weight, help="Weight for asc-ordered blend")
    p.add_argument("--desc", type=float, default=Defaults.desc_weight, help="Weight for desc-ordered blend")

    p.add_argument("--clip", action="store_true", help="Clip output predictions to [0,1]")

    args = p.parse_args()

    in_paths = _parse_csv_list(args.inputs)
    if len(in_paths) < 2:
        raise SystemExit("Provide at least 2 inputs")

    missing = [p for p in in_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing input files:\n- " + "\n- ".join(missing))

    frames = [_read_sub(pth, args.target) for pth in in_paths]
    _assert_same_ids(frames)

    ids = frames[0]["id"].to_numpy(dtype=int)
    preds = np.stack([df[args.target].to_numpy(dtype=float) for df in frames], axis=0)

    m = preds.shape[0]

    if args.weights.strip():
        w = np.asarray(_parse_float_list(args.weights), dtype=float)
        if w.shape != (m,):
            raise SystemExit(f"--weights must have {m} floats")
    else:
        w = np.full(m, 1.0 / m, dtype=float)

    if args.nina_deltas:
        if m != 4:
            raise SystemExit("--nina-deltas requires exactly 4 inputs")
        delta_matrix = nina_delta_matrix_m4()
    else:
        if not args.delta_matrix.strip():
            raise SystemExit("Provide --delta-matrix or use --nina-deltas")
        delta_matrix = parse_delta_matrix(args.delta_matrix)

    segments = parse_segments(args.segments)
    different = _parse_int_list(args.different)
    if len(different) != 4:
        raise SystemExit("--different must be 4 ints")

    asc = float(args.asc)
    desc = float(args.desc)
    if abs((asc + desc) - 1.0) > 1e-6:
        raise SystemExit("--asc + --desc must sum to 1.0")

    y = segment_rank_blend(
        preds=preds,
        base_weights=w,
        delta_matrix=delta_matrix,
        segments=segments,
        different=different,
        asc_weight=asc,
        desc_weight=desc,
    )

    if args.clip:
        y = np.clip(y, 0.0, 1.0)

    pd.DataFrame({"id": ids, args.target: y}).to_csv(args.out, index=False)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
