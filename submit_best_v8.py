import os
import sys
from datetime import datetime, timezone

from kaggle.api.kaggle_api_extended import KaggleApi


COMP = "playground-series-s5e12"
CANDIDATES = [
    ("submission_v8_gated_base_sub50_sigmoid_sharp.csv", "V8 gated base+sub50 sharp sigmoid (p_test)"),
    ("submission_v8_gated_base_sub50_sigmoid.csv", "V8 gated base+sub50 sigmoid (p_test)"),
    ("submission_v8_blend_base_weighted_60_40.csv", "V8 blend base + importance-weighted 60/40"),
]


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    # Quick auth sanity check (will 401 if creds are invalid).
    try:
        api.competitions_list(page=1)
    except Exception as e:
        _die(
            "Kaggle API auth failed (likely invalid/expired key).\n"
            "Fix by downloading a fresh API token from Kaggle -> Account -> API -> Create New Token,\n"
            "then place kaggle.json at ~/.kaggle/kaggle.json (chmod 600).\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for filename, base_msg in CANDIDATES:
        path = os.path.abspath(filename)
        if not os.path.exists(path):
            _die(f"Missing submission file: {path}")

        msg = f"{base_msg} [{ts}]"
        print(f"Submitting {path} -> {COMP} | message: {msg}")
        api.competition_submit(file_name=path, message=msg, competition=COMP)


if __name__ == "__main__":
    main()
