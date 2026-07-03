from __future__ import annotations

from pathlib import Path

import pandas as pd

from inverse_construct_24_config import get_inverse_construct_24_csv_path


def load_kept_points_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    resolved = csv_path or get_inverse_construct_24_csv_path()
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved)
    for col in ("pmns_pass", "eta_pass", "solve_ok", "coherence_pass"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df
