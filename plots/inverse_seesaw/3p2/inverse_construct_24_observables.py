from __future__ import annotations

import numpy as np
import pandas as pd

from inverse_construct_24_kept_points import load_kept_points_dataframe


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")


def load_construct24_points(*, eta_only: bool = False) -> pd.DataFrame:
    df = load_kept_points_dataframe()
    if df.empty:
        raise RuntimeError("CSV construct_24 introuvable ou vide.")

    for col in ("pmns_pass", "eta_pass", "solve_ok", "coherence_pass"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    coherent = df.get("coherence_pass", pd.Series(1, index=df.index)).astype(int) == 1
    mask = (df.get("pmns_pass", 0).astype(int) == 1) & coherent
    if eta_only:
        mask &= df.get("eta_pass", 0).astype(int) == 1

    selected = df.loc[mask].copy()
    if selected.empty:
        raise RuntimeError("Aucun point construct_24 coherent avec pmns_pass=1.")
    return selected


def build_mu4_abs(df: pd.DataFrame) -> np.ndarray:
    columns = [
        "mu00_11_abs", "mu00_12_abs", "mu00_21_abs", "mu00_22_abs",
        "muH0_11_abs", "muH0_12_abs", "muH0_21_abs", "muH0_22_abs",
        "muH11_abs", "muH12_abs", "muH21_abs", "muH22_abs",
    ]
    require_columns(df, columns)
    n = len(df)
    mu4 = np.zeros((n, 4, 4), dtype=float)

    mu00 = np.stack(
        [
            np.column_stack([df["mu00_11_abs"], df["mu00_12_abs"]]),
            np.column_stack([df["mu00_21_abs"], df["mu00_22_abs"]]),
        ],
        axis=1,
    ).astype(float)
    muH0 = np.stack(
        [
            np.column_stack([df["muH0_11_abs"], df["muH0_12_abs"]]),
            np.column_stack([df["muH0_21_abs"], df["muH0_22_abs"]]),
        ],
        axis=1,
    ).astype(float)
    muH = np.stack(
        [
            np.column_stack([df["muH11_abs"], df["muH12_abs"]]),
            np.column_stack([df["muH21_abs"], df["muH22_abs"]]),
        ],
        axis=1,
    ).astype(float)

    mu4[:, 0:2, 0:2] = mu00
    mu4[:, 0:2, 2:4] = np.transpose(muH0, (0, 2, 1))
    mu4[:, 2:4, 0:2] = muH0
    mu4[:, 2:4, 2:4] = muH
    return mu4


def log_bins(values: np.ndarray, nbins: int = 70):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return nbins
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        vmin *= 0.9
        vmax *= 1.1
    return np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)


def finite_xy_eta(x, y, eta):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=bool)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], eta[mask]
