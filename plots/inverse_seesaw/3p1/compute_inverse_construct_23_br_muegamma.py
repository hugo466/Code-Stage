#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np

from inverse_construct_23_config import get_inverse_construct_23_csv_path, get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

DATA_DIR = get_inverse_kept_points_dir()
UNIFIED_CSV = get_inverse_construct_23_csv_path()

# Same default value used by inverse_pmns_filter presets.
EXPERIMENTAL_LIMIT = 1.5e-13
ALPHA_EM = 1.0 / 137.035999084
M_W_EV = 80.379e9


def loop_function_g_gamma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)

    positive = x > 0.0
    if not np.any(positive):
        return g

    xp = x[positive]
    close_to_one = np.abs(xp - 1.0) < 1e-8

    gp = np.zeros_like(xp)
    gp[close_to_one] = 0.125

    normal = ~close_to_one
    xn = xp[normal]
    if xn.size > 0:
        gp[normal] = (
            -xn * (2.0 * xn * xn + 5.0 * xn - 1.0) / (4.0 * (1.0 - xn) ** 3)
            - 3.0 * xn * xn * xn * np.log(xn) / (2.0 * (1.0 - xn) ** 4)
        )

    g[positive] = gp
    return g


def compute_br_mu_to_e_gamma(df):
    required = ["theta14_deg", "theta24_deg", "dm41_target_eV2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes pour BR(mu->e gamma): {missing}")

    theta14 = np.deg2rad(df["theta14_deg"].to_numpy(dtype=float))
    theta24 = np.deg2rad(df["theta24_deg"].to_numpy(dtype=float))
    dm41 = np.maximum(df["dm41_target_eV2"].to_numpy(dtype=float), 0.0)

    # 3+1 light-sterile approximation from effective angles.
    ue4 = np.sin(theta14)
    umu4 = np.cos(theta14) * np.sin(theta24)

    x4 = dm41 / (M_W_EV * M_W_EV)
    g4 = loop_function_g_gamma(x4)
    amplitude = umu4 * ue4 * g4

    prefactor = 3.0 * ALPHA_EM / (32.0 * math.pi)
    return prefactor * amplitude * amplitude


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier de points conserves introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point trouve.")

    if "pmns_pass" not in df.columns:
        df["pmns_pass"] = 1
    if "eta_pass" not in df.columns:
        df["eta_pass"] = 0

    br = compute_br_mu_to_e_gamma(df)
    df_out = df.copy()
    df_out["br_muegamma_calc"] = br

    pmns_mask = df_out["pmns_pass"] == 1
    pmns_eta_mask = pmns_mask & (df_out["eta_pass"] == 1)
    above_limit_mask = df_out["br_muegamma_calc"] > EXPERIMENTAL_LIMIT

    pmns_total = int(pmns_mask.sum())
    eta_total = int(pmns_eta_mask.sum())
    pmns_above = int((pmns_mask & above_limit_mask).sum())
    eta_above = int((pmns_eta_mask & above_limit_mask).sum())

    UNIFIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(UNIFIED_CSV, index=False)

    print("BR(mu->e gamma) calcule en approximation 3+1 (etat sterile leger uniquement).")
    print(f"Limite experimentale utilisee: BR < {EXPERIMENTAL_LIMIT:.3e}")
    print(f"Points PMNS OK: {pmns_total}")
    print(f"Points PMNS OK au-dessus de la limite: {pmns_above}")
    print(f"Points PMNS+eta OK: {eta_total}")
    print(f"Points PMNS+eta OK au-dessus de la limite: {eta_above}")
    print(f"CSV unique mis a jour: {UNIFIED_CSV}")


if __name__ == "__main__":
    main()
