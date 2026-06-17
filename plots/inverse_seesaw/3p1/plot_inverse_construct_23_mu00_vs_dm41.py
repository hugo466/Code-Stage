#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_mu00_vs_dm41.png"


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")


def _finite_xyb(x, y, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=bool)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], b[mask]


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point disponible.")

    required = ["solve_ok", "pmns_pass", "eta_pass", "mu00_eV", "dm41_target_eV2"]
    _require_columns(df, required)

    for col in ["solve_ok", "pmns_pass", "eta_pass"]:
        df[col] = df[col].fillna(0).astype(int)

    df_sel = df[(df["solve_ok"] == 1) & (df["pmns_pass"] == 1)].copy()
    if df_sel.empty:
        raise RuntimeError("Aucun point solve_ok=1 et pmns_pass=1.")

    dm41 = df_sel["dm41_target_eV2"].to_numpy(float)
    mu00 = np.abs(df_sel["mu00_eV"].to_numpy(float))
    eta_mask = df_sel["eta_pass"].to_numpy(int) == 1

    dm41, mu00, eta_mask = _finite_xyb(dm41, mu00, eta_mask)
    positive = (dm41 > 0.0) & (mu00 > 0.0)
    dm41 = dm41[positive]
    mu00 = mu00[positive]
    eta_mask = eta_mask[positive]
    if dm41.size == 0:
        raise RuntimeError("Aucun point positif fini pour dm41_target_eV2 et |mu00_eV|.")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        dm41,
        mu00,
        s=8,
        alpha=0.22,
        color="#1f77b4",
        edgecolors="none",
        label="pmns_pass=1",
    )

    if np.any(eta_mask):
        ax.scatter(
            dm41[eta_mask],
            mu00[eta_mask],
            s=10,
            alpha=0.45,
            color="#d62728",
            edgecolors="none",
            label="eta_pass=1",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta m^2_{41}$ target [eV$^2$]")
    ax.set_ylabel(r"$|\mu_{00}|$ [eV]")
    ax.set_title(r"$|\mu_{00}|$ en fonction de $\Delta m^2_{41}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)

    n = int(dm41.size)
    n_eta = int(np.sum(eta_mask))
    fig.suptitle(f"N={n}, eta_pass={n_eta}", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
