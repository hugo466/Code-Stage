#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_dm41_vs_zeta_norm.png"
NBINS_DM41 = 30
NBINS_ZETA = 30


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier de points conservés introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point conservé trouvé.")

    required = ["dm41_target_eV2", "zeta_norm", "eta_pass"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")

    dm41 = df["dm41_target_eV2"].to_numpy(dtype=float)
    zeta_norm = df["zeta_norm"].to_numpy(dtype=float)
    eta_pass = df["eta_pass"].to_numpy(dtype=int)

    dm41_edges = np.logspace(np.log10(dm41.min()), np.log10(dm41.max()), NBINS_DM41 + 1)
    zeta_edges = np.linspace(zeta_norm.min(), zeta_norm.max(), NBINS_ZETA + 1)

    total_counts, _, _ = np.histogram2d(dm41, zeta_norm, bins=[dm41_edges, zeta_edges])
    eta_counts, _, _ = np.histogram2d(dm41[eta_pass == 1], zeta_norm[eta_pass == 1], bins=[dm41_edges, zeta_edges])

    with np.errstate(divide="ignore", invalid="ignore"):
        acceptance = np.divide(eta_counts, total_counts, out=np.full_like(eta_counts, np.nan, dtype=float), where=total_counts > 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    mesh0 = axes[0].pcolormesh(dm41_edges, zeta_edges, total_counts.T, shading="auto", cmap="viridis")
    # axes[0].set_xscale("log")
    axes[0].set_title("Densité des points PMNS OK")
    axes[0].set_xlabel(r"$\Delta m_{41}^2$ [eV$^2$]")
    axes[0].set_ylabel(r"$\|\zeta\|$")
    cbar0 = fig.colorbar(mesh0, ax=axes[0])
    cbar0.set_label("Nombre de points")

    mesh1 = axes[1].pcolormesh(dm41_edges, zeta_edges, acceptance.T, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_xscale("log")
    axes[1].set_title(r"Acceptation $\eta$ parmi les points PMNS OK")
    axes[1].set_xlabel(r"$\Delta m_{41}^2$ [eV$^2$]")
    axes[1].set_ylabel(r"$\|\zeta\|$")
    cbar1 = fig.colorbar(mesh1, ax=axes[1])
    cbar1.set_label(r"Fraction $\eta$-pass")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
