from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_24_observables import finite_xy_eta, load_construct24_points, require_columns

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_mu00_vs_dm41.png"


def scatter(ax, x, y, eta, title, xlabel, ylabel):
    x, y, eta = finite_xy_eta(x, y, eta)
    ax.scatter(x, y, s=6, alpha=0.22, color="tab:blue", edgecolors="none", label="PMNS")
    if np.any(eta):
        ax.scatter(x[eta], y[eta], s=8, alpha=0.42, color="tab:red", edgecolors="none", label="PMNS+eta")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, which="both")


def main() -> None:
    df = load_construct24_points()
    required = ["eta_pass", "dm41_target_eV2", "dm51_target_eV2", "mu00_11_abs", "mu00_12_abs", "mu00_21_abs", "mu00_22_abs"]
    require_columns(df, required)
    eta = df["eta_pass"].astype(int).to_numpy() == 1
    mu00_00 = df["mu00_11_abs"].to_numpy(float)
    mu00_11 = df["mu00_22_abs"].to_numpy(float)
    mu00_01 = df["mu00_12_abs"].to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    scatter(
        axes[0, 0],
        df["dm41_target_eV2"],
        mu00_00,
        eta,
        r"$|\mu_{00,00}|$ vs $\Delta m^2_{41}$",
        r"$\Delta m^2_{41}$ [eV$^2$]",
        r"$|\mu_{00,00}|$ [eV]",
    )
    scatter(
        axes[0, 1],
        df["dm51_target_eV2"],
        mu00_11,
        eta,
        r"$|\mu_{00,11}|$ vs $\Delta m^2_{51}$",
        r"$\Delta m^2_{51}$ [eV$^2$]",
        r"$|\mu_{00,11}|$ [eV]",
    )
    scatter(
        axes[1, 0],
        df["dm41_target_eV2"],
        mu00_01,
        eta,
        r"$|\mu_{00,01}|$ vs $\Delta m^2_{41}$",
        r"$\Delta m^2_{41}$ [eV$^2$]",
        r"$|\mu_{00,01}|$ [eV]",
    )
    scatter(
        axes[1, 1],
        df["dm51_target_eV2"],
        mu00_01,
        eta,
        r"$|\mu_{00,01}|$ vs $\Delta m^2_{51}$",
        r"$\Delta m^2_{51}$ [eV$^2$]",
        r"$|\mu_{00,01}|$ [eV]",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"ISS(2,4) construct_24 - bloc mu00 ({len(df)} points)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
