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

from inverse_construct_24_config import load_config_value
from inverse_construct_24_observables import finite_xy_eta, load_construct24_points, require_columns

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_rebuild_masses.png"


def plot_mass_panel(ax, target, calc, eta_mask, title):
    x, y, eta = finite_xy_eta(target, calc, eta_mask)
    positive = (x > 0.0) & (y > 0.0)
    x, y, eta = x[positive], y[positive], eta[positive]
    ax.scatter(x, y, s=5, alpha=0.22, color="tab:blue", edgecolors="none", label="PMNS")
    if np.any(eta):
        ax.scatter(x[eta], y[eta], s=8, alpha=0.40, color="tab:red", edgecolors="none", label="PMNS+eta")
    if x.size:
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        lo = max(lo, np.finfo(float).tiny)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("target [eV$^2$]")
    ax.set_ylabel("9x9 Takagi [eV$^2$]")
    ax.grid(alpha=0.25, which="both")


def main() -> None:
    df = load_construct24_points()
    required = [
        "eta_pass",
        "dm21_calc_eV2", "dm31_calc_eV2", "dm41_calc_eV2", "dm51_calc_eV2",
        "dm41_target_eV2", "dm51_target_eV2",
    ]
    require_columns(df, required)
    df["dm21_target_eV2"] = float(load_config_value("dm21_eV2", "7.49e-5"))
    df["dm31_target_eV2"] = float(load_config_value("dm31_eV2", "2.513e-3"))
    eta = df["eta_pass"].astype(int).to_numpy() == 1

    panels = [
        ("dm21", "dm21_target_eV2", "dm21_calc_eV2", r"$\Delta m^2_{21}$"),
        ("dm31", "dm31_target_eV2", "dm31_calc_eV2", r"$\Delta m^2_{31}$"),
        ("dm41", "dm41_target_eV2", "dm41_calc_eV2", r"$\Delta m^2_{41}$"),
        ("dm51", "dm51_target_eV2", "dm51_calc_eV2", r"$\Delta m^2_{51}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    for ax, (_, target_col, calc_col, title) in zip(axes.ravel(), panels):
        plot_mass_panel(ax, df[target_col].to_numpy(float), df[calc_col].to_numpy(float), eta, title)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"ISS(2,4) construct_24 - masses cible vs diagonalisation 9x9 ({len(df)} points)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
