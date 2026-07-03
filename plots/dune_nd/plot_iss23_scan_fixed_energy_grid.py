import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from plot_iss23_scan_max_event_ratio_map import (
    ACTIVE_FLAVORS,
    AVOGADRO,
    GLOBES,
    PANELS,
    POINTS_CSV,
    POT_PER_YEAR,
    SOURCE_PROFILE_FHC,
    SOURCE_PROFILE_RHC,
    TARGET_MASS_KT,
    EXPOSURE_YEARS,
    build_mixing,
    panel_components,
    read_dk2nu_flux_from_z,
    read_source_profile,
    read_xsec,
)


ENERGIES_GEV = (1.0, 2.0, 4.0, 6.0)
BIN_WIDTH_GEV = 0.25
OUT = ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps" / "iss23_scan_map_mixing_fixed_energy_grid.png"
OUT_CSV = ROOT / "data" / "dune_nd" / "scan_maps" / "iss23_fixed_energy_event_ratio_grid.csv"

LABELS = {
    "FHC_app": r"FHC $\nu_e$ appearance",
    "RHC_app": r"RHC $\bar{\nu}_e$ appearance",
    "FHC_dis": r"FHC $\nu_\mu$ disappearance",
    "RHC_dis": r"RHC $\bar{\nu}_\mu$ disappearance",
}


def load_points() -> pd.DataFrame:
    df = pd.read_csv(POINTS_CSV)
    for column in ["pmns_pass", "eta_pass"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    selected = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    required = [
        "point_id",
        "dm21_calc_eV2",
        "dm31_calc_eV2",
        "dm41_calc_eV2",
        "U_solver_14",
        "U_solver_24",
        "U_solver_44",
    ]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    selected = selected[selected["dm41_calc_eV2"] > 0.0].copy()
    if selected.empty:
        raise RuntimeError("Aucun point ISS(2,3) pmns_pass=1 et eta_pass=1 exploitable.")
    return selected


def compute_fixed_energy_ratios(points: pd.DataFrame) -> pd.DataFrame:
    u = build_mixing(points)
    masses4 = np.column_stack(
        [
            np.zeros(len(points), dtype=float),
            points["dm21_calc_eV2"].to_numpy(dtype=float),
            points["dm31_calc_eV2"].to_numpy(dtype=float),
            points["dm41_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses4[:, :3]

    fluxes = {
        "FHC": read_dk2nu_flux_from_z(SOURCE_PROFILE_FHC),
        "RHC": read_dk2nu_flux_from_z(SOURCE_PROFILE_RHC),
    }
    profiles = {
        "FHC": read_source_profile(SOURCE_PROFILE_FHC),
        "RHC": read_source_profile(SOURCE_PROFILE_RHC),
    }
    cc = read_xsec(GLOBES / "xsec" / "xsec_cc.dat")
    nc = read_xsec(GLOBES / "xsec" / "xsec_nc.dat")
    scale = POT_PER_YEAR * EXPOSURE_YEARS * TARGET_MASS_KT * 1.0e9 * AVOGADRO

    rows = points[["point_id", "dm41_calc_eV2", "theta24_deg"]].copy()
    rows["Ue4Umu4_abs"] = pd.to_numeric(points["U_solver_14"], errors="coerce").abs() * pd.to_numeric(
        points["U_solver_24"], errors="coerce"
    ).abs()
    rows["Umu4_sq"] = pd.to_numeric(points["U_solver_24"], errors="coerce").abs() ** 2

    for panel in PANELS:
        mode = "RHC" if panel.startswith("RHC") else "FHC"
        for energy in ENERGIES_GEV:
            comp3 = panel_components(panel, fluxes[mode], cc, nc, u, masses3, energy, BIN_WIDTH_GEV, profiles[mode], 3)
            comp4 = panel_components(panel, fluxes[mode], cc, nc, u, masses4, energy, BIN_WIDTH_GEV, profiles[mode], 4)
            n3 = sum(comp3.values()) * scale
            n4 = sum(comp4.values()) * scale
            rel = np.divide(n4 - n3, n3, out=np.zeros_like(n4), where=np.abs(n3) > 1.0e-30)
            rows[f"{panel}_E{energy:g}_rel"] = rel
            rows[f"{panel}_E{energy:g}_percent"] = 100.0 * rel
    return rows


def plot_grid(table: pd.DataFrame) -> None:
    value_cols = [f"{panel}_E{energy:g}_percent" for panel in PANELS for energy in ENERGIES_GEV]
    values = table[value_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    max_abs = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if max_abs <= 0.0:
        max_abs = 1.0e-12
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, axes = plt.subplots(4, 4, figsize=(15.0, 13.0), sharex=True)
    image = None
    for row, panel in enumerate(PANELS):
        is_app = panel.endswith("_app")
        y_col = "Ue4Umu4_abs" if is_app else "Umu4_sq"
        y_label = r"$|U_{e4}U_{\mu4}|$" if is_app else r"$|U_{\mu4}|^2$"
        for col, energy in enumerate(ENERGIES_GEV):
            ax = axes[row, col]
            image = ax.scatter(
                table["dm41_calc_eV2"],
                table[y_col],
                c=table[f"{panel}_E{energy:g}_percent"],
                s=12,
                alpha=0.76,
                cmap="coolwarm",
                norm=norm,
                edgecolors="none",
            )
            if row == 0:
                ax.set_title(rf"$E_\nu={energy:g}$ GeV", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(LABELS[panel] + "\n" + y_label, fontsize=10)
            if row == len(PANELS) - 1:
                ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]", fontsize=10)
            ax.grid(alpha=0.22)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.012, fraction=0.025)
        cbar.set_label(r"$N_{4\nu}/N_{active3\nu}-1$ [%] at fixed $E_\nu$")

    fig.suptitle("DUNE ND 6.5 yr/mode - ISS(2,3)", fontsize=14, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = load_points()
    table = compute_fixed_energy_ratios(points)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    plot_grid(table)
    print(f"Points utilises: {len(table)}")
    print(f"CSV sauvegarde: {OUT_CSV}")
    print(f"Figure sauvegardee: {OUT}")
    value_cols = [f"{panel}_E{energy:g}_percent" for panel in PANELS for energy in ENERGIES_GEV]
    finite = table[value_cols].to_numpy(dtype=float)
    print(f"Amplitude signee: min={np.nanmin(finite):.6g}% max={np.nanmax(finite):.6g}%")


if __name__ == "__main__":
    main()
