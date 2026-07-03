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
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


POINTS_CSV = ROOT / "data" / "inverse_seesaw" / "3p2" / "inverse_construct_24_kept_points" / "inverse_construct_24_kept_points.csv"
OUT_FIG = ROOT / "figures" / "oscillations" / "3p2" / "delta_pmue_heatmap_iss24_points_4panels.png"
OUT_CSV = ROOT / "data" / "oscillations" / "3p2" / "delta_pmue_heatmap_iss24_points_4panels.csv"

BASELINE_KM = 0.574
ENERGY_MIN_GEV = 0.5
ENERGY_MAX_GEV = 6.0
ENERGY_STEP_GEV = 0.02
DM41_TARGETS = (0.1, 1.0, 10.0, 100.0)
DM41_REL_TOL = 0.01
DM54_BINS = np.linspace(0.1, 100.0, 91)
PHASE_COEFF = 1.267


def load_points() -> pd.DataFrame:
    df = pd.read_csv(POINTS_CSV)
    for column in ("pmns_pass", "eta_pass"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    selected = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    required = ["dm21_calc_eV2", "dm31_calc_eV2", "dm41_calc_eV2", "dm51_calc_eV2"]
    required += [f"U5_solver{i}_re" for i in range(1, 26)]
    required += [f"U5_solver{i}_im" for i in range(1, 26)]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    selected = selected[(selected["dm41_calc_eV2"] > 0.0) & (selected["dm51_calc_eV2"] > selected["dm41_calc_eV2"])].copy()
    if selected.empty:
        raise RuntimeError("Aucun point ISS(2,4) pmns_pass=1 eta_pass=1 avec diagonalisation exploitable.")
    selected["dm54_calc_eV2"] = selected["dm51_calc_eV2"] - selected["dm41_calc_eV2"]
    selected = selected[(selected["dm54_calc_eV2"] >= DM54_BINS[0]) & (selected["dm54_calc_eV2"] <= DM54_BINS[-1])].copy()
    return selected


def build_mixing(df: pd.DataFrame) -> np.ndarray:
    u = np.zeros((len(df), 5, 5), dtype=complex)
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c + 1
            u[:, r, c] = df[f"U5_solver{idx}_re"].to_numpy(dtype=float) + 1j * df[f"U5_solver{idx}_im"].to_numpy(dtype=float)
    return u


def probability_vacuum(u: np.ndarray, masses: np.ndarray, alpha: int, beta: int, energy: float, n_states: int) -> np.ndarray:
    p = np.ones(u.shape[0], dtype=float) if alpha == beta else np.zeros(u.shape[0], dtype=float)
    for i in range(n_states):
        mi2 = masses[:, i]
        for j in range(i + 1, n_states):
            mj2 = masses[:, j]
            phase = PHASE_COEFF * (mj2 - mi2) * BASELINE_KM / energy
            amp = u[:, alpha, i] * np.conjugate(u[:, beta, i]) * np.conjugate(u[:, alpha, j]) * u[:, beta, j]
            p -= 4.0 * np.real(amp) * np.sin(phase) ** 2
            p += 2.0 * np.imag(amp) * np.sin(2.0 * phase)
    return np.clip(p, 0.0, 1.0)


def compute_binned_heatmaps(points: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]], pd.DataFrame]:
    energies = np.arange(ENERGY_MIN_GEV, ENERGY_MAX_GEV + 0.5 * ENERGY_STEP_GEV, ENERGY_STEP_GEV)
    u = build_mixing(points)
    masses5 = np.column_stack(
        [
            np.zeros(len(points), dtype=float),
            points["dm21_calc_eV2"].to_numpy(dtype=float),
            points["dm31_calc_eV2"].to_numpy(dtype=float),
            points["dm41_calc_eV2"].to_numpy(dtype=float),
            points["dm51_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses5[:, :3]
    dm54 = points["dm54_calc_eV2"].to_numpy(dtype=float)
    dm41 = points["dm41_calc_eV2"].to_numpy(dtype=float)

    panels: list[dict[str, object]] = []
    csv_rows: list[dict[str, float | int | str]] = []
    for target in DM41_TARGETS:
        lo = target * (1.0 - DM41_REL_TOL)
        hi = target * (1.0 + DM41_REL_TOL)
        point_mask = np.abs(dm41 - target) <= DM41_REL_TOL * target
        panel_point_count = int(np.count_nonzero(point_mask))
        bin_index = np.digitize(dm54, DM54_BINS) - 1
        grid = np.full((len(DM54_BINS) - 1, len(energies)), np.nan)
        count_grid = np.zeros_like(grid, dtype=int)
        for y_idx in range(len(DM54_BINS) - 1):
            mask = point_mask & (bin_index == y_idx)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            local_u = u[mask]
            local_m5 = masses5[mask]
            local_m3 = masses3[mask]
            for e_idx, energy in enumerate(energies):
                p5 = probability_vacuum(local_u, local_m5, 1, 0, float(energy), 5)
                p3 = probability_vacuum(local_u, local_m3, 1, 0, float(energy), 3)
                delta = p5 - p3
                value = float(np.mean(delta))
                grid[y_idx, e_idx] = value
                count_grid[y_idx, e_idx] = count
                csv_rows.append(
                    {
                        "dm41_target_eV2": target,
                        "dm41_bin_low_eV2": lo,
                        "dm41_bin_high_eV2": hi,
                        "dm41_relative_tolerance": DM41_REL_TOL,
                        "dm54_bin_low_eV2": float(DM54_BINS[y_idx]),
                        "dm54_bin_high_eV2": float(DM54_BINS[y_idx + 1]),
                        "energy_GeV": float(energy),
                        "delta_P_mue_mean": value,
                        "point_count": count,
                    }
                )
        panels.append({"target": target, "low": lo, "high": hi, "grid": grid, "counts": count_grid, "point_count": panel_point_count})
    return energies, panels, pd.DataFrame(csv_rows)


def plot_heatmap(energies: np.ndarray, panels: list[dict[str, object]], points_count: int) -> None:
    all_values = np.concatenate([np.asarray(panel["grid"], dtype=float).ravel() for panel in panels])
    finite = all_values[np.isfinite(all_values)]
    abs_max = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if abs_max <= 0.0:
        abs_max = 1.0e-12
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.06])
    axes = [
        fig.add_subplot(grid_spec[0, 0]),
        fig.add_subplot(grid_spec[0, 1]),
        fig.add_subplot(grid_spec[1, 0]),
        fig.add_subplot(grid_spec[1, 1]),
    ]
    cbar_ax = fig.add_subplot(grid_spec[:, 2])

    x_edges = np.r_[energies - 0.5 * ENERGY_STEP_GEV, energies[-1] + 0.5 * ENERGY_STEP_GEV]
    image = None
    for ax, panel in zip(axes, panels):
        target = float(panel["target"])
        lo = float(panel["low"])
        hi = float(panel["high"])
        point_count = int(panel["point_count"])
        grid = np.asarray(panel["grid"], dtype=float)
        image = ax.pcolormesh(x_edges, DM54_BINS, grid, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(
            rf"$|\Delta m_{{41}}^2-{target:g}|/{target:g}\leq 1\%$" + "\n" + rf"({point_count} points)",
            pad=8,
        )

    for idx, ax in enumerate(axes):
        if idx in (2, 3):
            ax.set_xlabel("Energie [GeV]")
        if idx in (0, 2):
            ax.set_ylabel(r"$\Delta m_{54}^2=\Delta m_{51}^2-\Delta m_{41}^2$ [eV$^2$]")

    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label(r"$\langle \Delta P_{\mu e}\rangle$ ISS(2,4) $-$ active 3$\nu$", fontsize=10)
    fig.suptitle(
        rf"ISS(2,4) construct_24 points: $\Delta P_{{\mu e}}(E,\Delta m^2_{{54}})$ binned ({points_count} points)",
        fontsize=13,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)


def main() -> None:
    points = load_points()
    energies, panels, table = compute_binned_heatmaps(points)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    plot_heatmap(energies, panels, len(points))
    print(f"Definition heatmap de reference: dm54_eV2 = dm51_eV2 - dm41_eV2, donc dm51_eV2 = dm41_eV2 + dm54_eV2.")
    print(f"Points ISS24 utilises: {len(points)}")
    print(f"CSV sauvegarde: {OUT_CSV}")
    print(f"Figure sauvegardee: {OUT_FIG}")


if __name__ == "__main__":
    main()
