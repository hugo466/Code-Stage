from __future__ import annotations

import numpy as np
import pandas as pd

import plot_delta_pmue_heatmap_iss24_points as base


OUT_FIG = base.ROOT / "figures" / "oscillations" / "3p2" / "delta_pmumu_heatmap_iss24_points_4panels.png"
OUT_CSV = base.ROOT / "data" / "oscillations" / "3p2" / "delta_pmumu_heatmap_iss24_points_4panels.csv"


def compute_binned_heatmaps(points: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]], pd.DataFrame]:
    energies = np.arange(base.ENERGY_MIN_GEV, base.ENERGY_MAX_GEV + 0.5 * base.ENERGY_STEP_GEV, base.ENERGY_STEP_GEV)
    u = base.build_mixing(points)
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
    csv_rows: list[dict[str, float | int]] = []
    for target in base.DM41_TARGETS:
        lo = target * (1.0 - base.DM41_REL_TOL)
        hi = target * (1.0 + base.DM41_REL_TOL)
        point_mask = np.abs(dm41 - target) <= base.DM41_REL_TOL * target
        panel_point_count = int(np.count_nonzero(point_mask))
        bin_index = np.digitize(dm54, base.DM54_BINS) - 1
        grid = np.full((len(base.DM54_BINS) - 1, len(energies)), np.nan)
        count_grid = np.zeros_like(grid, dtype=int)
        for y_idx in range(len(base.DM54_BINS) - 1):
            mask = point_mask & (bin_index == y_idx)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            local_u = u[mask]
            local_m5 = masses5[mask]
            local_m3 = masses3[mask]
            for e_idx, energy in enumerate(energies):
                p5 = base.probability_vacuum(local_u, local_m5, 1, 1, float(energy), 5)
                p3 = base.probability_vacuum(local_u, local_m3, 1, 1, float(energy), 3)
                value = float(np.mean(p5 - p3))
                grid[y_idx, e_idx] = value
                count_grid[y_idx, e_idx] = count
                csv_rows.append(
                    {
                        "dm41_target_eV2": target,
                        "dm41_bin_low_eV2": lo,
                        "dm41_bin_high_eV2": hi,
                        "dm41_relative_tolerance": base.DM41_REL_TOL,
                        "dm54_bin_low_eV2": float(base.DM54_BINS[y_idx]),
                        "dm54_bin_high_eV2": float(base.DM54_BINS[y_idx + 1]),
                        "energy_GeV": float(energy),
                        "delta_P_mumu_mean": value,
                        "point_count": count,
                    }
                )
        panels.append({"target": target, "low": lo, "high": hi, "grid": grid, "counts": count_grid, "point_count": panel_point_count})
    return energies, panels, pd.DataFrame(csv_rows)


def plot_heatmap(energies: np.ndarray, panels: list[dict[str, object]], points_count: int) -> None:
    old_out_fig = base.OUT_FIG
    try:
        base.OUT_FIG = OUT_FIG
        base.plot_heatmap(energies, panels, points_count)
    finally:
        base.OUT_FIG = old_out_fig


def main() -> None:
    points = base.load_points()
    energies, panels, table = compute_binned_heatmaps(points)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    plot_heatmap(energies, panels, len(points))

    # Fix labels/titles produced by the shared plotter.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

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
    x_edges = np.r_[energies - 0.5 * base.ENERGY_STEP_GEV, energies[-1] + 0.5 * base.ENERGY_STEP_GEV]
    image = None
    for ax, panel in zip(axes, panels):
        target = float(panel["target"])
        point_count = int(panel["point_count"])
        image = ax.pcolormesh(x_edges, base.DM54_BINS, np.asarray(panel["grid"], dtype=float), shading="auto", cmap="RdBu_r", norm=norm)
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
    colorbar.set_label(r"$\langle \Delta P_{\mu\mu}\rangle$ ISS(2,4) $-$ active 3$\nu$", fontsize=10)
    fig.suptitle(
        rf"ISS(2,4) construct_24 points: $\Delta P_{{\mu\mu}}(E,\Delta m^2_{{54}})$ binned ({len(points)} points)",
        fontsize=13,
        fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)

    print("Definition heatmap de reference: dm54_eV2 = dm51_eV2 - dm41_eV2, donc dm51_eV2 = dm41_eV2 + dm54_eV2.")
    print(f"Points ISS24 utilises: {len(points)}")
    print(f"CSV sauvegarde: {OUT_CSV}")
    print(f"Figure sauvegardee: {OUT_FIG}")


if __name__ == "__main__":
    main()
