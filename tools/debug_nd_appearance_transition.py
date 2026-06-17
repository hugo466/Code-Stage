import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.dune_nd.plot_iss23_scan_maps_physical_params import (
    POINTS_CSV,
    SUMMARY_CSV,
    add_dk2nu_probability_amplitudes,
    load_merged_table,
)
from plots.dune_nd.plot_iss23_scan_max_event_ratio_map import (
    FLUX_DIR,
    GLOBES,
    PANELS,
    build_mixing,
    panel_components,
    read_flux,
    read_source_profile,
    read_xsec,
    SOURCE_PROFILE_FHC,
    SOURCE_PROFILE_RHC,
)


OUT_DIR = ROOT / "data" / "dune_nd" / "scan_maps" / "debug"
FIG_DIR = ROOT / "figures" / "dune_nd" / "scan_maps" / "debug"
POINTS_OUT = OUT_DIR / "appearance_transition_point_debug.csv"
BINS_OUT = OUT_DIR / "appearance_transition_binned_debug.csv"
REPORT_OUT = OUT_DIR / "appearance_transition_report.txt"
MIXING_SLOPES_OUT = OUT_DIR / "appearance_transition_mixing_axis_slopes.csv"

APP_PANELS = ("FHC_app", "RHC_app")
COMPONENTS = ("nc", "numu", "beam", "signal")
COLORS = {
    "total": "black",
    "signal": "tab:red",
    "numu": "tab:green",
    "beam": "tab:blue",
    "nc": "tab:orange",
}

MIXING_X_AXES = (
    ("Ue4_sq", r"$|U_{e4}|^2$"),
    ("Umu4_sq", r"$|U_{\mu4}|^2$"),
    ("Ue4_sq_Umu4_sq", r"$|U_{e4}|^2 |U_{\mu4}|^2$"),
)
Y_COLUMNS = (
    ("total", "max_abs_rel_percent", r"total $|\Delta N/N|$ [%]"),
    ("signal", "signal_abs_contrib_percent", r"signal contribution [%]"),
    ("beam", "beam_abs_contrib_percent", r"beam contribution [%]"),
    ("numu", "numu_abs_contrib_percent", r"$\nu_\mu$ mis-ID contribution [%]"),
    ("nc", "nc_abs_contrib_percent", r"NC contribution [%]"),
)


def prepare_tables(df: pd.DataFrame):
    fluxes = {
        "FHC": read_flux(FLUX_DIR / "flux_dune_neutrino_ND_globes.txt"),
        "RHC": read_flux(FLUX_DIR / "flux_dune_antineutrino_ND_globes.txt"),
    }
    profiles = {
        "FHC": read_source_profile(SOURCE_PROFILE_FHC),
        "RHC": read_source_profile(SOURCE_PROFILE_RHC),
    }
    cc = read_xsec(GLOBES / "xsec" / "xsec_cc.dat")
    nc = read_xsec(GLOBES / "xsec" / "xsec_nc.dat")
    u = build_mixing(df)
    masses4 = np.column_stack(
        [
            np.zeros(len(df), dtype=float),
            df["dm21_calc_eV2"].to_numpy(dtype=float),
            df["dm31_calc_eV2"].to_numpy(dtype=float),
            df["dm41_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    return fluxes, profiles, cc, nc, u, masses4, masses4[:, :3]


def appearance_debug_for_panel(df: pd.DataFrame, panel: str, e_min=0.5, e_max=8.0, n_bins=30) -> pd.DataFrame:
    fluxes, profiles, cc, nc, u, masses4, masses3 = prepare_tables(df)
    mode = "RHC" if panel.startswith("RHC") else "FHC"
    energies = e_min + (np.arange(n_bins, dtype=float) + 0.5) * ((e_max - e_min) / n_bins)
    width = (e_max - e_min) / n_bins

    best_abs_rel = np.full(len(df), -np.inf, dtype=float)
    rows = {
        "point_id": df["point_id"].to_numpy(dtype=int),
        "panel": np.full(len(df), panel, dtype=object),
        "probability_amplitude": df[f"{panel}_probability_amplitude"].to_numpy(dtype=float),
        "dm41_calc_eV2": df["dm41_calc_eV2"].to_numpy(dtype=float),
        "Ue4_abs": df["Ue4_abs"].to_numpy(dtype=float),
        "Umu4_abs": df["Umu4_abs"].to_numpy(dtype=float),
        "Ue4Umu4_abs": df["Ue4Umu4_abs"].to_numpy(dtype=float),
        "best_energy_GeV": np.zeros(len(df), dtype=float),
        "total3": np.zeros(len(df), dtype=float),
        "total4": np.zeros(len(df), dtype=float),
        "total_rel": np.zeros(len(df), dtype=float),
    }
    for component in COMPONENTS:
        rows[f"{component}3"] = np.zeros(len(df), dtype=float)
        rows[f"{component}4"] = np.zeros(len(df), dtype=float)
        rows[f"{component}_contrib_rel"] = np.zeros(len(df), dtype=float)
        rows[f"{component}_fraction3"] = np.zeros(len(df), dtype=float)

    for energy in energies:
        comp3 = panel_components(panel, fluxes[mode], cc, nc, u, masses3, energy, width, profiles[mode], 3)
        comp4 = panel_components(panel, fluxes[mode], cc, nc, u, masses4, energy, width, profiles[mode], 4)
        total3 = sum(comp3.values())
        total4 = sum(comp4.values())
        rel = np.divide(total4 - total3, total3, out=np.zeros_like(total4), where=np.abs(total3) > 1.0e-300)
        update = np.abs(rel) > best_abs_rel
        if not np.any(update):
            continue
        best_abs_rel[update] = np.abs(rel[update])
        rows["best_energy_GeV"][update] = energy
        rows["total3"][update] = total3[update]
        rows["total4"][update] = total4[update]
        rows["total_rel"][update] = rel[update]
        for component in COMPONENTS:
            c3 = comp3[component]
            c4 = comp4[component]
            rows[f"{component}3"][update] = c3[update]
            rows[f"{component}4"][update] = c4[update]
            rows[f"{component}_contrib_rel"][update] = np.divide(
                c4[update] - c3[update],
                total3[update],
                out=np.zeros_like(c4[update]),
                where=np.abs(total3[update]) > 1.0e-300,
            )
            rows[f"{component}_fraction3"][update] = np.divide(
                c3[update],
                total3[update],
                out=np.zeros_like(c3[update]),
                where=np.abs(total3[update]) > 1.0e-300,
            )

    out = pd.DataFrame(rows)
    out["max_abs_rel_percent"] = 100.0 * np.abs(out["total_rel"])
    for component in COMPONENTS:
        out[f"{component}_abs_contrib_percent"] = 100.0 * np.abs(out[f"{component}_contrib_rel"])
    out["Ue4_sq"] = out["Ue4_abs"] ** 2
    out["Umu4_sq"] = out["Umu4_abs"] ** 2
    out["Ue4_sq_Umu4_sq"] = out["Ue4_sq"] * out["Umu4_sq"]
    return out


def binned_debug(point_debug: pd.DataFrame, n_bins=18) -> pd.DataFrame:
    rows = []
    for panel in APP_PANELS:
        sub = point_debug[(point_debug["panel"] == panel) & (point_debug["probability_amplitude"] > 0.0)].copy()
        lx = np.log10(sub["probability_amplitude"].to_numpy(dtype=float))
        edges = np.linspace(float(np.nanmin(lx)), float(np.nanmax(lx)), n_bins + 1)
        for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (lx >= lo) & (lx < hi)
            if np.count_nonzero(mask) < 6:
                continue
            chunk = sub.iloc[np.where(mask)[0]]
            row = {
                "panel": panel,
                "bin": idx,
                "n_points": len(chunk),
                "log10_x_low": lo,
                "log10_x_high": hi,
                "x_median": float(np.nanmedian(chunk["probability_amplitude"])),
                "total_abs_rel_percent_median": float(np.nanmedian(chunk["max_abs_rel_percent"])),
                "best_energy_GeV_median": float(np.nanmedian(chunk["best_energy_GeV"])),
            }
            for component in COMPONENTS:
                row[f"{component}_abs_contrib_percent_median"] = float(np.nanmedian(chunk[f"{component}_abs_contrib_percent"]))
                row[f"{component}_fraction3_median"] = float(np.nanmedian(chunk[f"{component}_fraction3"]))
            rows.append(row)
    return pd.DataFrame(rows)


def fit_slope_from_binned(bins: pd.DataFrame, panel: str, y_col: str, x_min=None, x_max=None):
    sub = bins[bins["panel"] == panel].copy()
    if x_min is not None:
        sub = sub[sub["x_median"] >= x_min]
    if x_max is not None:
        sub = sub[sub["x_median"] <= x_max]
    sub = sub[(sub["x_median"] > 0.0) & (sub[y_col] > 0.0)]
    if len(sub) < 3:
        return None
    lx = np.log10(sub["x_median"].to_numpy(dtype=float))
    ly = np.log10(sub[y_col].to_numpy(dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept, sub


def binned_median_curve(x: np.ndarray, y: np.ndarray, n_bins=18, min_points=8) -> pd.DataFrame:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < min_points:
        return pd.DataFrame()
    lx = np.log10(x[valid])
    yv = y[valid]
    xv = x[valid]
    edges = np.linspace(float(np.nanmin(lx)), float(np.nanmax(lx)), n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lx >= lo) & (lx < hi)
        if np.count_nonzero(mask) < min_points:
            continue
        rows.append(
            {
                "x_median": float(np.nanmedian(xv[mask])),
                "y_median": float(np.nanmedian(yv[mask])),
                "n_points": int(np.count_nonzero(mask)),
                "log10_x_low": lo,
                "log10_x_high": hi,
            }
        )
    return pd.DataFrame(rows)


def fit_binned_curve(curve: pd.DataFrame):
    if curve.empty or len(curve) < 3:
        return None
    valid = (curve["x_median"] > 0.0) & (curve["y_median"] > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    lx = np.log10(curve.loc[valid, "x_median"].to_numpy(dtype=float))
    ly = np.log10(curve.loc[valid, "y_median"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = intercept + slope * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return float(slope), float(intercept), float(r2)


def draw_mixing_axis_debug(point_debug: pd.DataFrame) -> pd.DataFrame:
    slope_rows = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for panel in APP_PANELS:
        sub = point_debug[point_debug["panel"] == panel].copy()
        fig, axes = plt.subplots(
            len(Y_COLUMNS),
            len(MIXING_X_AXES),
            figsize=(15.0, 17.0),
            sharex="col",
            constrained_layout=True,
        )
        for row_idx, (component, y_col, y_label) in enumerate(Y_COLUMNS):
            y = sub[y_col].to_numpy(dtype=float)
            for col_idx, (x_col, x_label) in enumerate(MIXING_X_AXES):
                ax = axes[row_idx, col_idx]
                x = sub[x_col].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
                ax.scatter(x[valid], y[valid], s=7, alpha=0.18, color=COLORS.get(component, "tab:blue"), edgecolors="none")
                curve = binned_median_curve(x, y)
                fit = fit_binned_curve(curve)
                if not curve.empty:
                    ax.plot(
                        curve["x_median"],
                        curve["y_median"],
                        color="black",
                        marker="x",
                        linewidth=1.35,
                        markersize=4,
                        label="binned median",
                    )
                if fit is not None:
                    slope, intercept, r2 = fit
                    xs = np.geomspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 100)
                    ys = 10.0 ** (intercept + slope * np.log10(xs))
                    ax.plot(xs, ys, color="0.25", linestyle=":", linewidth=1.1)
                    ax.text(
                        0.04,
                        0.92,
                        rf"$m={slope:.2f}$, $R^2={r2:.2f}$",
                        transform=ax.transAxes,
                        fontsize=8.5,
                        va="top",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                    )
                    slope_rows.append(
                        {
                            "panel": panel,
                            "component": component,
                            "x_axis": x_col,
                            "slope": slope,
                            "intercept": intercept,
                            "r2_log_binned": r2,
                            "n_binned_points": len(curve),
                        }
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(alpha=0.22, which="both")
                if row_idx == 0:
                    ax.set_title(x_label, fontsize=12, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(y_label)
                if row_idx == len(Y_COLUMNS) - 1:
                    ax.set_xlabel(x_label)
        fig.suptitle(
            f"{panel.replace('_', ' ')}: appearance component scaling with individual mixings",
            fontsize=15,
            fontweight="bold",
        )
        fig.savefig(FIG_DIR / f"appearance_transition_mixing_axes_{panel}.png", dpi=230, bbox_inches="tight")
        plt.close(fig)
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(MIXING_SLOPES_OUT, index=False)
    return slopes


def describe_range(bins: pd.DataFrame, panel: str, lo: float, hi: float, name: str) -> list[str]:
    sub = bins[(bins["panel"] == panel) & (bins["x_median"] >= lo) & (bins["x_median"] <= hi)].copy()
    if sub.empty:
        return [f"  {name}: no populated bins"]
    first = sub.iloc[0]
    last = sub.iloc[-1]
    lines = [
        (
            f"  {name}: {len(sub)} bins, x={sub['x_median'].min():.3e}..{sub['x_median'].max():.3e}, "
            f"total={first['total_abs_rel_percent_median']:.4g}%..{last['total_abs_rel_percent_median']:.4g}%"
        )
    ]
    for component in ("signal", "beam", "numu", "nc"):
        lines.append(
            (
                f"    {component:6s}: contribution="
                f"{first[f'{component}_abs_contrib_percent_median']:.4g}%.."
                f"{last[f'{component}_abs_contrib_percent_median']:.4g}%, "
                f"3nu fraction={first[f'{component}_fraction3_median']:.4g}.."
                f"{last[f'{component}_fraction3_median']:.4g}"
            )
        )
    return lines


def write_report(bins: pd.DataFrame) -> None:
    lines = [
        "Debug apparition ND: transition entre pente binned-median et pente globale",
        "",
        "Definitions:",
        "  x = max_E |Delta Pbar_mue^dk2nu(E)| for the corresponding horn mode.",
        "  y = max_E |N_4nu/N_active3nu - 1| in percent.",
        "  Component contribution = |N_component,4nu - N_component,3nu| / N_total,3nu.",
        "  Component fraction = N_component,3nu / N_total,3nu at the energy bin where y is maximal.",
        "",
    ]
    for panel in APP_PANELS:
        lines.append(panel)
        for y_col, label in (
            ("total_abs_rel_percent_median", "total"),
            ("signal_abs_contrib_percent_median", "signal"),
            ("beam_abs_contrib_percent_median", "beam"),
            ("numu_abs_contrib_percent_median", "numu"),
            ("nc_abs_contrib_percent_median", "nc"),
        ):
            fit = fit_slope_from_binned(bins, panel, y_col, x_min=1.0e-14, x_max=1.0e-8)
            if fit is not None:
                lines.append(f"  low-x binned slope {label:6s}: {fit[0]:.4g}")
        lines.extend(describe_range(bins, panel, 1.0e-14, 1.0e-8, "low-x regime used for binned fit"))
        lines.extend(describe_range(bins, panel, 1.0e-8, 1.0e-5, "transition regime"))
        lines.extend(describe_range(bins, panel, 1.0e-5, 1.0e-3, "high-x regime"))
        lines.append("")
    lines.extend(
        [
            "Interpretation guide:",
            "  If the low-x total slope is close to one, the event-ratio deviation tracks the dk2nu-averaged probability amplitude locally.",
            "  If the all-points/unweighted slope is smaller, the scatter is not a single power law: high-x appearance points saturate because the total selected sample includes large intrinsic-beam, mis-ID numu, and NC components.",
            "  In appearance, the signal fraction in the active-3nu denominator is tiny at low x, so the visible event deviation can be dominated by how the sterile point perturbs the background components in the bin where the total deviation is maximal.",
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_merged_table(POINTS_CSV, SUMMARY_CSV)
    df = add_dk2nu_probability_amplitudes(df)
    debug = pd.concat([appearance_debug_for_panel(df, panel) for panel in APP_PANELS], ignore_index=True)
    bins = binned_debug(debug)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    debug.to_csv(POINTS_OUT, index=False)
    bins.to_csv(BINS_OUT, index=False)
    slopes = draw_mixing_axis_debug(debug)
    write_report(bins)

    print(f"Point debug CSV: {POINTS_OUT}")
    print(f"Binned debug CSV: {BINS_OUT}")
    print(f"Debug report: {REPORT_OUT}")
    print(f"Mixing-axis slopes CSV: {MIXING_SLOPES_OUT}")
    for panel in APP_PANELS:
        print(f"{panel} mixing-axis checks")
        checks = (
            ("signal", "Ue4_sq_Umu4_sq"),
            ("beam", "Ue4_sq"),
            ("numu", "Umu4_sq"),
        )
        for component, x_axis in checks:
            row = slopes[(slopes["panel"] == panel) & (slopes["component"] == component) & (slopes["x_axis"] == x_axis)]
            if not row.empty:
                print(f"  {component:6s} vs {x_axis:15s}: slope={row.iloc[0]['slope']:.4g}, R2={row.iloc[0]['r2_log_binned']:.4g}")
    for panel in APP_PANELS:
        total_fit = fit_slope_from_binned(bins, panel, "total_abs_rel_percent_median", x_min=1.0e-14, x_max=1.0e-8)
        signal_fit = fit_slope_from_binned(bins, panel, "signal_abs_contrib_percent_median", x_min=1.0e-14, x_max=1.0e-8)
        print(panel)
        if total_fit:
            print(f"  low-x total binned slope: {total_fit[0]:.4g}")
        if signal_fit:
            print(f"  low-x signal-contribution binned slope: {signal_fit[0]:.4g}")


if __name__ == "__main__":
    main()
