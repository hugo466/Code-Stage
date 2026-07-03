import os
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


POINTS_CSV = ROOT / "data" / "inverse_seesaw" / "3p1" / "inverse_construct_23_kept_points" / "inverse_construct_23_kept_points.csv"
SUMMARY_CSV = ROOT / "data" / "dune_nd" / "scan_maps" / "iss23_active3nu_max_event_ratio_map.csv"
SOURCE_PROFILE_FHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_FHC_ND_raw.csv"
SOURCE_PROFILE_RHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_RHC_ND_raw.csv"
OUT_THETA24 = ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps" / "iss23_scan_map_theta24.png"
OUT_MIXING = ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps" / "iss23_scan_map_mixing_observables.png"
OUT_AMPLITUDE = ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps" / "iss23_max_event_ratio_vs_expected_amplitude.png"
OUT_PROB_AMPLITUDE = ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps" / "iss23_max_event_ratio_vs_dk2nu_probability_amplitude.png"

BASELINE_KM = 0.574
PHASE_COEFF = 1.267

PANELS = ("FHC_app", "RHC_app", "FHC_dis", "RHC_dis")
LABELS = {
    "FHC_app": r"FHC $\nu_e$ appearance",
    "RHC_app": r"RHC $\bar{\nu}_e$ appearance",
    "FHC_dis": r"FHC $\nu_\mu$ disappearance",
    "RHC_dis": r"RHC $\bar{\nu}_\mu$ disappearance",
}


def load_merged_table(points_csv: Path, summary_csv: Path) -> pd.DataFrame:
    points = pd.read_csv(points_csv)
    summary = pd.read_csv(summary_csv)

    for col in ["pmns_pass", "eta_pass"]:
        points[col] = pd.to_numeric(points[col], errors="coerce").fillna(0).astype(int)

    points = points[(points["pmns_pass"] == 1) & (points["eta_pass"] == 1)].copy()
    mixing_columns = [f"U_solver_{r}{c}" for r in range(1, 5) for c in range(1, 5)]
    complex_mixing_columns = [
        name
        for r in range(1, 5)
        for c in range(1, 5)
        for name in (f"U_solver_re_{r}{c}", f"U_solver_im_{r}{c}", f"U_solver_phase_deg_{r}{c}")
        if name in points.columns
    ]
    merged = summary.merge(
        points[
            [
                "point_id",
                "dm21_calc_eV2",
                "dm31_calc_eV2",
                "dm41_calc_eV2",
                *mixing_columns,
                *complex_mixing_columns,
            ]
        ],
        on=["point_id", "dm41_calc_eV2"],
        how="inner",
    )

    merged["theta24_deg"] = pd.to_numeric(merged["theta24_deg"], errors="coerce")
    merged["Ue4_abs"] = pd.to_numeric(merged["U_solver_14"], errors="coerce").abs()
    merged["Umu4_abs"] = pd.to_numeric(merged["U_solver_24"], errors="coerce").abs()
    merged["Ue4Umu4_abs"] = merged["Ue4_abs"] * merged["Umu4_abs"]
    merged["Umu4_sq"] = merged["Umu4_abs"] ** 2
    merged["appearance_amplitude"] = 4.0 * merged["Ue4_abs"] ** 2 * merged["Umu4_abs"] ** 2
    merged["disappearance_amplitude"] = 4.0 * merged["Umu4_sq"] * (1.0 - merged["Umu4_sq"])
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.dropna(
        subset=[
            "dm41_calc_eV2",
            "dm21_calc_eV2",
            "dm31_calc_eV2",
            "theta24_deg",
            "Ue4Umu4_abs",
            "Umu4_sq",
            "appearance_amplitude",
            "disappearance_amplitude",
            *mixing_columns,
        ]
    ).copy()


def color_norm(df: pd.DataFrame) -> Normalize | None:
    cols = [f"{panel}_max_abs_percent" for panel in PANELS]
    values = df[cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0e-12
    return Normalize(vmin=vmin, vmax=vmax)


def scatter_panel(ax, df: pd.DataFrame, panel: str, y_col: str, y_label: str, norm: Normalize | None, x_log: bool = False) -> None:
    image = ax.scatter(
        df["dm41_calc_eV2"],
        df[y_col],
        c=df[f"{panel}_max_abs_percent"],
        s=16,
        alpha=0.78,
        cmap="viridis",
        norm=norm,
        edgecolors="none",
    )
    ax.set_title(LABELS[panel], fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    if x_log:
        ax.set_xscale("log")
    return image


def draw_figure(df: pd.DataFrame, out_path: Path, mode: str, x_log: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=True)
    norm = color_norm(df)
    image = None

    for ax, panel in zip(axes.flat, PANELS):
        if mode == "theta24":
            y_col = "theta24_deg"
            y_label = r"$\theta_{24}$ [deg]"
        else:
            if panel.endswith("_app"):
                y_col = "Ue4Umu4_abs"
                y_label = r"$|U_{e4}U_{\mu4}|$"
            else:
                y_col = "Umu4_sq"
                y_label = r"$|U_{\mu4}|^2$"
        image = scatter_panel(ax, df, panel, y_col, y_label, norm, x_log=x_log)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(r"$\max_E |N_{4\nu}/N_{active3\nu}-1|$ [%]")

    if mode == "theta24":
        title = rf"DUNE ND 6.5 yr/mode - scan map ({len(df)} points)"
    else:
        title = rf"DUNE ND 6.5 yr/mode - scan maps ({len(df)} points)"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_amplitude_scaling(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=False)
    panel_specs = {
        "FHC_app": ("appearance_amplitude", r"$4 |U_{e4}|^2 |U_{\mu4}|^2$"),
        "RHC_app": ("appearance_amplitude", r"$4 |U_{e4}|^2 |U_{\mu4}|^2$"),
        "FHC_dis": ("disappearance_amplitude", r"$4 |U_{\mu4}|^2(1-|U_{\mu4}|^2)$"),
        "RHC_dis": ("disappearance_amplitude", r"$4 |U_{\mu4}|^2(1-|U_{\mu4}|^2)$"),
    }

    for ax, panel in zip(axes.flat, PANELS):
        x_col, x_label = panel_specs[panel]
        x = df[x_col].to_numpy(dtype=float)
        y = df[f"{panel}_max_abs_percent"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        ax.scatter(x[valid], y[valid], s=13, alpha=0.42, color="tab:blue", edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[panel], fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\max_E |\Delta N/N|$ [%]")
        ax.grid(alpha=0.25, which="both")

        if np.count_nonzero(valid) > 2:
            lx = np.log10(x[valid])
            ly = np.log10(y[valid])
            slope, intercept = np.polyfit(lx, ly, 1)
            edges = np.linspace(float(np.nanmin(lx)), float(np.nanmax(lx)), 18)
            bin_centers = []
            bin_medians = []
            min_points_per_bin = 6
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (lx >= lo) & (lx < hi)
                if np.count_nonzero(mask) >= min_points_per_bin:
                    bin_centers.append(0.5 * (lo + hi))
                    bin_medians.append(float(np.nanmedian(ly[mask])))
            median_slope = None
            median_intercept = None
            fit_bin_centers = np.asarray(bin_centers)
            fit_bin_medians = np.asarray(bin_medians)
            if panel.endswith("_app") and len(fit_bin_centers) > 0:
                app_fit_mask = (fit_bin_centers >= -14.0) & (fit_bin_centers <= -8.0)
                fit_bin_centers = fit_bin_centers[app_fit_mask]
                fit_bin_medians = fit_bin_medians[app_fit_mask]
            if len(fit_bin_centers) > 2:
                median_slope, median_intercept = np.polyfit(fit_bin_centers, fit_bin_medians, 1)
            xs = np.geomspace(float(np.nanmin(x[valid])), float(np.nanmax(x[valid])), 150)
            ys = 10.0 ** (intercept + slope * np.log10(xs))
            ax.plot(xs, ys, color="0.35", linewidth=1.0, linestyle=":", label=rf"unweighted = {slope:.2f}")
            if median_slope is not None:
                median_ys = 10.0 ** (median_intercept + median_slope * np.log10(xs))
                ax.plot(xs, median_ys, color="black", linewidth=1.4, label=rf"binned median = {median_slope:.2f}")
                ax.scatter(
                    10.0 ** fit_bin_centers,
                    10.0 ** fit_bin_medians,
                    s=28,
                    color="black",
                    marker="x",
                    linewidths=1.0,
                    zorder=4,
                )
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("DUNE ND 6.5 yr/mode - ISS(2,3)", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def read_source_profile(path: Path) -> dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]]:
    table = pd.read_csv(path)
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for (flavor, e_low, e_high), group in table.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high"], sort=False):
        z = 0.5 * (group["z_decay_m_bin_low"].to_numpy(dtype=float) + group["z_decay_m_bin_high"].to_numpy(dtype=float))
        w = group["weight"].to_numpy(dtype=float)
        total = float(np.sum(w))
        if total > 0.0:
            profile[(str(flavor), float(e_low), float(e_high))] = (z, w / total)
    return profile


def source_weights(profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]], flavor: str, energy: float) -> tuple[np.ndarray, np.ndarray]:
    for (row_flavor, e_low, e_high), zw in profile.items():
        if row_flavor == flavor and e_low <= energy < e_high:
            return zw
    return np.asarray([0.0]), np.asarray([1.0])


def mixing_array(df: pd.DataFrame) -> np.ndarray:
    u = np.zeros((len(df), 4, 4), dtype=complex)
    has_complex_columns = all(
        f"U_solver_re_{r + 1}{c + 1}" in df.columns and f"U_solver_im_{r + 1}{c + 1}" in df.columns
        for r in range(4)
        for c in range(4)
    )
    for r in range(4):
        for c in range(4):
            if has_complex_columns:
                re = df[f"U_solver_re_{r + 1}{c + 1}"].to_numpy(dtype=float)
                im = df[f"U_solver_im_{r + 1}{c + 1}"].to_numpy(dtype=float)
                u[:, r, c] = re + 1j * im
            else:
                u[:, r, c] = df[f"U_solver_{r + 1}{c + 1}"].to_numpy(dtype=float)
    return u


def probability_c_like(
    u: np.ndarray,
    masses: np.ndarray,
    alpha: int,
    beta: int,
    energy: float,
    baseline_km: float,
    n_states: int,
    antineutrino: bool = False,
) -> np.ndarray:
    p = np.ones(u.shape[0], dtype=float) if alpha == beta else np.zeros(u.shape[0], dtype=float)
    im_sign = -1.0 if antineutrino else 1.0
    for i in range(n_states):
        mi2 = masses[:, i]
        for j in range(i + 1, n_states):
            mj2 = masses[:, j]
            phase = PHASE_COEFF * (mj2 - mi2) * baseline_km / energy
            a = (
                u[:, alpha, i]
                * np.conjugate(u[:, beta, i])
                * np.conjugate(u[:, alpha, j])
                * u[:, beta, j]
            )
            p -= 4.0 * np.real(a) * np.sin(phase) ** 2
            p += im_sign * 2.0 * np.imag(a) * np.sin(2.0 * phase)
    return np.clip(p, 0.0, 1.0)


def source_averaged_probability(
    u: np.ndarray,
    masses: np.ndarray,
    alpha: int,
    beta: int,
    flavor: str,
    energy: float,
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
    n_states: int,
) -> np.ndarray:
    z_values, weights = source_weights(profile, flavor, energy)
    out = np.zeros(u.shape[0], dtype=float)
    antineutrino = flavor.endswith("bar")
    for z_m, weight in zip(z_values, weights):
        baseline_km = max(0.0, BASELINE_KM - z_m * 1.0e-3)
        out += weight * probability_c_like(u, masses, alpha, beta, energy, baseline_km, n_states, antineutrino)
    return out


def add_dk2nu_probability_amplitudes(df: pd.DataFrame, e_min: float = 0.5, e_max: float = 8.0, n_bins: int = 30) -> pd.DataFrame:
    out = df.copy()
    u = mixing_array(out)
    masses4 = np.column_stack(
        [
            np.zeros(len(out), dtype=float),
            out["dm21_calc_eV2"].to_numpy(dtype=float),
            out["dm31_calc_eV2"].to_numpy(dtype=float),
            out["dm41_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses4[:, :3]
    profiles = {
        "FHC": read_source_profile(SOURCE_PROFILE_FHC),
        "RHC": read_source_profile(SOURCE_PROFILE_RHC),
    }
    energies = e_min + (np.arange(n_bins, dtype=float) + 0.5) * ((e_max - e_min) / n_bins)
    specs = {
        "FHC_app": ("FHC", "numu", 1, 0),
        "RHC_app": ("RHC", "numubar", 1, 0),
        "FHC_dis": ("FHC", "numu", 1, 1),
        "RHC_dis": ("RHC", "numubar", 1, 1),
    }

    for panel, (mode, flavor, alpha, beta) in specs.items():
        amp = np.zeros(len(out), dtype=float)
        for energy in energies:
            p3 = source_averaged_probability(u, masses3, alpha, beta, flavor, energy, profiles[mode], 3)
            p4 = source_averaged_probability(u, masses4, alpha, beta, flavor, energy, profiles[mode], 4)
            amp = np.maximum(amp, np.abs(p4 - p3))
        out[f"{panel}_probability_amplitude"] = amp
    return out


def draw_probability_amplitude_scaling(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=False)
    for ax, panel in zip(axes.flat, PANELS):
        x_col = f"{panel}_probability_amplitude"
        x = df[x_col].to_numpy(dtype=float)
        y = df[f"{panel}_max_abs_percent"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        ax.set_title(LABELS[panel], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\max_E |\Delta \bar{P}_{\alpha\beta}^{dk2nu}(E)|$")
        ax.set_ylabel(r"$\max_E |\Delta N/N|$ [%]")
        ax.grid(alpha=0.25, which="both")

        if np.count_nonzero(valid) == 0:
            ax.text(0.5, 0.5, "no positive values", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(x[valid], y[valid], s=13, alpha=0.42, color="tab:blue", edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if np.count_nonzero(valid) > 2:
            lx = np.log10(x[valid])
            ly = np.log10(y[valid])
            slope, intercept = np.polyfit(lx, ly, 1)
            edges = np.linspace(float(np.nanmin(lx)), float(np.nanmax(lx)), 18)
            bin_centers = []
            bin_medians = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (lx >= lo) & (lx < hi)
                if np.count_nonzero(mask) >= 6:
                    bin_centers.append(0.5 * (lo + hi))
                    bin_medians.append(float(np.nanmedian(ly[mask])))
            median_slope = None
            median_intercept = None
            fit_bin_centers = np.asarray(bin_centers)
            fit_bin_medians = np.asarray(bin_medians)
            if panel.endswith("_app") and len(fit_bin_centers) > 0:
                app_fit_mask = (fit_bin_centers >= -14.0) & (fit_bin_centers <= -8.0)
                fit_bin_centers = fit_bin_centers[app_fit_mask]
                fit_bin_medians = fit_bin_medians[app_fit_mask]
            if len(fit_bin_centers) > 2:
                median_slope, median_intercept = np.polyfit(fit_bin_centers, fit_bin_medians, 1)
            xs = np.geomspace(float(np.nanmin(x[valid])), float(np.nanmax(x[valid])), 150)
            ax.plot(xs, 10.0 ** (intercept + slope * np.log10(xs)), color="0.35", linewidth=1.0, linestyle=":", label=rf"unweighted = {slope:.2f}")
            if median_slope is not None:
                ax.plot(xs, 10.0 ** (median_intercept + median_slope * np.log10(xs)), color="black", linewidth=1.4, label=rf"binned median = {median_slope:.2f}")
                ax.scatter(10.0 ** fit_bin_centers, 10.0 ** fit_bin_medians, s=28, color="black", marker="x", linewidths=1.0, zorder=4)
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("DUNE ND 6.5 yr/mode - ISS(2,3)", fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ISS(2,3) DUNE ND scan-map figures from a summary CSV.")
    parser.add_argument("--points-csv", type=Path, default=POINTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "dune_nd" / "iss23" / "scan_maps")
    parser.add_argument("--x-log", action="store_true", help="Use a logarithmic x-axis for dm41 scan maps.")
    args = parser.parse_args()

    out_theta24 = args.out_dir / "iss23_scan_map_theta24.png"
    out_mixing = args.out_dir / "iss23_scan_map_mixing_observables.png"
    out_amplitude = args.out_dir / "iss23_max_event_ratio_vs_expected_amplitude.png"
    out_prob_amplitude = args.out_dir / "iss23_max_event_ratio_vs_dk2nu_probability_amplitude.png"

    df = load_merged_table(args.points_csv, args.summary_csv)
    df = add_dk2nu_probability_amplitudes(df)
    draw_figure(df, out_theta24, mode="theta24", x_log=args.x_log)
    draw_figure(df, out_mixing, mode="mixing", x_log=args.x_log)
    draw_amplitude_scaling(df, out_amplitude)
    draw_probability_amplitude_scaling(df, out_prob_amplitude)
    print(f"Points utilises: {len(df)}")
    print(f"Figure theta24: {out_theta24}")
    print(f"Figure mixing observables: {out_mixing}")
    print(f"Figure amplitude scaling: {out_amplitude}")
    print(f"Figure dk2nu probability amplitude scaling: {out_prob_amplitude}")


if __name__ == "__main__":
    main()
