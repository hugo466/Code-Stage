from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
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
from matplotlib.colors import Normalize


POINTS_CSV = ROOT / "data" / "inverse_seesaw" / "3p2" / "inverse_construct_24_kept_points" / "inverse_construct_24_kept_points.csv"
GLOBES = ROOT / "data" / "dune" / "2103.04797v2" / "dune_globes"
FLUX_DIR = ROOT / "data" / "dune" / "flux"
SOURCE_PROFILE_FHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_FHC_ND_raw.csv"
SOURCE_PROFILE_RHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_RHC_ND_raw.csv"
OUT_DIR = ROOT / "figures" / "dune_nd" / "iss24" / "scan_maps"
OUT_CSV = ROOT / "data" / "dune_nd" / "iss24" / "scan_maps" / "iss24_active3nu_max_event_ratio_map.csv"

AVOGADRO = 6.02214076e23
M2_TO_CM2 = 1.0e4
XSEC_SCALE_CM2 = 1.0e-38
POT_PER_YEAR = 11.0e20
EXPOSURE_YEARS = 6.5
TARGET_MASS_KT = 0.067
BASELINE_KM = 0.574
PHASE_COEFF = 1.267

FLAVORS = ["nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
ACTIVE_FLAVORS = ["nue", "numu", "nuebar", "numubar"]
FLAVOR_ALPHA = {"nue": 0, "numu": 1, "nutau": 2, "nuebar": 0, "numubar": 1, "nutaubar": 2}
PANELS = ("FHC_app", "RHC_app", "FHC_dis", "RHC_dis")
LABELS = {
    "FHC_app": r"FHC $\nu_e$ appearance",
    "RHC_app": r"RHC $\bar{\nu}_e$ appearance",
    "FHC_dis": r"FHC $\nu_\mu$ disappearance",
    "RHC_dis": r"RHC $\bar{\nu}_\mu$ disappearance",
}


def read_flux(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["E_GeV", *FLAVORS],
        decimal=",",
        comment="#",
        engine="python",
    ).apply(pd.to_numeric, errors="coerce").dropna(subset=["E_GeV"])

def read_dk2nu_flux_from_z(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for column in ["E_GeV_bin_low", "E_GeV_bin_high", "weight"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["flavor", "E_GeV_bin_low", "E_GeV_bin_high", "weight"])
    grouped = table.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high"], as_index=False)["weight"].sum()
    grouped["E_GeV"] = 0.5 * (grouped["E_GeV_bin_low"] + grouped["E_GeV_bin_high"])
    out = grouped.pivot_table(index="E_GeV", columns="flavor", values="weight", aggfunc="sum").reset_index()
    for flavor in FLAVORS:
        if flavor not in out:
            out[flavor] = 0.0
    return out[["E_GeV", *FLAVORS]].sort_values("E_GeV")


def read_xsec(path: Path) -> pd.DataFrame:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean:
            tokens.extend(clean.split())
    values = [float(token.replace(",", ".")) for token in tokens]
    rows = np.asarray(values, dtype=float).reshape((-1, 7))
    df = pd.DataFrame(rows, columns=["log10_E_GeV", *FLAVORS])
    df.insert(0, "E_GeV", np.power(10.0, df.pop("log10_E_GeV")))
    return df


def interp(df: pd.DataFrame, column: str, energy: float) -> float:
    return float(np.interp(energy, df["E_GeV"].to_numpy(dtype=float), df[column].to_numpy(dtype=float), left=0.0, right=0.0))


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


def build_mixing(df: pd.DataFrame) -> np.ndarray:
    u = np.zeros((len(df), 5, 5), dtype=complex)
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c + 1
            u[:, r, c] = df[f"U5_solver{idx}_re"].to_numpy(dtype=float) + 1j * df[f"U5_solver{idx}_im"].to_numpy(dtype=float)
    return u


def probability_source_average_channel(
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
    baselines_km = np.maximum(0.0, BASELINE_KM - z_values * 1.0e-3)
    im_sign = -1.0 if flavor.endswith("bar") else 1.0
    p = np.ones(u.shape[0], dtype=float) if alpha == beta else np.zeros(u.shape[0], dtype=float)
    for i in range(n_states):
        mi2 = masses[:, i]
        for j in range(i + 1, n_states):
            mj2 = masses[:, j]
            phase = PHASE_COEFF * (mj2 - mi2)[:, None] * baselines_km[None, :] / energy
            avg_sin2 = np.sin(phase) ** 2 @ weights
            avg_sin2phase = np.sin(2.0 * phase) @ weights
            a = u[:, alpha, i] * np.conjugate(u[:, beta, i]) * np.conjugate(u[:, alpha, j]) * u[:, beta, j]
            p -= 4.0 * np.real(a) * avg_sin2
            p += im_sign * 2.0 * np.imag(a) * avg_sin2phase
    return np.clip(p, 0.0, 1.0)


def probability_source_average(
    u: np.ndarray,
    masses: np.ndarray,
    alpha: int,
    beta: int | None,
    flavor: str,
    energy: float,
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
    n_states: int,
) -> np.ndarray:
    if beta is not None:
        return probability_source_average_channel(u, masses, alpha, beta, flavor, energy, profile, n_states)
    out = np.zeros(u.shape[0], dtype=float)
    for active_beta in range(3):
        out += probability_source_average_channel(u, masses, alpha, active_beta, flavor, energy, profile, n_states)
    return np.clip(out, 0.0, 1.0)


def cc_rate(flux, xsec, u, masses, initial, final, energy, width, profile, n_states) -> np.ndarray:
    alpha = FLAVOR_ALPHA[initial]
    beta = FLAVOR_ALPHA[final]
    phi = interp(flux, initial, energy) / M2_TO_CM2
    sigma = interp(xsec, final, energy) * energy * XSEC_SCALE_CM2
    prob = probability_source_average(u, masses, alpha, beta, initial, energy, profile, n_states)
    return phi * sigma * prob * width


def nc_rate_all(flux, xsec, u, masses, energy, width, profile, n_states) -> np.ndarray:
    total = np.zeros(u.shape[0], dtype=float)
    for flavor in ACTIVE_FLAVORS:
        alpha = FLAVOR_ALPHA[flavor]
        phi = interp(flux, flavor, energy) / M2_TO_CM2
        sigma = interp(xsec, flavor, energy) * energy * XSEC_SCALE_CM2
        prob = probability_source_average(u, masses, alpha, None, flavor, energy, profile, n_states)
        total += phi * sigma * prob * width
    return total


def panel_components(panel, flux, cc, nc, u, masses, energy, width, profile, n_states) -> dict[str, np.ndarray]:
    is_rhc = panel.startswith("RHC")
    is_app = panel.endswith("_app")
    right_mu = "numubar" if is_rhc else "numu"
    wrong_mu = "numu" if is_rhc else "numubar"
    right_e = "nuebar" if is_rhc else "nue"
    wrong_e = "nue" if is_rhc else "nuebar"
    if is_app:
        return {
            "nc": 0.025 * nc_rate_all(flux, nc, u, masses, energy, width, profile, n_states),
            "numu": 0.015
            * (
                cc_rate(flux, cc, u, masses, right_mu, right_mu, energy, width, profile, n_states)
                + cc_rate(flux, cc, u, masses, wrong_mu, wrong_mu, energy, width, profile, n_states)
            ),
            "beam": cc_rate(flux, cc, u, masses, right_e, right_e, energy, width, profile, n_states)
            + cc_rate(flux, cc, u, masses, wrong_e, wrong_e, energy, width, profile, n_states),
            "signal": cc_rate(flux, cc, u, masses, right_mu, right_e, energy, width, profile, n_states)
            + cc_rate(flux, cc, u, masses, wrong_mu, wrong_e, energy, width, profile, n_states),
        }
    return {
        "nc": 0.015 * nc_rate_all(flux, nc, u, masses, energy, width, profile, n_states),
        "wrong_mu": cc_rate(flux, cc, u, masses, wrong_mu, wrong_mu, energy, width, profile, n_states),
        "signal": cc_rate(flux, cc, u, masses, right_mu, right_mu, energy, width, profile, n_states),
    }


def compute_scan(df: pd.DataFrame, e_min: float, e_max: float, n_bins: int) -> pd.DataFrame:
    u = build_mixing(df)
    masses5 = np.column_stack(
        [
            np.zeros(len(df), dtype=float),
            df["dm21_calc_eV2"].to_numpy(dtype=float),
            df["dm31_calc_eV2"].to_numpy(dtype=float),
            df["dm41_calc_eV2"].to_numpy(dtype=float),
            df["dm51_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses5[:, :3]
    width = (e_max - e_min) / n_bins
    energies = e_min + (np.arange(n_bins, dtype=float) + 0.5) * width
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
    metrics = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}
    prob_amps = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}

    for panel in PANELS:
        mode = "RHC" if panel.startswith("RHC") else "FHC"
        flavor = "numubar" if panel.startswith("RHC") else "numu"
        beta = 0 if panel.endswith("_app") else 1
        for energy in energies:
            comp3 = panel_components(panel, fluxes[mode], cc, nc, u, masses3, energy, width, profiles[mode], 3)
            comp5 = panel_components(panel, fluxes[mode], cc, nc, u, masses5, energy, width, profiles[mode], 5)
            n3 = sum(comp3.values()) * scale
            n5 = sum(comp5.values()) * scale
            rel = np.divide(n5 - n3, n3, out=np.zeros_like(n5), where=np.abs(n3) > 1.0e-30)
            metrics[panel] = np.maximum(metrics[panel], np.abs(rel))

            p3 = probability_source_average(u, masses3, 1, beta, flavor, energy, profiles[mode], 3)
            p5 = probability_source_average(u, masses5, 1, beta, flavor, energy, profiles[mode], 5)
            prob_amps[panel] = np.maximum(prob_amps[panel], np.abs(p5 - p3))

    out_cols = [
        "point_id",
        "dm41_calc_eV2",
        "dm51_calc_eV2",
        "theta14_deg",
        "theta24_deg",
        "theta34_deg",
        "theta15_deg",
        "theta25_deg",
        "theta35_deg",
        "eta11_abs",
        "eta22_abs",
        "eta33_abs",
    ]
    out = df[out_cols].copy()
    out["dm45_calc_eV2"] = np.abs(out["dm51_calc_eV2"] - out["dm41_calc_eV2"])
    out["Ue4_abs"] = np.abs(u[:, 0, 3])
    out["Umu4_abs"] = np.abs(u[:, 1, 3])
    out["Ue5_abs"] = np.abs(u[:, 0, 4])
    out["Umu5_abs"] = np.abs(u[:, 1, 4])
    out["appearance_mixing_sum"] = out["Ue4_abs"] * out["Umu4_abs"] + out["Ue5_abs"] * out["Umu5_abs"]
    out["disappearance_mixing_sum"] = out["Umu4_abs"] ** 2 + out["Umu5_abs"] ** 2
    out["appearance_amplitude_sum"] = 4.0 * (
        out["Ue4_abs"] ** 2 * out["Umu4_abs"] ** 2 + out["Ue5_abs"] ** 2 * out["Umu5_abs"] ** 2
    )
    umu_sum = out["disappearance_mixing_sum"]
    out["disappearance_amplitude_sum"] = 4.0 * umu_sum * (1.0 - umu_sum)
    for panel in PANELS:
        out[f"{panel}_max_abs_rel"] = metrics[panel]
        out[f"{panel}_max_abs_percent"] = 100.0 * metrics[panel]
        out[f"{panel}_probability_amplitude"] = prob_amps[panel]
    out["global_max_abs_rel"] = np.maximum.reduce([metrics[p] for p in PANELS])
    out["global_max_abs_percent"] = 100.0 * out["global_max_abs_rel"]
    return out


def ensure_derived_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "dm45_calc_eV2" not in out.columns:
        out["dm45_calc_eV2"] = np.abs(out["dm51_calc_eV2"] - out["dm41_calc_eV2"])
    return out


def worker(payload):
    return compute_scan(*payload)


def split_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def color_norm(summary: pd.DataFrame):
    values = summary[[f"{panel}_max_abs_percent" for panel in PANELS]].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0e-12
    return Normalize(vmin=vmin, vmax=vmax)


def draw_scan_map(summary: pd.DataFrame, out_path: Path, x_col: str, y_col: str, x_label: str, y_label: str, title: str, x_log: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=True, sharey=False)
    norm = color_norm(summary)
    image = None
    for ax, panel in zip(axes.flat, PANELS):
        image = ax.scatter(
            summary[x_col],
            summary[y_col],
            c=summary[f"{panel}_max_abs_percent"],
            s=16,
            alpha=0.78,
            cmap="viridis",
            norm=norm,
            edgecolors="none",
        )
        ax.set_title(LABELS[panel], fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        if x_log:
            ax.set_xscale("log")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(r"$\max_E |N_{5\nu}/N_{active3\nu}-1|$ [%]")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_mixing_map(summary: pd.DataFrame, out_path: Path, x_log: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), sharex=True)
    norm = color_norm(summary)
    image = None
    for ax, panel in zip(axes.flat, PANELS):
        if panel.endswith("_app"):
            y_col = "appearance_mixing_sum"
            y_label = r"$|U_{e4}U_{\mu4}|+|U_{e5}U_{\mu5}|$"
        else:
            y_col = "disappearance_mixing_sum"
            y_label = r"$|U_{\mu4}|^2+|U_{\mu5}|^2$"
        image = ax.scatter(summary["dm41_calc_eV2"], summary[y_col], c=summary[f"{panel}_max_abs_percent"], s=16, alpha=0.78, cmap="viridis", norm=norm, edgecolors="none")
        ax.set_title(LABELS[panel], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        if x_log:
            ax.set_xscale("log")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(r"$\max_E |N_{5\nu}/N_{active3\nu}-1|$ [%]")
    fig.suptitle(f"DUNE ND 6.5 yr/mode - ISS(2,4) mixing-observable maps ({len(summary)} points)", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_scaling(summary: pd.DataFrame, out_path: Path, x_cols: dict[str, tuple[str, str]], title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8))
    for ax, panel in zip(axes.flat, PANELS):
        x_col, x_label = x_cols[panel]
        x = summary[x_col].to_numpy(dtype=float)
        y = summary[f"{panel}_max_abs_percent"].to_numpy(dtype=float)
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
            xs = np.geomspace(float(np.nanmin(x[valid])), float(np.nanmax(x[valid])), 150)
            ax.plot(xs, 10.0 ** (intercept + slope * np.log10(xs)), color="0.35", lw=1.1, ls=":", label=rf"slope = {slope:.2f}")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_points(max_points: int, points_csv: Path = POINTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    for col in ["pmns_pass", "eta_pass"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    selected = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    required = ["dm21_calc_eV2", "dm31_calc_eV2", "dm41_calc_eV2", "dm51_calc_eV2", "U5_solver1_re", "U5_solver25_re"]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    selected = selected[(selected["dm41_calc_eV2"] > 0.0) & (selected["dm51_calc_eV2"] > 0.0)].copy()
    if max_points > 0:
        selected = selected.head(max_points).copy()
    if selected.empty:
        raise RuntimeError("Aucun point ISS(2,4) pmns_pass=1 et eta_pass=1 exploitable.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DUNE ND ISS(2,4) scan maps.")
    parser.add_argument("--points-csv", type=Path, default=POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--plots-only", action="store_true", help="Reuse the existing scan CSV and only redraw maps.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--e-min-GeV", type=float, default=0.5)
    parser.add_argument("--e-max-GeV", type=float, default=8.0)
    parser.add_argument("--e-bins", type=int, default=30)
    parser.add_argument("--x-log", action="store_true", help="Use logarithmic x-axes for scan-map figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plots_only:
        if not args.out_csv.exists():
            raise FileNotFoundError(f"CSV de scan absent: {args.out_csv}")
        summary = pd.read_csv(args.out_csv)
        summary = ensure_derived_columns(summary)
        summary.to_csv(args.out_csv, index=False)
    else:
        points = load_points(args.max_points, args.points_csv)
        if args.workers > 1 and len(points) > args.chunk_size:
            chunks = split_dataframe(points, args.chunk_size)
            payloads = [(chunk, args.e_min_GeV, args.e_max_GeV, args.e_bins) for chunk in chunks]
            print(f"Calcul parallele ISS(2,4): {len(chunks)} blocs, {args.workers} workers, {len(points)} points")
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                summary = pd.concat(pool.map(worker, payloads), ignore_index=True)
        else:
            summary = compute_scan(points, args.e_min_GeV, args.e_max_GeV, args.e_bins)
        summary = ensure_derived_columns(summary)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False)

    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta24.png",
        "dm41_calc_eV2",
        "theta24_deg",
        r"$\Delta m^2_{41}$ [eV$^2$]",
        r"$\theta_{24}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta24_dm51.png",
        "dm51_calc_eV2",
        "theta24_deg",
        r"$\Delta m^2_{51}$ [eV$^2$]",
        r"$\theta_{24}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta25_dm41.png",
        "dm41_calc_eV2",
        "theta25_deg",
        r"$\Delta m^2_{41}$ [eV$^2$]",
        r"$\theta_{25}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta25_dm51.png",
        "dm51_calc_eV2",
        "theta25_deg",
        r"$\Delta m^2_{51}$ [eV$^2$]",
        r"$\theta_{25}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta24_dm45.png",
        "dm45_calc_eV2",
        "theta24_deg",
        r"$|\Delta m^2_{54}|$ [eV$^2$]",
        r"$\theta_{24}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_scan_map(
        summary,
        args.out_dir / "iss24_scan_map_theta25_dm45.png",
        "dm45_calc_eV2",
        "theta25_deg",
        r"$|\Delta m^2_{54}|$ [eV$^2$]",
        r"$\theta_{25}$ [deg]",
        f"DUNE ND 6.5 yr/mode - ISS(2,4) map ({len(summary)} points)",
        x_log=args.x_log,
    )
    draw_mixing_map(summary, args.out_dir / "iss24_scan_map_mixing_observables.png", x_log=args.x_log)
    draw_scaling(
        summary,
        args.out_dir / "iss24_max_event_ratio_vs_expected_amplitude.png",
        {
            "FHC_app": ("appearance_amplitude_sum", r"$4(|U_{e4}|^2|U_{\mu4}|^2+|U_{e5}|^2|U_{\mu5}|^2)$"),
            "RHC_app": ("appearance_amplitude_sum", r"$4(|U_{e4}|^2|U_{\mu4}|^2+|U_{e5}|^2|U_{\mu5}|^2)$"),
            "FHC_dis": ("disappearance_amplitude_sum", r"$4S_\mu(1-S_\mu)$"),
            "RHC_dis": ("disappearance_amplitude_sum", r"$4S_\mu(1-S_\mu)$"),
        },
        "DUNE ND 6.5 yr/mode - ISS(2,4)",
    )
    draw_scaling(
        summary,
        args.out_dir / "iss24_max_event_ratio_vs_dk2nu_probability_amplitude.png",
        {panel: (f"{panel}_probability_amplitude", r"$\max_E |\Delta \bar{P}_{\alpha\beta}^{dk2nu}(E)|$") for panel in PANELS},
        "DUNE ND 6.5 yr/mode - ISS(2,4)",
    )

    print(f"Points utilises: {len(summary)}")
    print(f"CSV sauvegarde: {args.out_csv}")
    for panel in PANELS:
        print(f"{panel}: max={summary[f'{panel}_max_abs_percent'].max():.6g}% median={summary[f'{panel}_max_abs_percent'].median():.6g}%")


if __name__ == "__main__":
    main()
