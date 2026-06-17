import argparse
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
from matplotlib.colors import LogNorm


POINTS_CSV = Path("data/inverse_seesaw/3p1/inverse_construct_23_kept_points/inverse_construct_23_kept_points.csv")
GLOBES = Path("data/dune/2103.04797v2/dune_globes")
FLUX_DIR = Path("data/dune/flux")
SOURCE_PROFILE_FHC = Path("data/dune/dk2nu/source_profile_z_FHC_ND.csv")
SOURCE_PROFILE_RHC = Path("data/dune/dk2nu/source_profile_z_RHC_ND.csv")
OUT = Path("figures/dune_nd/scan_maps/iss23_active3nu_max_event_ratio_map.png")
OUT_CSV = Path("data/dune_nd/scan_maps/iss23_active3nu_max_event_ratio_map.csv")

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
ANTI_FLAVOR = {"nue": False, "numu": False, "nutau": False, "nuebar": True, "numubar": True, "nutaubar": True}
PANELS = ("FHC_app", "RHC_app", "FHC_dis", "RHC_dis")


def read_flux(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["E_GeV", *FLAVORS],
        decimal=",",
        comment="#",
        engine="python",
    )
    return df.apply(pd.to_numeric, errors="coerce").dropna(subset=["E_GeV"])


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


def interp(df: pd.DataFrame, column: str, energies: np.ndarray) -> np.ndarray:
    return np.interp(energies, df["E_GeV"].to_numpy(dtype=float), df[column].to_numpy(dtype=float), left=0.0, right=0.0)


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
    u = np.zeros((len(df), 4, 4), dtype=float)
    for r in range(4):
        for c in range(4):
            u[:, r, c] = df[f"U_solver_{r + 1}{c + 1}"].to_numpy(dtype=float)
    return u


def probability_at_baseline(u: np.ndarray, masses: np.ndarray, alpha: int, beta: int, energy: float, baseline_km: float, n_states: int) -> np.ndarray:
    p = np.ones(u.shape[0], dtype=float) if alpha == beta else np.zeros(u.shape[0], dtype=float)
    for i in range(n_states):
        mi2 = masses[:, i]
        for j in range(i + 1, n_states):
            mj2 = masses[:, j]
            phase = PHASE_COEFF * (mj2 - mi2) * baseline_km / energy
            a = (
                u[:, alpha, i]
                * u[:, beta, i]
                * u[:, alpha, j]
                * u[:, beta, j]
            )
            p -= 4.0 * np.real(a) * np.sin(phase) ** 2
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
    z_values, weights = source_weights(profile, flavor, energy)
    out = np.zeros(u.shape[0], dtype=float)
    for z_m, weight in zip(z_values, weights):
        baseline_km = max(0.0, BASELINE_KM - z_m * 1.0e-3)
        if beta is None:
            p = np.zeros(u.shape[0], dtype=float)
            for active_beta in range(3):
                p += probability_at_baseline(u, masses, alpha, active_beta, energy, baseline_km, n_states)
            p = np.clip(p, 0.0, 1.0)
        else:
            p = probability_at_baseline(u, masses, alpha, beta, energy, baseline_km, n_states)
        out += weight * p
    return out


def xsec_flavor(final_flavor: str, interaction: str) -> str:
    if interaction == "nc":
        return final_flavor
    return final_flavor


def cc_rate(
    flux: pd.DataFrame,
    xsec: pd.DataFrame,
    u: np.ndarray,
    masses: np.ndarray,
    initial: str,
    final: str,
    energy: float,
    width: float,
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
    n_states: int,
) -> np.ndarray:
    alpha = FLAVOR_ALPHA[initial]
    beta = FLAVOR_ALPHA[final]
    flux_value = float(interp(flux, initial, np.asarray([energy]))[0]) / M2_TO_CM2
    xsec_value = float(interp(xsec, final, np.asarray([energy]))[0]) * energy * XSEC_SCALE_CM2
    prob = probability_source_average(u, masses, alpha, beta, initial, energy, profile, n_states)
    return flux_value * xsec_value * prob * width


def nc_rate_all(
    flux: pd.DataFrame,
    xsec: pd.DataFrame,
    u: np.ndarray,
    masses: np.ndarray,
    energy: float,
    width: float,
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
    n_states: int,
) -> np.ndarray:
    total = np.zeros(u.shape[0], dtype=float)
    for flavor in ACTIVE_FLAVORS:
        alpha = FLAVOR_ALPHA[flavor]
        flux_value = float(interp(flux, flavor, np.asarray([energy]))[0]) / M2_TO_CM2
        xsec_value = float(interp(xsec, flavor, np.asarray([energy]))[0]) * energy * XSEC_SCALE_CM2
        prob = probability_source_average(u, masses, alpha, None, flavor, energy, profile, n_states)
        total += flux_value * xsec_value * prob * width
    return total


def panel_components(
    panel: str,
    flux: pd.DataFrame,
    cc: pd.DataFrame,
    nc: pd.DataFrame,
    u: np.ndarray,
    masses: np.ndarray,
    energy: float,
    width: float,
    profile: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
    n_states: int,
) -> dict[str, np.ndarray]:
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
        "tau": np.zeros(u.shape[0], dtype=float),
        "signal": cc_rate(flux, cc, u, masses, right_mu, right_mu, energy, width, profile, n_states),
    }


def compute_scan(df: pd.DataFrame, e_min: float, e_max: float, n_bins: int) -> pd.DataFrame:
    u = build_mixing(df)
    masses4 = np.column_stack(
        [
            np.zeros(len(df), dtype=float),
            df["dm21_calc_eV2"].to_numpy(dtype=float),
            df["dm31_calc_eV2"].to_numpy(dtype=float),
            df["dm41_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses4[:, :3]
    energies = e_min + (np.arange(n_bins, dtype=float) + 0.5) * ((e_max - e_min) / n_bins)
    width = (e_max - e_min) / n_bins

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

    scale = POT_PER_YEAR * EXPOSURE_YEARS * TARGET_MASS_KT * 1.0e9 * AVOGADRO
    metrics: dict[str, np.ndarray] = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}

    for panel in PANELS:
        mode = "RHC" if panel.startswith("RHC") else "FHC"
        for energy in energies:
            comp3 = panel_components(panel, fluxes[mode], cc, nc, u, masses3, energy, width, profiles[mode], 3)
            comp4 = panel_components(panel, fluxes[mode], cc, nc, u, masses4, energy, width, profiles[mode], 4)
            n3 = sum(comp3.values()) * scale
            n4 = sum(comp4.values()) * scale
            rel = np.divide(n4 - n3, n3, out=np.zeros_like(n4), where=np.abs(n3) > 1.0e-30)
            metrics[panel] = np.maximum(metrics[panel], np.abs(rel))

    out = df[
        [
            "point_id",
            "dm41_calc_eV2",
            "dm41_target_eV2",
            "zeta_norm",
            "theta14_deg",
            "theta24_deg",
            "theta34_deg",
            "eta11_abs",
            "eta22_abs",
            "eta33_abs",
            "mu00_eV",
        ]
    ].copy()
    for panel in PANELS:
        out[f"{panel}_max_abs_rel"] = metrics[panel]
        out[f"{panel}_max_abs_percent"] = 100.0 * metrics[panel]
    out["global_max_abs_rel"] = np.maximum.reduce([metrics[panel] for panel in PANELS])
    out["global_max_abs_percent"] = 100.0 * out["global_max_abs_rel"]
    return out


def plot_map(summary: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8), sharex=True, sharey=True)
    values = summary[[f"{panel}_max_abs_percent" for panel in PANELS]].to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    norm = LogNorm(vmin=max(float(np.nanmin(positive)), 1.0e-5), vmax=max(float(np.nanmax(positive)), 1.0e-4)) if positive.size else None
    labels = {
        "FHC_app": r"FHC $\nu_e$ appearance",
        "RHC_app": r"RHC $\bar{\nu}_e$ appearance",
        "FHC_dis": r"FHC $\nu_\mu$ disappearance",
        "RHC_dis": r"RHC $\bar{\nu}_\mu$ disappearance",
    }

    image = None
    for ax, panel in zip(axes.flat, PANELS):
        image = ax.scatter(
            summary["dm41_calc_eV2"],
            summary["zeta_norm"],
            c=summary[f"{panel}_max_abs_percent"],
            s=16,
            alpha=0.78,
            cmap="viridis",
            norm=norm,
            edgecolors="none",
        )
        ax.set_title(labels[panel], fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]")
        ax.set_ylabel(r"$\|\zeta\|$")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(r"$\max_E |N_{4\nu}/N_{active3\nu}-1|$ [%]")
    fig.suptitle(
        rf"DUNE ND 6.5 yr/mode, dk2nu source average - {len(summary)} points with pmns_pass=1 and eta_pass=1",
        fontsize=13,
        fontweight="bold",
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map max ND event-ratio deviation for ISS(2,3) scan points.")
    parser.add_argument("--points-csv", type=Path, default=POINTS_CSV)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--e-min-GeV", type=float, default=0.5)
    parser.add_argument("--e-max-GeV", type=float, default=8.0)
    parser.add_argument("--e-bins", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=0, help="Optional debug limit after filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.points_csv)
    for column in ["pmns_pass", "eta_pass"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    selected = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    selected = selected.replace([np.inf, -np.inf], np.nan)
    required = ["dm41_calc_eV2", "zeta_norm", "dm21_calc_eV2", "dm31_calc_eV2", "U_solver_11", "U_solver_44"]
    selected = selected.dropna(subset=required)
    selected = selected[(selected["dm41_calc_eV2"] > 0.0) & (selected["zeta_norm"] > 0.0)].copy()
    if args.max_points > 0:
        selected = selected.head(args.max_points).copy()
    if selected.empty:
        raise RuntimeError("Aucun point pmns_pass=1 et eta_pass=1 exploitable dans le CSV.")

    summary = compute_scan(selected, args.e_min_GeV, args.e_max_GeV, args.e_bins)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    plot_map(summary, args.out)
    print(f"Points utilises: {len(summary)}")
    print(f"CSV sauvegarde: {args.out_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")
    for panel in PANELS:
        col = f"{panel}_max_abs_percent"
        print(f"{panel}: max={summary[col].max():.6g}% median={summary[col].median():.6g}%")


if __name__ == "__main__":
    main()
