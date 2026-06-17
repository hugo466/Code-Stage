import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(".")
GLOBES = Path("data/dune/2103.04797v2/dune_globes")
FLUX_DIR = Path("data/dune/flux")
OUT = Path("figures/dune_nd/reference/fig4_reconstructed_energy_nd_globes.png")
OUT_CSV = Path("data/dune/validation/fig4_nd_reference_spectra.csv")
OUT_NO_SMEARING = Path("figures/dune_nd/reference/fig4_reconstructed_energy_nd_globes_no_smearing.png")
OUT_NO_SMEARING_CSV = Path("data/dune/validation/fig4_nd_reference_spectra_no_smearing.csv")
OUT_IDEAL_RESPONSE = Path("figures/dune_nd/reference/fig4_reconstructed_energy_nd_globes_no_smearing_no_efficiency.png")
OUT_IDEAL_RESPONSE_CSV = Path("data/dune/validation/fig4_nd_reference_spectra_no_smearing_no_efficiency.csv")

AVOGADRO = 6.02214076e23
M2_TO_CM2 = 1.0e4
XSEC_SCALE_CM2 = 1.0e-38
BASELINE_KM = 0.574
DENSITY_G_CM3 = 2.848
YE = 0.5
MATTER_A_COEFF = 1.52e-4
POT_PER_YEAR = 11.0e20
NUTIME_YEARS = 6.5
NUBARTIME_YEARS = 6.5
TARGET_MASS_KT = 0.067
TRUE_ENERGY_MAX_GEV = 20.0
GLOBES_EVENT_NORM = 1.017718
E_MIN_GEV = 0.5
E_MAX_GEV = 8.0

FHC_FLUX = FLUX_DIR / "flux_dune_neutrino_ND_globes.txt"
RHC_FLUX = FLUX_DIR / "flux_dune_antineutrino_ND_globes.txt"
DK2NU_FHC = Path("data/dune/dk2nu/flux_z_FHC_ND_raw.csv")
DK2NU_RHC = Path("data/dune/dk2nu/flux_z_RHC_ND_raw.csv")

THETA12_RAD = 0.5903
THETA23_RAD = 0.866
THETA13_RAD = 0.150
DELTA_CP_RAD = 0.0
DM21_EV2 = 7.39e-5
DM32_EV2 = 2.451e-3

FLAVORS = ["nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
DK2NU_COLORS = {
    "blue": "#4c6fff",
    "limegreen": "#239f4b",
    "red": "#ff5a5f",
}


@dataclass(frozen=True)
class Channel:
    flux_mode: str
    initial: str
    final: str
    interaction: str
    smear: str
    eff: str
    anti: bool = False
    no_osc: bool = False
    scale: float = 1.0


def read_flux(path):
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


def read_dk2nu_flux(path):
    table = pd.read_csv(path)
    table["E_GeV"] = 0.5 * (table["E_GeV_bin_low"] + table["E_GeV_bin_high"])
    flux = (
        table.groupby(["E_GeV", "flavor"], as_index=False)["weight"]
        .sum()
        .pivot(index="E_GeV", columns="flavor", values="weight")
        .reset_index()
    )
    for flavor in FLAVORS:
        if flavor not in flux:
            flux[flavor] = 0.0
    return flux[["E_GeV", *FLAVORS]]


def read_xsec(path):
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean:
            tokens.extend(clean.split())
    values = [float(token.replace(",", ".")) for token in tokens]
    rows = np.asarray(values, dtype=float).reshape((-1, 7))
    df = pd.DataFrame(rows, columns=["log10_E_GeV", *FLAVORS])
    df.insert(0, "E_GeV", np.power(10.0, df.pop("log10_E_GeV")))
    return df


def read_post_eff(path):
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"\{([^{}]+)\}", text, flags=re.S)
    if not rows:
        raise ValueError(f"No efficiency vector found in {path}")
    return np.asarray([float(x) for x in split_globes_list(rows[0])], dtype=float)


def split_globes_list(text):
    return [x.strip() for x in text.replace("\n", " ").split(",") if x.strip()]


def parse_glb_vector(path, name):
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\${name}\s*=\s*\{{([^}}]+)\}}", text, flags=re.S)
    if not match:
        raise ValueError(f"{name} not found in {path}")
    return np.asarray([float(x.strip()) for x in match.group(1).split(",") if x.strip()], dtype=float)


def bin_centers(widths):
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    return 0.5 * (edges[:-1] + edges[1:]), edges


def read_smearing(path, n_true):
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"\{([^{}]+)\}", text, flags=re.S)
    matrix = np.zeros((len(rows), n_true), dtype=float)
    for reco_index, row in enumerate(rows):
        values = [float(x) for x in split_globes_list(row)]
        if len(values) < 3:
            continue
        lo = max(0, int(round(values[0])))
        hi = min(n_true - 1, int(round(values[1])))
        count = max(0, hi - lo + 1)
        matrix[reco_index, lo : hi + 1] = values[2 : 2 + count]
    return matrix


def identity_response(true_edges, rec_edges):
    response = np.zeros((len(rec_edges) - 1, len(true_edges) - 1), dtype=float)
    for true_index, (true_low, true_high) in enumerate(zip(true_edges[:-1], true_edges[1:])):
        true_width = true_high - true_low
        if true_width <= 0.0:
            continue
        overlap_low = np.maximum(rec_edges[:-1], true_low)
        overlap_high = np.minimum(rec_edges[1:], true_high)
        response[:, true_index] = np.maximum(0.0, overlap_high - overlap_low) / true_width
    return response


def nd_effective_response(smear_name, true_edges, rec_edges):
    return read_smearing(GLOBES / "smr" / f"{smear_name}.txt", len(true_edges) - 1)


def interp(df, column, energies):
    return np.interp(energies, df["E_GeV"].to_numpy(), df[column].to_numpy(), left=0.0, right=0.0)


def pmns_matrix(delta_cp=0.0):
    s12, c12 = np.sin(THETA12_RAD), np.cos(THETA12_RAD)
    s13, c13 = np.sin(THETA13_RAD), np.cos(THETA13_RAD)
    s23, c23 = np.sin(THETA23_RAD), np.cos(THETA23_RAD)
    d = delta_cp
    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * d)],
            [
                -s12 * c23 - c12 * s23 * s13 * np.exp(1j * d),
                c12 * c23 - s12 * s23 * s13 * np.exp(1j * d),
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * np.exp(1j * d),
                -c12 * s23 - s12 * c23 * s13 * np.exp(1j * d),
                c23 * c13,
            ],
        ],
        dtype=complex,
    )


def prob(alpha, beta, energy_GeV, anti=False):
    if energy_GeV <= 0.0:
        return 0.0
    masses = np.array([0.0, DM21_EV2, DM21_EV2 + DM32_EV2], dtype=float)
    u = pmns_matrix(delta_cp=DELTA_CP_RAD)
    if anti:
        u = np.conjugate(u)
    h = u @ np.diag(masses) @ np.conjugate(u).T
    a = MATTER_A_COEFF * YE * DENSITY_G_CM3 * energy_GeV
    h[0, 0] += -a if anti else a
    eigvals, eigvecs = np.linalg.eigh(h)
    phases = np.exp(-1j * 2.533865 * eigvals * BASELINE_KM / energy_GeV)
    amp = eigvecs @ np.diag(phases) @ np.conjugate(eigvecs).T
    return float(np.clip(abs(amp[beta, alpha]) ** 2, 0.0, 1.0))


def probability_vector(initial, final, energies, anti=False, no_osc=False):
    if no_osc:
        return np.ones_like(energies, dtype=float)
    flavor_index = {"e": 0, "m": 1, "t": 2}
    return np.asarray([prob(flavor_index[initial], flavor_index[final], e, anti=anti) for e in energies], dtype=float)


def final_xsec_flavor(final, anti):
    if final == "e":
        return "nuebar" if anti else "nue"
    if final == "m":
        return "numubar" if anti else "numu"
    if final == "t":
        return "nutaubar" if anti else "nutau"
    raise ValueError(final)


def initial_flux_flavor(initial, anti):
    if initial == "e":
        return "nuebar" if anti else "nue"
    if initial == "m":
        return "numubar" if anti else "numu"
    if initial == "t":
        return "nutaubar" if anti else "nutau"
    raise ValueError(initial)


CHANNELS = {
    "FHC_app": {
        "signal": [
            Channel("FHC", "m", "e", "cc", "app_nue_sig", "post_app_FHC_nue_sig"),
            Channel("FHC", "m", "e", "cc", "app_nuebar_sig", "post_app_FHC_nuebar_sig", anti=True),
        ],
        "beam": [
            Channel("FHC", "e", "e", "cc", "app_nue_bkg", "post_app_FHC_nue_bkg", no_osc=True),
            Channel("FHC", "e", "e", "cc", "app_nuebar_bkg", "post_app_FHC_nuebar_bkg", anti=True, no_osc=True),
        ],
        "numu": [
            Channel("FHC", "m", "m", "cc", "app_numu_bkg", "post_app_FHC_numu_bkg", no_osc=True, scale=0.015),
            Channel("FHC", "m", "m", "cc", "app_numubar_bkg", "post_app_FHC_numubar_bkg", anti=True, no_osc=True, scale=0.015),
        ],
        "nc": [
            Channel("FHC", "m", "m", "nc", "app_NC_bkg", "post_app_FHC_NC_bkg", no_osc=True, scale=0.025),
            Channel("FHC", "m", "m", "nc", "app_NC_bkg", "post_app_FHC_aNC_bkg", anti=True, no_osc=True, scale=0.025),
        ],
    },
    "RHC_app": {
        "signal": [
            Channel("RHC", "m", "e", "cc", "app_nue_sig", "post_app_RHC_nue_sig"),
            Channel("RHC", "m", "e", "cc", "app_nuebar_sig", "post_app_RHC_nuebar_sig", anti=True),
        ],
        "beam": [
            Channel("RHC", "e", "e", "cc", "app_nue_bkg", "post_app_RHC_nue_bkg", no_osc=True),
            Channel("RHC", "e", "e", "cc", "app_nuebar_bkg", "post_app_RHC_nuebar_bkg", anti=True, no_osc=True),
        ],
        "numu": [
            Channel("RHC", "m", "m", "cc", "app_numu_bkg", "post_app_RHC_numu_bkg", no_osc=True, scale=0.015),
            Channel("RHC", "m", "m", "cc", "app_numubar_bkg", "post_app_RHC_numubar_bkg", anti=True, no_osc=True, scale=0.015),
        ],
        "nc": [
            Channel("RHC", "m", "m", "nc", "app_NC_bkg", "post_app_RHC_NC_bkg", no_osc=True, scale=0.025),
            Channel("RHC", "m", "m", "nc", "app_aNC_bkg", "post_app_RHC_aNC_bkg", anti=True, no_osc=True, scale=0.025),
        ],
    },
    "FHC_dis": {
        "signal": [Channel("FHC", "m", "m", "cc", "dis_numu_sig", "post_dis_FHC_numu_sig")],
        "wrong_mu": [Channel("FHC", "m", "m", "cc", "dis_numubar_sig", "post_dis_FHC_numubar_sig", anti=True)],
        "tau": [
            Channel("FHC", "m", "t", "cc", "dis_nutau_bkg", "post_dis_FHC_nutau_bkg", scale=0.0),
            Channel("FHC", "m", "t", "cc", "dis_nutaubar_bkg", "post_dis_FHC_nutaubar_bkg", anti=True, scale=0.0),
        ],
        "nc": [
            Channel("FHC", "m", "m", "nc", "dis_NC_bkg", "post_dis_FHC_NC_bkg", no_osc=True, scale=0.015),
            Channel("FHC", "m", "m", "nc", "dis_aNC_bkg", "post_dis_FHC_NC_bkg", anti=True, no_osc=True, scale=0.015),
        ],
    },
    "RHC_dis": {
        "signal": [Channel("RHC", "m", "m", "cc", "dis_numubar_sig", "post_dis_RHC_numubar_sig", anti=True)],
        "wrong_mu": [Channel("RHC", "m", "m", "cc", "dis_numu_sig", "post_dis_RHC_numu_sig")],
        "tau": [
            Channel("RHC", "m", "t", "cc", "dis_nutau_bkg", "post_dis_RHC_nutau_bkg", scale=0.0),
            Channel("RHC", "m", "t", "cc", "dis_nutaubar_bkg", "post_dis_RHC_nutaubar_bkg", anti=True, scale=0.0),
        ],
        "nc": [
            Channel("RHC", "m", "m", "nc", "dis_NC_bkg", "post_dis_RHC_NC_bkg", no_osc=True, scale=0.015),
            Channel("RHC", "m", "m", "nc", "dis_aNC_bkg", "post_dis_RHC_NC_bkg", anti=True, no_osc=True, scale=0.015),
        ],
    },
}


def compute_channel(channel, datasets, sampling_centers, sampling_widths, sampling_edges, rec_edges, use_smearing, use_efficiency):
    flux = datasets["flux"][channel.flux_mode]
    xsec = datasets["xsec"][channel.interaction]
    flux_col = initial_flux_flavor(channel.initial, channel.anti)
    xsec_col = final_xsec_flavor(channel.final, channel.anti) if channel.interaction == "cc" else flux_col
    flux_values = interp(flux, flux_col, sampling_centers) / M2_TO_CM2
    xsec_values = interp(xsec, xsec_col, sampling_centers) * sampling_centers * XSEC_SCALE_CM2
    probs = probability_vector(channel.initial, channel.final, sampling_centers, anti=channel.anti, no_osc=channel.no_osc)
    pot = POT_PER_YEAR * (NUBARTIME_YEARS if channel.flux_mode == "RHC" else NUTIME_YEARS)
    targets = TARGET_MASS_KT * 1.0e9 * AVOGADRO
    true_counts = channel.scale * GLOBES_EVENT_NORM * flux_values * xsec_values * probs * sampling_widths * pot * targets
    true_counts = np.where(sampling_centers <= TRUE_ENERGY_MAX_GEV, true_counts, 0.0)
    response = (
        nd_effective_response(channel.smear, sampling_edges, rec_edges)
        if use_smearing
        else identity_response(sampling_edges, rec_edges)
    )
    reco = response @ true_counts
    if not use_efficiency:
        return reco
    eff = read_post_eff(GLOBES / "eff" / f"{channel.eff}.txt")
    return reco * eff


def aggregate_025(values, rec_edges):
    mask = rec_edges[:-1] < 8.0
    values = values[mask]
    centers_0125 = 0.5 * (rec_edges[:-1] + rec_edges[1:])[mask]
    n = min(len(values), 64)
    values = values[:n]
    centers_0125 = centers_0125[:n]
    if n % 2:
        values = values[:-1]
        centers_0125 = centers_0125[:-1]
    return centers_0125.reshape(-1, 2).mean(axis=1), values.reshape(-1, 2).sum(axis=1)


def compute_spectra(out_csv=None, flux_source="globes", use_smearing=True, use_efficiency=True):
    binsize = parse_glb_vector(GLOBES / "DUNE_GLoBES.glb", "binsize")
    sampling = parse_glb_vector(GLOBES / "DUNE_GLoBES.glb", "sampling_stepsize")
    _, rec_edges = bin_centers(binsize)
    sampling_centers, sampling_edges = bin_centers(sampling)
    if flux_source == "globes":
        fhc_flux = read_flux(FHC_FLUX)
        rhc_flux = read_flux(RHC_FLUX)
    elif flux_source == "dk2nu":
        fhc_flux = read_dk2nu_flux(DK2NU_FHC)
        rhc_flux = read_dk2nu_flux(DK2NU_RHC)
    else:
        raise ValueError(f"Unknown flux source: {flux_source}")

    datasets = {
        "flux": {"FHC": fhc_flux, "RHC": rhc_flux},
        "xsec": {
            "cc": read_xsec(GLOBES / "xsec" / "xsec_cc.dat"),
            "nc": read_xsec(GLOBES / "xsec" / "xsec_nc.dat"),
        },
    }

    spectra = {}
    rows = []
    for panel, groups in CHANNELS.items():
        spectra[panel] = {}
        for group, channels in groups.items():
            reco = np.zeros(len(rec_edges) - 1, dtype=float)
            for channel in channels:
                reco += compute_channel(
                    channel,
                    datasets,
                    sampling_centers,
                    sampling,
                    sampling_edges,
                    rec_edges,
                    use_smearing,
                    use_efficiency,
                )
            e025, values = aggregate_025(reco, rec_edges)
            spectra[panel][group] = values
            for e, y in zip(e025, values):
                rows.append({"panel": panel, "component": group, "Erec_GeV": e, "events_per_0p25_GeV": y})
        spectra[panel]["Erec_GeV"] = e025
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return spectra


def step(ax, x, y, **kwargs):
    ax.step(x, y, where="mid", **kwargs)


def draw_panel(ax, spectra, panel, title, app=True, anti=False, dk2nu_spectra=None, no_smearing=False, no_efficiency=False):
    x = spectra[panel]["Erec_GeV"]
    if app:
        nc = spectra[panel]["nc"]
        numu = nc + spectra[panel]["numu"]
        beam = numu + spectra[panel]["beam"]
        total = beam + spectra[panel]["signal"]
        labels = (
            r"Signal $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Signal $(\nu_e+\bar{\nu}_e)$ CC",
            r"Beam $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Beam $(\nu_e+\bar{\nu}_e)$ CC",
            r"$(\bar{\nu}_\mu+\nu_\mu)$ CC" if anti else r"$(\nu_\mu+\bar{\nu}_\mu)$ CC",
        )
        step(ax, x, total, color="black", linewidth=1.2, label=labels[0])
        ax.errorbar(x, total, yerr=np.sqrt(np.maximum(total, 0.0)), fmt="none", ecolor="black", elinewidth=0.7)
        step(ax, x, beam, color="blue", linewidth=1.0, label=labels[1])
        step(ax, x, numu, color="limegreen", linewidth=1.0, label=labels[2])
        step(ax, x, nc, color="red", linewidth=1.0, label="NC")
        if dk2nu_spectra is not None:
            dk_nc = dk2nu_spectra[panel]["nc"]
            dk_numu = dk_nc + dk2nu_spectra[panel]["numu"]
            dk_beam = dk_numu + dk2nu_spectra[panel]["beam"]
            dk_total = dk_beam + dk2nu_spectra[panel]["signal"]
            step(ax, x, dk_total, color="#555555", linewidth=1.2, linestyle="--")
            step(ax, x, dk_beam, color=DK2NU_COLORS["blue"], linewidth=1.0, linestyle="--")
            step(ax, x, dk_numu, color=DK2NU_COLORS["limegreen"], linewidth=1.0, linestyle="--")
            step(ax, x, dk_nc, color=DK2NU_COLORS["red"], linewidth=1.0, linestyle="--")
    else:
        nc = spectra[panel]["nc"]
        wrong_mu = nc + spectra[panel]["wrong_mu"]
        total = wrong_mu + spectra[panel]["signal"] + spectra[panel]["tau"]
        step(ax, x, total, color="black", linewidth=1.2, label=r"Signal $\bar{\nu}_\mu$ CC" if anti else r"Signal $\nu_\mu$ CC")
        ax.errorbar(x, total, yerr=np.sqrt(np.maximum(total, 0.0)), fmt="none", ecolor="black", elinewidth=0.7)
        step(ax, x, wrong_mu, color="limegreen", linewidth=1.0, label=r"$\nu_\mu$ CC" if anti else r"$\bar{\nu}_\mu$ CC")
        step(ax, x, nc, color="red", linewidth=1.0, label="NC")
        if dk2nu_spectra is not None:
            dk_nc = dk2nu_spectra[panel]["nc"]
            dk_wrong_mu = dk_nc + dk2nu_spectra[panel]["wrong_mu"]
            dk_total = dk_wrong_mu + dk2nu_spectra[panel]["signal"] + dk2nu_spectra[panel]["tau"]
            step(ax, x, dk_total, color="#555555", linewidth=1.2, linestyle="--")
            step(ax, x, dk_wrong_mu, color=DK2NU_COLORS["limegreen"], linewidth=1.0, linestyle="--")
            step(ax, x, dk_nc, color=DK2NU_COLORS["red"], linewidth=1.0, linestyle="--")

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.0, 1.12 * max(float(np.max(total)), 1.0))
    energy_label = "Energy (GeV), $E_{rec}=E_{true}$" if no_smearing else "Reconstructed Energy (GeV)"
    ax.set_xlabel(energy_label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Events per 0.25 GeV", fontsize=10, fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.text(
        0.52,
        0.88,
        title
        + "\nDUNE ND reference"
        + ("\nNo ND smearing" if no_smearing else "\n$M_{ND}^{eff}=M_{GLoBES}$")
        + ("\nEfficiency = 1" if no_efficiency else "\nFD post-smearing efficiency")
        + f"\n$L_{{ND}}={BASELINE_KM * 1000:g}$ m"
        + f"\n{NUTIME_YEARS:g} years/mode",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
    )
    handles, labels = ax.get_legend_handles_labels()
    if dk2nu_spectra is not None:
        handles.extend(
            [
                Line2D([0], [0], color="0.2", linewidth=1.1, linestyle="-"),
                Line2D([0], [0], color="0.2", linewidth=1.1, linestyle="--"),
            ]
        )
        labels.extend(["GLoBES ND flux", "dk2nu ND absolute flux"])
    ax.legend(handles, labels, loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.98, 0.68))


def plot_figure(spectra, dk2nu_spectra, outpath, no_smearing=False, no_efficiency=False):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
    draw_panel(axes[0, 0], spectra, "FHC_app", r"$\nu_e$ Appearance", app=True, anti=False, dk2nu_spectra=dk2nu_spectra, no_smearing=no_smearing, no_efficiency=no_efficiency)
    draw_panel(axes[0, 1], spectra, "RHC_app", r"$\bar{\nu}_e$ Appearance", app=True, anti=True, dk2nu_spectra=dk2nu_spectra, no_smearing=no_smearing, no_efficiency=no_efficiency)
    draw_panel(axes[1, 0], spectra, "FHC_dis", r"$\nu_\mu$ Disappearance", app=False, anti=False, dk2nu_spectra=dk2nu_spectra, no_smearing=no_smearing, no_efficiency=no_efficiency)
    draw_panel(axes[1, 1], spectra, "RHC_dis", r"$\bar{\nu}_\mu$ Disappearance", app=False, anti=True, dk2nu_spectra=dk2nu_spectra, no_smearing=no_smearing, no_efficiency=no_efficiency)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Build DUNE ND Fig.4-like reference spectra.")
    parser.add_argument("--no-smearing", action="store_true", help="Use Erec=Etrue instead of M_ND_eff = M_GLoBES.")
    parser.add_argument("--no-efficiency", action="store_true", help="Set all detector efficiencies to one.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ideal = args.no_smearing and args.no_efficiency
    default_out = OUT_IDEAL_RESPONSE if ideal else (OUT_NO_SMEARING if args.no_smearing else OUT)
    default_csv = OUT_IDEAL_RESPONSE_CSV if ideal else (OUT_NO_SMEARING_CSV if args.no_smearing else OUT_CSV)
    out = args.out or default_out
    out_csv = args.out_csv or default_csv
    spectra = compute_spectra(out_csv=out_csv, flux_source="globes", use_smearing=not args.no_smearing, use_efficiency=not args.no_efficiency)
    dk2nu_spectra = compute_spectra(out_csv=None, flux_source="dk2nu", use_smearing=not args.no_smearing, use_efficiency=not args.no_efficiency)
    plot_figure(spectra, dk2nu_spectra, out, no_smearing=args.no_smearing, no_efficiency=args.no_efficiency)
    print(f"CSV sauvegarde: {out_csv.resolve()}")
    print(f"Figure sauvegardee: {out.resolve()}")
    for panel in ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]:
        components = [k for k in spectra[panel] if k != "Erec_GeV"]
        total = sum(spectra[panel][k] for k in components)
        print(panel, "total max", float(total.max()), "total sum", float(total.sum()))


if __name__ == "__main__":
    main()
