import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = Path("data/dune/2103.04797v2/dune_globes")
OUT = Path("figures/dune_fd/reference/fig4_reconstructed_energy_globes.png")
OUT_CSV = Path("data/dune/validation/fig4_reconstructed_energy_spectra.csv")
OUT_NO_SMEARING = Path("figures/dune_fd/reference/fig4_reconstructed_energy_globes_no_smearing.png")
OUT_NO_SMEARING_CSV = Path("data/dune/validation/fig4_reconstructed_energy_spectra_no_smearing.csv")
OUT_IDEAL_RESPONSE = Path("figures/dune_fd/reference/fig4_reconstructed_energy_globes_no_smearing_no_efficiency.png")
OUT_IDEAL_RESPONSE_CSV = Path(
    "data/dune/validation/fig4_reconstructed_energy_spectra_no_smearing_no_efficiency.csv"
)

AVOGADRO = 6.02214076e23
M2_TO_CM2 = 1.0e4
XSEC_SCALE_CM2 = 1.0e-38
BASELINE_KM = 1284.9
DENSITY_G_CM3 = 2.848
YE = 0.5
MATTER_A_COEFF = 1.52e-4
POT_PER_YEAR = 11.0e20
NUTIME_YEARS = 6.5
NUBARTIME_YEARS = 6.5
TARGET_MASS_KT = 40.0
TRUE_ENERGY_MAX_GEV = 20.0
# Beam.inc uses @norm = 1.017718e17. The physical event-rate convention
# implemented below already contains the canonical 1e17 factor.
GLOBES_EVENT_NORM = 1.017718
DK2NU_FHC = Path("data/dune/dk2nu/flux_z_FHC_FD_raw.csv")
DK2NU_RHC = Path("data/dune/dk2nu/flux_z_RHC_FD_raw.csv")

# NuFit 4.0 normal-ordering central values used in Table II of
# arXiv:2103.04797v2.
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


def split_globes_list(text):
    return [x.strip() for x in text.replace("\n", " ").split(",") if x.strip()]


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


def read_smearing(path):
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"\{([^{}]+)\}", text, flags=re.S)
    matrix = []
    for row in rows:
        values = [float(x) for x in split_globes_list(row)]
        if len(values) < 3:
            continue
        lo = int(values[0])
        hi = int(values[1])
        payload = values[2:]
        full = np.zeros(hi + 1, dtype=float)
        full[lo : hi + 1] = payload[: hi - lo + 1]
        matrix.append(full)
    if not matrix:
        raise ValueError(f"No smearing matrix found in {path}")
    return np.vstack(matrix)


def parse_glb_vector(path, name):
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\${name}\s*=\s*\{{([^}}]+)\}}", text, flags=re.S)
    if not match:
        raise ValueError(f"{name} not found in {path}")
    return np.asarray([float(x.strip()) for x in match.group(1).split(",") if x.strip()], dtype=float)


def bin_centers(widths):
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    return 0.5 * (edges[:-1] + edges[1:]), edges


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
    p = abs(amp[beta, alpha]) ** 2
    return float(np.clip(p, 0.0, 1.0))


def probability_vector(initial, final, energies, anti=False, no_osc=False):
    if no_osc:
        return np.ones_like(energies, dtype=float)
    flavor_index = {"e": 0, "m": 1, "t": 2}
    a = flavor_index[initial]
    b = flavor_index[final]
    return np.asarray([prob(a, b, e, anti=anti) for e in energies], dtype=float)


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


def identity_response(true_edges, rec_edges):
    """Project true-bin counts onto reconstructed bins with Erec = Etrue."""
    response = np.zeros((len(rec_edges) - 1, len(true_edges) - 1), dtype=float)
    for true_index, (true_low, true_high) in enumerate(zip(true_edges[:-1], true_edges[1:])):
        true_width = true_high - true_low
        if true_width <= 0.0:
            continue
        overlap_low = np.maximum(rec_edges[:-1], true_low)
        overlap_high = np.minimum(rec_edges[1:], true_high)
        response[:, true_index] = np.maximum(0.0, overlap_high - overlap_low) / true_width
    return response


def compute_channel(
    channel,
    datasets,
    sampling_centers,
    sampling_widths,
    sampling_edges,
    rec_edges,
    use_smearing,
    use_efficiency,
):
    flux = datasets["flux"][channel.flux_mode]
    xsec = datasets["xsec"][channel.interaction]
    if use_smearing:
        response = read_smearing(BASE / "smr" / f"{channel.smear}.txt")
    else:
        response = identity_response(sampling_edges, rec_edges)
    eff = read_post_eff(BASE / "eff" / f"{channel.eff}.txt") if use_efficiency else None

    flux_col = initial_flux_flavor(channel.initial, channel.anti)
    xsec_col = final_xsec_flavor(channel.final, channel.anti) if channel.interaction == "cc" else flux_col
    flux_values = interp(flux, flux_col, sampling_centers) / M2_TO_CM2
    xsec_hat = interp(xsec, xsec_col, sampling_centers)
    xsec_values = xsec_hat * sampling_centers * XSEC_SCALE_CM2
    probs = probability_vector(
        channel.initial,
        channel.final,
        sampling_centers,
        anti=channel.anti,
        no_osc=channel.no_osc,
    )
    pot = POT_PER_YEAR * (NUBARTIME_YEARS if channel.flux_mode == "RHC" else NUTIME_YEARS)
    target_nucleons = TARGET_MASS_KT * 1.0e9 * AVOGADRO
    true_counts = (
        GLOBES_EVENT_NORM
        * flux_values
        * xsec_values
        * probs
        * sampling_widths
        * pot
        * target_nucleons
    )
    true_counts = np.where(sampling_centers <= TRUE_ENERGY_MAX_GEV, true_counts, 0.0)
    reco = response @ true_counts
    return reco * eff if use_efficiency else reco


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
            Channel("FHC", "m", "m", "cc", "app_numu_bkg", "post_app_FHC_numu_bkg", no_osc=True),
            Channel("FHC", "m", "m", "cc", "app_numubar_bkg", "post_app_FHC_numubar_bkg", anti=True, no_osc=True),
        ],
        "nc": [
            Channel("FHC", "m", "m", "nc", "app_NC_bkg", "post_app_FHC_NC_bkg", no_osc=True),
            Channel("FHC", "m", "m", "nc", "app_NC_bkg", "post_app_FHC_aNC_bkg", anti=True, no_osc=True),
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
            Channel("RHC", "m", "m", "cc", "app_numu_bkg", "post_app_RHC_numu_bkg", no_osc=True),
            Channel("RHC", "m", "m", "cc", "app_numubar_bkg", "post_app_RHC_numubar_bkg", anti=True, no_osc=True),
        ],
        "nc": [
            Channel("RHC", "m", "m", "nc", "app_NC_bkg", "post_app_RHC_NC_bkg", no_osc=True),
            Channel("RHC", "m", "m", "nc", "app_aNC_bkg", "post_app_RHC_aNC_bkg", anti=True, no_osc=True),
        ],
    },
    "FHC_dis": {
        "signal": [
            Channel("FHC", "m", "m", "cc", "dis_numu_sig", "post_dis_FHC_numu_sig"),
        ],
        "wrong_mu": [
            Channel("FHC", "m", "m", "cc", "dis_numubar_sig", "post_dis_FHC_numubar_sig", anti=True),
        ],
        "tau": [
            Channel("FHC", "m", "t", "cc", "dis_nutau_bkg", "post_dis_FHC_nutau_bkg"),
            Channel("FHC", "m", "t", "cc", "dis_nutaubar_bkg", "post_dis_FHC_nutaubar_bkg", anti=True),
        ],
        "nc": [
            Channel("FHC", "m", "m", "nc", "dis_NC_bkg", "post_dis_FHC_NC_bkg", no_osc=True),
            Channel("FHC", "m", "m", "nc", "dis_aNC_bkg", "post_dis_FHC_NC_bkg", anti=True, no_osc=True),
        ],
    },
    "RHC_dis": {
        "signal": [
            Channel("RHC", "m", "m", "cc", "dis_numubar_sig", "post_dis_RHC_numubar_sig", anti=True),
        ],
        "wrong_mu": [
            Channel("RHC", "m", "m", "cc", "dis_numu_sig", "post_dis_RHC_numu_sig"),
        ],
        "tau": [
            Channel("RHC", "m", "t", "cc", "dis_nutau_bkg", "post_dis_RHC_nutau_bkg"),
            Channel("RHC", "m", "t", "cc", "dis_nutaubar_bkg", "post_dis_RHC_nutaubar_bkg", anti=True),
        ],
        "nc": [
            Channel("RHC", "m", "m", "nc", "dis_NC_bkg", "post_dis_RHC_NC_bkg", no_osc=True),
            Channel("RHC", "m", "m", "nc", "dis_aNC_bkg", "post_dis_RHC_NC_bkg", anti=True, no_osc=True),
        ],
    },
}


def compute_spectra(
    use_smearing=True,
    use_efficiency=True,
    out_csv=OUT_CSV,
    flux_source="globes",
):
    binsize = parse_glb_vector(BASE / "DUNE_GLoBES.glb", "binsize")
    sampling = parse_glb_vector(BASE / "DUNE_GLoBES.glb", "sampling_stepsize")
    rec_centers, rec_edges = bin_centers(binsize)
    sampling_centers, sampling_edges = bin_centers(sampling)

    if flux_source == "globes":
        fhc_flux = read_flux(
            BASE / "flux" / "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEFD_globes_flux.txt"
        )
        rhc_flux = read_flux(
            BASE / "flux" / "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_antineutrino_LBNEFD_globes_flux.txt"
        )
    elif flux_source == "dk2nu":
        fhc_flux = read_dk2nu_flux(DK2NU_FHC)
        rhc_flux = read_dk2nu_flux(DK2NU_RHC)
    else:
        raise ValueError(f"Unknown flux source: {flux_source}")

    datasets = {
        "flux": {
            "FHC": fhc_flux,
            "RHC": rhc_flux,
        },
        "xsec": {
            "cc": read_xsec(BASE / "xsec" / "xsec_cc.dat"),
            "nc": read_xsec(BASE / "xsec" / "xsec_nc.dat"),
        },
    }

    spectra = {}
    rows = []
    for panel, groups in CHANNELS.items():
        spectra[panel] = {}
        for group, channels in groups.items():
            reco = np.zeros_like(rec_centers)
            for ch in channels:
                reco += compute_channel(
                    ch,
                    datasets,
                    sampling_centers,
                    sampling,
                    sampling_edges,
                    rec_edges,
                    use_smearing,
                    use_efficiency,
                )
            e025, y025 = aggregate_025(reco, rec_edges)
            spectra[panel][group] = y025
            for e, y in zip(e025, y025):
                rows.append({"panel": panel, "component": group, "Erec_GeV": e, "events_per_0p25_GeV": y})
        spectra[panel]["Erec_GeV"] = e025

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return spectra


def step(ax, x, y, **kwargs):
    ax.step(x, y, where="mid", **kwargs)


def draw_panel(
    ax,
    spectra,
    panel,
    title,
    ylim,
    app=True,
    anti=False,
    no_smearing=False,
    no_efficiency=False,
    dk2nu_spectra=None,
    header="DUNE TDR GLoBES",
    propagation_note=None,
):
    x = spectra[panel]["Erec_GeV"]
    if app:
        nc = spectra[panel]["nc"]
        numu = nc + spectra[panel]["numu"]
        beam = numu + spectra[panel]["beam"]
        total = beam + spectra[panel]["signal"]
        app_label = r"Signal $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Signal $(\nu_e+\bar{\nu}_e)$ CC"
        beam_label = r"Beam $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Beam $(\nu_e+\bar{\nu}_e)$ CC"
        numu_label = r"$(\bar{\nu}_\mu+\nu_\mu)$ CC" if anti else r"$(\nu_\mu+\bar{\nu}_\mu)$ CC"
        step(ax, x, total, color="black", linewidth=1.2, label=app_label)
        ax.errorbar(x, total, yerr=np.sqrt(np.maximum(total, 0.0)), fmt="none", ecolor="black", elinewidth=0.7)
        step(ax, x, beam, color="blue", linewidth=1.0, label=beam_label)
        step(ax, x, numu, color="limegreen", linewidth=1.0, label=numu_label)
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
        signal_label = r"Signal $\bar{\nu}_\mu$ CC" if anti else r"Signal $\nu_\mu$ CC"
        wrong_mu_label = r"$\nu_\mu$ CC" if anti else r"$\bar{\nu}_\mu$ CC"
        step(ax, x, total, color="black", linewidth=1.2, label=signal_label)
        ax.errorbar(x, total, yerr=np.sqrt(np.maximum(total, 0.0)), fmt="none", ecolor="black", elinewidth=0.7)
        step(ax, x, wrong_mu, color="limegreen", linewidth=1.0, label=wrong_mu_label)
        step(ax, x, nc, color="red", linewidth=1.0, label="NC")
        if dk2nu_spectra is not None:
            dk_nc = dk2nu_spectra[panel]["nc"]
            dk_wrong_mu = dk_nc + dk2nu_spectra[panel]["wrong_mu"]
            dk_total = (
                dk_wrong_mu
                + dk2nu_spectra[panel]["signal"]
                + dk2nu_spectra[panel]["tau"]
            )
            step(ax, x, dk_total, color="#555555", linewidth=1.2, linestyle="--")
            step(ax, x, dk_wrong_mu, color=DK2NU_COLORS["limegreen"], linewidth=1.0, linestyle="--")
            step(ax, x, dk_nc, color=DK2NU_COLORS["red"], linewidth=1.0, linestyle="--")

    ax.set_xlim(0.5, 7.5)
    if ylim is None:
        ax.set_ylim(0.0, 1.08 * float(np.max(total)))
    else:
        ax.set_ylim(0.0, ylim)
    energy_label = "Energy (GeV), $E_{rec}=E_{true}$" if no_smearing else "Reconstructed Energy (GeV)"
    ax.set_xlabel(energy_label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Events per 0.25 GeV", fontsize=10, fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.text(
        0.62,
        0.88,
        title
        + f"\n{header}"
        + ("\nNo smearing" if no_smearing else "")
        + ("\nEfficiency = 1" if no_efficiency else "")
        + (f"\n{propagation_note}" if propagation_note else "")
        + f"\n$E_{{true}}\\leq {TRUE_ENERGY_MAX_GEV:g}$ GeV"
        + f"\n{NUTIME_YEARS:g} years/mode\nNormal Ordering, $\\delta_{{CP}}=0$",
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
        labels.extend(["GLoBES flux", "dk2nu absolute flux"])
    ax.legend(handles, labels, loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.98, 0.70))


def plot_figure(
    spectra,
    dk2nu_spectra,
    outpath,
    no_smearing=False,
    no_efficiency=False,
    header="DUNE TDR GLoBES",
    propagation_note=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
    limits = (None, None, None, None) if no_smearing else (280, 95, 1450, 620)
    draw_panel(
        axes[0, 0], spectra, "FHC_app", r"$\nu_e$ Appearance", limits[0],
        app=True, anti=False, no_smearing=no_smearing, no_efficiency=no_efficiency,
        dk2nu_spectra=dk2nu_spectra,
        header=header, propagation_note=propagation_note,
    )
    draw_panel(
        axes[0, 1], spectra, "RHC_app", r"$\bar{\nu}_e$ Appearance", limits[1],
        app=True, anti=True, no_smearing=no_smearing, no_efficiency=no_efficiency,
        dk2nu_spectra=dk2nu_spectra,
        header=header, propagation_note=propagation_note,
    )
    draw_panel(
        axes[1, 0], spectra, "FHC_dis", r"$\nu_\mu$ Disappearance", limits[2],
        app=False, anti=False, no_smearing=no_smearing, no_efficiency=no_efficiency,
        dk2nu_spectra=dk2nu_spectra,
        header=header, propagation_note=propagation_note,
    )
    draw_panel(
        axes[1, 1], spectra, "RHC_dis", r"$\bar{\nu}_\mu$ Disappearance", limits[3],
        app=False, anti=True, no_smearing=no_smearing, no_efficiency=no_efficiency,
        dk2nu_spectra=dk2nu_spectra,
        header=header, propagation_note=propagation_note,
    )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce arXiv:2103.04797 Fig. 4 using the ancillary GLoBES files.")
    parser.add_argument("--no-smearing", action="store_true", help="Use Erec = Etrue instead of GLoBES migration matrices")
    parser.add_argument("--no-efficiency", action="store_true", help="Set all detector efficiencies to one")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ideal_response = args.no_smearing and args.no_efficiency
    default_out = OUT_IDEAL_RESPONSE if ideal_response else (OUT_NO_SMEARING if args.no_smearing else OUT)
    default_csv = (
        OUT_IDEAL_RESPONSE_CSV
        if ideal_response
        else (OUT_NO_SMEARING_CSV if args.no_smearing else OUT_CSV)
    )
    out = args.out or default_out
    out_csv = args.out_csv or default_csv
    spectra = compute_spectra(
        use_smearing=not args.no_smearing,
        use_efficiency=not args.no_efficiency,
        out_csv=out_csv,
        flux_source="globes",
    )
    dk2nu_spectra = compute_spectra(
        use_smearing=not args.no_smearing,
        use_efficiency=not args.no_efficiency,
        out_csv=None,
        flux_source="dk2nu",
    )
    plot_figure(
        spectra,
        dk2nu_spectra,
        out,
        no_smearing=args.no_smearing,
        no_efficiency=args.no_efficiency,
    )
    print(f"CSV sauvegarde: {out_csv.resolve()}")
    print(f"Figure sauvegardee: {out.resolve()}")
    for panel in ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]:
        components = [k for k in spectra[panel] if k != "Erec_GeV"]
        total = sum(spectra[panel][k] for k in components)
        print(panel, "total max", float(total.max()), "total sum", float(total.sum()))


if __name__ == "__main__":
    main()
