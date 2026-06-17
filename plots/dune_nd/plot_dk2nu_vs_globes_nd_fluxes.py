import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FHC_DK2NU = Path("data/dune/dk2nu/flux_z_FHC_ND_raw.csv")
DEFAULT_RHC_DK2NU = Path("data/dune/dk2nu/flux_z_RHC_ND_raw.csv")
DEFAULT_FHC_GLOBES = Path("data/dune/flux/flux_dune_neutrino_ND_globes.txt")
DEFAULT_RHC_GLOBES = Path("data/dune/flux/flux_dune_antineutrino_ND_globes.txt")
DEFAULT_OUT = Path("figures/dune_nd/flux/nd_globes_vs_dk2nu_fluxes_linear.png")
DEFAULT_LOG_OUT = Path("figures/dune_nd/flux/nd_globes_vs_dk2nu_fluxes_log.png")

GLOBES_COLUMNS = [
    "E_GeV",
    "nue",
    "numu",
    "nutau",
    "nuebar",
    "numubar",
    "nutaubar",
]

FLAVORS = [
    ("numu", r"$\nu_\mu$", "#1f77b4"),
    ("nue", r"$\nu_e$", "#d62728"),
    ("numubar", r"$\bar{\nu}_\mu$", "#2ca02c"),
    ("nuebar", r"$\bar{\nu}_e$", "#9467bd"),
]


def read_globes_flux(path):
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=GLOBES_COLUMNS,
        decimal=",",
        comment="#",
        engine="python",
    ).apply(pd.to_numeric, errors="coerce").dropna(subset=["E_GeV"])


def integrate_dk2nu_over_z(path):
    table = pd.read_csv(path)
    numeric = [
        "E_GeV_bin_low",
        "E_GeV_bin_high",
        "z_decay_m_bin_low",
        "z_decay_m_bin_high",
        "weight",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["flavor", *numeric])
    profile_sums = table.groupby(
        ["flavor", "E_GeV_bin_low", "E_GeV_bin_high"],
        as_index=False,
    )["weight"].sum()
    if (profile_sums["weight"].between(0.99, 1.01)).mean() > 0.8:
        raise ValueError(
            f"{path} ressemble a un profil normalise p(z|E,flavor), pas a un flux dk2nu absolu. "
            "Utilise flux_z_*_ND_raw.csv pour ce plot."
        )

    flux = (
        table.groupby(
            ["flavor", "E_GeV_bin_low", "E_GeV_bin_high"],
            as_index=False,
        )["weight"]
        .sum()
        .rename(columns={"weight": "sum_z_weight"})
    )
    flux["E_GeV"] = 0.5 * (flux["E_GeV_bin_low"] + flux["E_GeV_bin_high"])
    flux["delta_E_GeV"] = flux["E_GeV_bin_high"] - flux["E_GeV_bin_low"]
    if (flux["delta_E_GeV"] <= 0.0).any():
        raise ValueError(f"{path}: largeur de bin d'energie non positive")

    flux["flux_per_GeV"] = flux["sum_z_weight"]
    return flux


def flavor_curve(table, flavor):
    return table.loc[table["flavor"] == flavor].sort_values("E_GeV")


def plot_mode(ax_flux, ax_ratio, mode, dk2nu, globes, emax, *, log_scale):
    for flavor, label, color in FLAVORS:
        curve = flavor_curve(dk2nu, flavor)
        if curve.empty:
            continue

        energy = curve["E_GeV"].to_numpy(dtype=float)
        dk_flux = curve["flux_per_GeV"].to_numpy(dtype=float)
        gl_flux = np.interp(
            energy,
            globes["E_GeV"].to_numpy(dtype=float),
            globes[flavor].to_numpy(dtype=float),
        )
        ratio = np.divide(
            dk_flux,
            gl_flux,
            out=np.full_like(dk_flux, np.nan),
            where=gl_flux > 0.0,
        )
        residual = ratio - 1.0
        ax_flux.plot(
            globes["E_GeV"],
            globes[flavor],
            color=color,
            linewidth=1.8,
            label=f"{label} GLoBES ND",
        )
        ax_flux.step(
            energy,
            dk_flux,
            where="mid",
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label} dk2nu raw, $\\sum_z d\\Phi/dE$",
        )
        ax_ratio.step(energy, residual, where="mid", color=color, linewidth=1.4)

        finite = np.isfinite(ratio)
        if np.any(finite):
            print(
                f"{mode} {flavor}: residual (dk2nu-GLoBES)/GLoBES "
                f"median={np.nanmedian(residual):.6g}, "
                f"min={np.nanmin(residual):.6g}, max={np.nanmax(residual):.6g}"
            )

    if log_scale:
        ax_flux.set_yscale("log")
    ax_flux.set_xlim(0.0, emax)
    ax_flux.set_title(mode)
    ax_flux.set_ylabel(r"Flux ND [$\nu\,m^{-2}\,GeV^{-1}\,POT^{-1}$]")
    ax_flux.grid(alpha=0.25)
    ax_flux.legend(fontsize=7, ncol=2)

    ax_ratio.axhline(0.0, color="black", linewidth=1.0)
    ax_ratio.set_xlim(0.0, emax)
    ax_ratio.set_xlabel(r"$E_\nu$ [GeV]")
    ax_ratio.set_ylabel(r"$\Delta\Phi/\Phi$")
    ax_ratio.grid(alpha=0.25)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sum_z(weight) from dk2nu with the tabulated DUNE ND flux."
    )
    parser.add_argument("--fhc-dk2nu", type=Path, default=DEFAULT_FHC_DK2NU)
    parser.add_argument("--rhc-dk2nu", type=Path, default=DEFAULT_RHC_DK2NU)
    parser.add_argument("--fhc-globes", type=Path, default=DEFAULT_FHC_GLOBES)
    parser.add_argument("--rhc-globes", type=Path, default=DEFAULT_RHC_GLOBES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-out", type=Path, default=DEFAULT_LOG_OUT)
    parser.add_argument("--emax-GeV", type=float, default=8.0)
    return parser.parse_args()


def write_comparison_figure(
    outpath,
    fhc_dk2nu,
    rhc_dk2nu,
    fhc_globes,
    rhc_globes,
    *,
    emax_gev,
    log_scale,
):
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.08, wspace=0.22)
    fhc_flux = fig.add_subplot(grid[0, 0])
    rhc_flux = fig.add_subplot(grid[0, 1], sharey=fhc_flux)
    fhc_ratio = fig.add_subplot(grid[1, 0], sharex=fhc_flux)
    rhc_ratio = fig.add_subplot(grid[1, 1], sharex=rhc_flux)

    plot_mode(fhc_flux, fhc_ratio, "FHC", fhc_dk2nu, fhc_globes, emax_gev, log_scale=log_scale)
    plot_mode(rhc_flux, rhc_ratio, "RHC", rhc_dk2nu, rhc_globes, emax_gev, log_scale=log_scale)
    rhc_flux.set_ylabel("")

    fig.suptitle(
        r"Flux ND: intégration dk2nu sur la source, "
        r"$\Phi(E_i)=\sum_z w(E_i,z)$",
        fontsize=13,
    )
    fig.suptitle(
        r"ND flux: absolute dk2nu integrated over source, "
        r"$d\Phi/dE=\sum_z d\Phi(E_i,z)/dE$",
        fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardee: {outpath.resolve()}")


def main():
    args = parse_args()
    fhc_dk2nu = integrate_dk2nu_over_z(args.fhc_dk2nu)
    rhc_dk2nu = integrate_dk2nu_over_z(args.rhc_dk2nu)
    fhc_globes = read_globes_flux(args.fhc_globes)
    rhc_globes = read_globes_flux(args.rhc_globes)

    common = dict(
        fhc_dk2nu=fhc_dk2nu,
        rhc_dk2nu=rhc_dk2nu,
        fhc_globes=fhc_globes,
        rhc_globes=rhc_globes,
        emax_gev=args.emax_GeV,
    )
    write_comparison_figure(args.out, log_scale=False, **common)
    write_comparison_figure(args.log_out, log_scale=True, **common)


if __name__ == "__main__":
    main()
