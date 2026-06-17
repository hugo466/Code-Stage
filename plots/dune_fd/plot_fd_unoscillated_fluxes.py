#!/usr/bin/env python3
"""Plot direct DUNE FD unoscillated fluxes from GLoBES and dk2nu."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_FHC = Path("data/dune/flux/flux_dune_neutrino_FD_globes.csv")
DEFAULT_RHC = Path("data/dune/flux/flux_dune_antineutrino_FD_globes.csv")
DEFAULT_DK2NU_FHC = Path("data/dune/dk2nu/flux_z_FHC_FD_raw.csv")
DEFAULT_DK2NU_RHC = Path("data/dune/dk2nu/flux_z_RHC_FD_raw.csv")
DEFAULT_OUT = Path("figures/dune_fd/flux/fd_globes_vs_dk2nu_fluxes_linear.png")
DEFAULT_LOG_OUT = Path("figures/dune_fd/flux/fd_globes_vs_dk2nu_fluxes_log.png")
DEFAULT_EXPOSURE_POT = 1.1e21

FLUX_COLUMNS = [
    "E_GeV",
    "nue",
    "numu",
    "nutau",
    "nuebar",
    "numubar",
    "nutaubar",
]

PLOT_SPECS = [
    ("numu", r"$\Phi_{\nu_\mu}(E)$"),
    ("nue", r"$\Phi_{\nu_e}(E)$"),
    ("numubar", r"$\Phi_{\bar{\nu}_\mu}(E)$"),
    ("nuebar", r"$\Phi_{\bar{\nu}_e}(E)$"),
]


def read_flux(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, comment="#")
    else:
        table = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=FLUX_COLUMNS,
            decimal=",",
            comment="#",
            engine="python",
        )
    table = table.apply(pd.to_numeric, errors="coerce")
    return table.dropna(subset=["E_GeV"])


def validate_flux_table(table: pd.DataFrame, path: Path) -> None:
    missing = [column for column in FLUX_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: colonnes manquantes: {missing}")
    if table.empty:
        raise ValueError(f"{path}: table de flux vide")
    if (table["E_GeV"] <= 0.0).any():
        raise ValueError(f"{path}: energies non positives detectees")


def integrate_dk2nu_fd_flux(path: Path) -> pd.DataFrame:
    """Integrate the direct FD dk2nu weights over the decay coordinate z."""
    table = pd.read_csv(path)
    required = {
        "flavor",
        "E_GeV_bin_low",
        "E_GeV_bin_high",
        "z_decay_m_bin_low",
        "z_decay_m_bin_high",
        "weight",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: colonnes manquantes: {sorted(missing)}")

    table["E_GeV"] = 0.5 * (
        table["E_GeV_bin_low"] + table["E_GeV_bin_high"]
    )
    integrated = (
        table.groupby(["E_GeV", "flavor"], as_index=False)["weight"]
        .sum()
        .pivot(index="E_GeV", columns="flavor", values="weight")
        .reset_index()
    )
    for column in FLUX_COLUMNS[1:]:
        if column not in integrated:
            integrated[column] = 0.0
    return integrated[FLUX_COLUMNS]


def relative_residual(globes: pd.DataFrame, dk2nu: pd.DataFrame, column: str) -> pd.Series:
    reference = np.interp(dk2nu["E_GeV"], globes["E_GeV"], globes[column])
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.where(reference > 0.0, dk2nu[column].to_numpy() / reference - 1.0, np.nan)
    return pd.Series(residual, index=dk2nu.index)


def plot_fluxes(
    fhc: pd.DataFrame,
    rhc: pd.DataFrame,
    dk2nu_fhc: pd.DataFrame,
    dk2nu_rhc: pd.DataFrame,
    outpath: Path,
    *,
    log_scale: bool,
    emax_gev: float,
    exposure_pot: float,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 7),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )

    for ax, ax_res, title, globes, dk2nu in (
        (axes[0, 0], axes[1, 0], "FHC", fhc, dk2nu_fhc),
        (axes[0, 1], axes[1, 1], "RHC", rhc, dk2nu_rhc),
    ):
        flavor_handles = []
        for column, label in PLOT_SPECS:
            (line,) = ax.plot(
                globes["E_GeV"],
                exposure_pot * globes[column],
                linewidth=1.8,
                label=label,
            )
            ax.step(
                dk2nu["E_GeV"],
                exposure_pot * dk2nu[column],
                where="mid",
                color=line.get_color(),
                linewidth=1.5,
                linestyle="--",
            )
            ax_res.step(
                dk2nu["E_GeV"],
                relative_residual(globes, dk2nu, column),
                where="mid",
                color=line.get_color(),
                linewidth=1.2,
            )
            flavor_handles.append(line)

        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"FD unoscillated fluxes - {title}")
        ax.set_xlim(0.0, emax_gev)
        ax.grid(alpha=0.3)
        ax_res.axhline(0.0, color="0.25", linewidth=0.8)
        ax_res.set_xlim(0.0, emax_gev)
        ax_res.set_xlabel(r"$E_\nu$ [GeV]")
        ax_res.set_ylabel(r"$\Delta\Phi/\Phi$")
        ax_res.grid(alpha=0.3)
        style_handles = [
            Line2D([0], [0], color="black", linewidth=1.8, label="FD GLoBES"),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label=r"dk2nu FD absolute: $\sum_z d\Phi(E,z)/dE$",
            ),
        ]
        ax.legend(
            handles=[*flavor_handles, *style_handles],
            fontsize=8,
            ncol=2,
            frameon=False,
        )

    axes[0, 0].set_ylabel(
        rf"$\nu\,m^{{-2}}\,GeV^{{-1}}$ for "
        rf"${exposure_pot:.1e}\ \mathrm{{POT}}$"
    )
    axes[1, 0].set_ylim(
        min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0]),
        max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1]),
    )
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def print_flux_checks(label: str, table: pd.DataFrame, exposure_pot: float) -> None:
    print(f"{label} GLoBES maxima:")
    for column, _ in PLOT_SPECS:
        index = table[column].idxmax()
        energy = table.loc[index, "E_GeV"]
        raw = table.loc[index, column]
        print(
            f"  {column:7s}: raw={raw:.6g} nu/m2/GeV/POT, "
            f"scaled={raw * exposure_pot:.6g} at E={energy:.3g} GeV"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fhc-file", type=Path, default=DEFAULT_FHC)
    parser.add_argument("--rhc-file", type=Path, default=DEFAULT_RHC)
    parser.add_argument("--dk2nu-fhc-file", type=Path, default=DEFAULT_DK2NU_FHC)
    parser.add_argument("--dk2nu-rhc-file", type=Path, default=DEFAULT_DK2NU_RHC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-out", type=Path, default=DEFAULT_LOG_OUT)
    parser.add_argument("--emax-GeV", type=float, default=10.0)
    parser.add_argument("--exposure-pot", type=float, default=DEFAULT_EXPOSURE_POT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.fhc_file,
        args.rhc_file,
        args.dk2nu_fhc_file,
        args.dk2nu_rhc_file,
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers de flux FD manquants:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    fhc = read_flux(args.fhc_file)
    rhc = read_flux(args.rhc_file)
    validate_flux_table(fhc, args.fhc_file)
    validate_flux_table(rhc, args.rhc_file)
    dk2nu_fhc = integrate_dk2nu_fd_flux(args.dk2nu_fhc_file)
    dk2nu_rhc = integrate_dk2nu_fd_flux(args.dk2nu_rhc_file)

    print_flux_checks("FHC", fhc, args.exposure_pot)
    print_flux_checks("RHC", rhc, args.exposure_pot)
    common = dict(
        fhc=fhc,
        rhc=rhc,
        dk2nu_fhc=dk2nu_fhc,
        dk2nu_rhc=dk2nu_rhc,
        emax_gev=args.emax_GeV,
        exposure_pot=args.exposure_pot,
    )
    plot_fluxes(outpath=args.out, log_scale=False, **common)
    plot_fluxes(outpath=args.log_out, log_scale=True, **common)
    print(f"Figure sauvegardee: {args.out.resolve()}")
    print(f"Figure log sauvegardee: {args.log_out.resolve()}")


if __name__ == "__main__":
    main()
