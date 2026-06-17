import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FHC = Path("data/dune/dk2nu/flux_z_FHC_ND.csv")
DEFAULT_RHC = Path("data/dune/dk2nu/flux_z_RHC_ND.csv")
DEFAULT_OUT = Path("figures/dune_nd/flux/dk2nu_fluxes_fhc_rhc.png")
DEFAULT_LOG_OUT = Path("figures/dune_nd/flux/dk2nu_fluxes_fhc_rhc_log.png")
DEFAULT_EXPOSURE_POT = 6.5 * 1.1e21


PLOT_SPECS = [
    ("numu", r"$\Phi_{\nu_\mu}(E)$"),
    ("nue", r"$\Phi_{\nu_e}(E)$"),
    ("numubar", r"$\Phi_{\bar{\nu}_\mu}(E)$"),
    ("nuebar", r"$\Phi_{\bar{\nu}_e}(E)$"),
]


REQUIRED_COLUMNS = [
    "flavor",
    "E_GeV_bin_low",
    "E_GeV_bin_high",
    "z_decay_m_bin_low",
    "z_decay_m_bin_high",
    "weight",
]


def read_flux_z(path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: colonnes manquantes: {missing}")
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS[1:])
    if df.empty:
        raise ValueError(f"{path}: table dk2nu vide")
    return df


def collapse_z_to_flux(df):
    grouped = (
        df.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high"], as_index=False)["weight"]
        .sum()
        .sort_values(["flavor", "E_GeV_bin_low"])
    )
    grouped["E_GeV"] = 0.5 * (grouped["E_GeV_bin_low"] + grouped["E_GeV_bin_high"])
    grouped["dE_GeV"] = grouped["E_GeV_bin_high"] - grouped["E_GeV_bin_low"]
    grouped["flux_per_GeV_per_POT"] = grouped["weight"]
    return grouped.pivot_table(
        index="E_GeV",
        columns="flavor",
        values="flux_per_GeV_per_POT",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()


def plot_fluxes(fhc, rhc, outpath, log_scale=False, emax_GeV=10.0, exposure_pot=DEFAULT_EXPOSURE_POT):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, title, df in [(axes[0], "FHC", fhc), (axes[1], "RHC", rhc)]:
        for column, label in PLOT_SPECS:
            if column not in df.columns:
                continue
            ax.plot(df["E_GeV"], exposure_pot * df[column], linewidth=1.8, label=label)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"ND dk2nu fluxes - {title}")
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_xlim(0.0, emax_GeV)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(
        rf"$\nu\,m^{{-2}}\,GeV^{{-1}}$ for ${exposure_pot:.1e}\ \mathrm{{POT}}$"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def print_checks(label, df, exposure_pot):
    print(f"{label} maxima dk2nu:")
    for column, _ in PLOT_SPECS:
        if column not in df.columns:
            print(f"  {column:7s}: absent")
            continue
        idx = df[column].idxmax()
        energy = df.loc[idx, "E_GeV"]
        raw = df.loc[idx, column]
        print(
            f"  {column:7s}: raw={raw:.6g} nu/m2/GeV/POT, "
            f"scaled={raw * exposure_pot:.6g} nu/m2/GeV at E={energy:.3g} GeV"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ND fluxes reconstructed from dk2nu flux_z tables.")
    parser.add_argument("--fhc-file", type=Path, default=DEFAULT_FHC)
    parser.add_argument("--rhc-file", type=Path, default=DEFAULT_RHC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-out", type=Path, default=DEFAULT_LOG_OUT)
    parser.add_argument("--emax-GeV", type=float, default=10.0)
    parser.add_argument("--exposure-pot", type=float, default=DEFAULT_EXPOSURE_POT)
    return parser.parse_args()


def main():
    args = parse_args()
    fhc = collapse_z_to_flux(read_flux_z(args.fhc_file))
    rhc = collapse_z_to_flux(read_flux_z(args.rhc_file))
    print_checks("FHC", fhc, args.exposure_pot)
    print_checks("RHC", rhc, args.exposure_pot)
    plot_fluxes(fhc, rhc, args.out, log_scale=False, emax_GeV=args.emax_GeV, exposure_pot=args.exposure_pot)
    plot_fluxes(fhc, rhc, args.log_out, log_scale=True, emax_GeV=args.emax_GeV, exposure_pot=args.exposure_pot)
    print(f"Figure sauvegardee: {args.out.resolve()}")
    print(f"Figure log sauvegardee: {args.log_out.resolve()}")


if __name__ == "__main__":
    main()
