import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_XSEC = Path("data/dune/2103.04797v2/dune_globes/xsec/xsec_cc.dat")
DEFAULT_XSEC_CSV = Path("data/dune/validation/xsec_cc.csv")
DEFAULT_FHC_FLUX = Path("data/dune/flux/flux_dune_neutrino_FD_globes.csv")
DEFAULT_RHC_FLUX = Path("data/dune/flux/flux_dune_antineutrino_FD_globes.csv")
DEFAULT_OUT = Path("figures/dune_fd/detector_response/cc_inclusive_argon_with_fd_flux.png")

XSEC_COLUMNS = ["E_GeV", "nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
FLUX_COLUMNS = ["E_GeV", "nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
M2_TO_CM2 = 1.0e4
FLUX_AXIS_UNIT = 1.0e-14


def parse_number(token):
    return float(token.replace(",", "."))


def read_xsec_globes(path):
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        tokens.extend(clean.split())

    values = [parse_number(token) for token in tokens]
    if len(values) % 7 != 0:
        raise ValueError(f"{path}: nombre de valeurs incompatible avec des lignes GLoBES a 7 colonnes")

    rows = np.asarray(values, dtype=float).reshape((-1, 7))
    out = pd.DataFrame(rows, columns=["log10_E_GeV", *XSEC_COLUMNS[1:]])
    out.insert(0, "E_GeV", np.power(10.0, out.pop("log10_E_GeV")))
    return out


def read_flux(path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, comment="#")
    else:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=FLUX_COLUMNS,
            decimal=",",
            comment="#",
            engine="python",
        )
    return df.apply(pd.to_numeric, errors="coerce").dropna(subset=["E_GeV"])


def write_xsec_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def flux_in_axis_units(flux_df, column):
    e = flux_df["E_GeV"].to_numpy(dtype=float)
    y = flux_df[column].to_numpy(dtype=float) / M2_TO_CM2 / FLUX_AXIS_UNIT
    mask = np.isfinite(e) & np.isfinite(y) & (y >= 0.0)
    return e[mask], y[mask]


def plot_xsec_with_flux(
    xsec,
    flux,
    outpath,
    xsec_flavor="numu",
    flux_flavor="numu",
    emax_GeV=8.0,
):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    xmask = (xsec["E_GeV"] >= 0.0) & (xsec["E_GeV"] <= emax_GeV)
    xsec_plot = xsec.loc[xmask].copy()
    if xsec_plot.empty:
        raise ValueError("Aucun point de section efficace dans la plage d'energie demandee")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    ax.plot(
        xsec_plot["E_GeV"],
        xsec_plot[xsec_flavor],
        color="black",
        linewidth=2.2,
        label="CC Inclusive",
    )

    ax.set_xlim(0.0, emax_GeV)
    ax.set_ylim(0.0, max(1.15 * float(xsec_plot[xsec_flavor].max()), 1.0))
    ax.set_xlabel(r"$E_\nu$ [GeV]", fontsize=16)
    ax.set_ylabel(r"$\sigma(E_\nu)/E_\nu\quad 10^{-38}\ (cm^2/GeV/Nucleon)$", fontsize=16)
    ax.set_title("GENIE 2.12, DUNE FD TDR CV Tune", fontsize=17)
    ax.tick_params(direction="in", top=True, right=False, labelsize=13, length=6)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=False, length=3)

    if flux is not None:
        fmask = (flux["E_GeV"] >= 0.0) & (flux["E_GeV"] <= emax_GeV)
        flux_plot = flux.loc[fmask].copy()
        if not flux_plot.empty:
            ax_flux = ax.twinx()
            e, y = flux_in_axis_units(flux_plot, flux_flavor)
            right_top = max(0.5, float(np.max(y)) / 0.78) if y.size else 0.5
            ax_flux.fill_between(e, y, color="#88a9cf", alpha=0.5, linewidth=0.0, label=rf"$\Phi^{{FD}}_{{{flux_flavor}}}$")
            ax_flux.set_ylim(0.0, right_top)
            ax_flux.set_ylabel(
                r"$\Phi^{FD}_{\nu_\mu}(E_\nu)\ 10^{-14}\ (cm^{-2}/GeV/POT)$",
                fontsize=15,
            )
            ax_flux.tick_params(direction="in", top=True, right=True, labelsize=13, length=6)
            ax_flux.minorticks_on()
            ax_flux.tick_params(which="minor", direction="in", top=True, right=True, length=3)

    ax.legend(loc="upper left", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot inclusive CC argon cross section and optional FD flux overlay.")
    parser.add_argument("--xsec-file", type=Path, default=DEFAULT_XSEC)
    parser.add_argument("--xsec-csv", type=Path, default=DEFAULT_XSEC_CSV)
    parser.add_argument("--flux-file", type=Path, default=DEFAULT_FHC_FLUX)
    parser.add_argument("--no-flux", action="store_true")
    parser.add_argument("--xsec-flavor", choices=XSEC_COLUMNS[1:], default="numu")
    parser.add_argument("--flux-flavor", choices=FLUX_COLUMNS[1:], default="numu")
    parser.add_argument("--emax-GeV", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.xsec_file.exists():
        raise FileNotFoundError(args.xsec_file)

    xsec = read_xsec_globes(args.xsec_file)
    write_xsec_csv(xsec, args.xsec_csv)

    flux = None
    if not args.no_flux:
        if not args.flux_file.exists():
            raise FileNotFoundError(args.flux_file)
        flux = read_flux(args.flux_file)

    plot_xsec_with_flux(
        xsec,
        flux,
        args.out,
        xsec_flavor=args.xsec_flavor,
        flux_flavor=args.flux_flavor,
        emax_GeV=args.emax_GeV,
    )

    print(f"CSV section efficace sauvegarde: {args.xsec_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")
    print("Note: les fichiers auxiliaires arXiv ne fournissent que CC inclusive/NC inclusive, pas les composantes 1p1h+2p2h, Res 1pi, DIS.")


if __name__ == "__main__":
    main()
