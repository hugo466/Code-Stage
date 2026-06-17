#!/usr/bin/env python3
"""Plot the effective DUNE ND migration matrices M_eff = M_GLoBES."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[2]
ND_REFERENCE_SCRIPT = ROOT / "plots" / "dune_nd" / "plot_fig4_reconstructed_energy_nd.py"
DEFAULT_OUTPUT = ROOT / "figures" / "dune_nd" / "detector_response" / "nd_effective_smearing_matrices.png"

MATRIX_GROUPS = (
    ("Appearance signal", ("app_nue_sig", "app_nuebar_sig")),
    (
        "Appearance CC backgrounds",
        ("app_nue_bkg", "app_nuebar_bkg", "app_numu_bkg", "app_numubar_bkg"),
    ),
    (
        "Appearance tau/NC backgrounds",
        ("app_nutau_bkg", "app_nutaubar_bkg", "app_NC_bkg", "app_aNC_bkg"),
    ),
    (
        "Disappearance",
        (
            "dis_numu_sig",
            "dis_numubar_sig",
            "dis_nutau_bkg",
            "dis_nutaubar_bkg",
            "dis_NC_bkg",
            "dis_aNC_bkg",
        ),
    ),
)


def load_reference_module():
    spec = importlib.util.spec_from_file_location("nd_fig4_reference", ND_REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot import {ND_REFERENCE_SCRIPT}")
    spec.loader.exec_module(module)
    return module


def display_name(name: str) -> str:
    return (
        name.replace("app_", "app: ")
        .replace("dis_", "dis: ")
        .replace("_sig", " signal")
        .replace("_bkg", " background")
        .replace("nuebar", r"$\bar{\nu}_e$")
        .replace("numubar", r"$\bar{\nu}_\mu$")
        .replace("nutaubar", r"$\bar{\nu}_\tau$")
        .replace("nue", r"$\nu_e$")
        .replace("numu", r"$\nu_\mu$")
        .replace("nutau", r"$\nu_\tau$")
        .replace("aNC", r"$\bar{\nu}$ NC")
        .replace("NC", r"$\nu$ NC")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-max", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ref = load_reference_module()
    glb_path = ref.GLOBES / "DUNE_GLoBES.glb"
    true_widths = ref.parse_glb_vector(glb_path, "sampling_stepsize")
    reco_widths = ref.parse_glb_vector(glb_path, "binsize")
    _, true_edges = ref.bin_centers(true_widths)
    _, reco_edges = ref.bin_centers(reco_widths)

    names = [name for _, group in MATRIX_GROUPS for name in group]
    matrices = {
        name: ref.nd_effective_response(name, true_edges, reco_edges)
        for name in names
    }

    positive = np.concatenate([matrix[matrix > 0.0] for matrix in matrices.values()])
    norm = LogNorm(vmin=max(1.0e-6, float(np.percentile(positive, 2.0))), vmax=min(1.0, float(np.max(positive))))

    fig, axes = plt.subplots(4, 4, figsize=(15.5, 14.0), sharex=True, sharey=True)
    image = None
    index = 0
    for group_name, group in MATRIX_GROUPS:
        for name in group:
            ax = axes.flat[index]
            matrix = matrices[name]
            used_reco_edges = reco_edges[: matrix.shape[0] + 1]
            image = ax.pcolormesh(
                true_edges,
                used_reco_edges,
                np.ma.masked_less_equal(matrix, 0.0),
                shading="flat",
                cmap="magma",
                norm=norm,
                rasterized=True,
            )
            ax.plot([0.0, args.energy_max], [0.0, args.energy_max], color="white", lw=0.7, alpha=0.7)
            ax.set_title(display_name(name), fontsize=10)
            ax.set_xlim(0.0, args.energy_max)
            ax.set_ylim(0.0, args.energy_max)
            if index % 4 == 0:
                ax.text(
                    -0.34,
                    0.5,
                    group_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
            index += 1

    for ax in axes[-1, :]:
        ax.set_xlabel(r"True energy $E_\mathrm{true}$ (GeV)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"ND reconstructed energy $E_\mathrm{rec}$ (GeV)")

    fig.suptitle(r"DUNE ND effective migration matrices: $M_{ND}^{eff}=M_{GLoBES}$", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.07, top=0.94, wspace=0.12, hspace=0.22)
    colorbar_ax = fig.add_axes((0.93, 0.15, 0.018, 0.72))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"Migration weight $M_{ND}^{eff}(E_\mathrm{rec}, E_\mathrm{true})$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Saved {args.output}")
    for name, matrix in matrices.items():
        column_sums = matrix.sum(axis=0)
        nonzero = column_sums[column_sums > 0.0]
        print(
            f"{name}: shape={matrix.shape}, "
            f"sum_reco median={np.median(nonzero):.6g}, "
            f"range=[{nonzero.min():.6g}, {nonzero.max():.6g}]"
        )


if __name__ == "__main__":
    main()
