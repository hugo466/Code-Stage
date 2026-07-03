#!/usr/bin/env python3
"""Plot the DUNE FD GLoBES migration matrices used by the Fig. 4 code."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[2]
GLOBES_DIR = ROOT / "data" / "dune" / "2103.04797v2" / "dune_globes"
DEFAULT_OUTPUT = ROOT / "figures" / "dune_fd" / "detector_response" / "fd_smearing_matrices.png"

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


def parse_globes_vector(path: Path, name: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\${re.escape(name)}\s*=\s*\{{([^{{}}]+)\}}", text, flags=re.S)
    if not match:
        raise ValueError(f"Vector ${name} not found in {path}")
    return np.asarray([float(value.strip()) for value in match.group(1).split(",")], dtype=float)


def bin_edges(widths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(widths)))


def read_smearing(path: Path, n_true: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"\{([^{}]+)\}", text, flags=re.S)
    matrix = np.zeros((len(rows), n_true), dtype=float)
    for reco_index, row in enumerate(rows):
        values = [float(value.strip()) for value in row.split(",")]
        if len(values) < 3:
            continue
        first_true = max(0, int(round(values[0])))
        last_true = min(n_true - 1, int(round(values[1])))
        count = max(0, last_true - first_true + 1)
        matrix[reco_index, first_true : last_true + 1] = values[2 : 2 + count]
    return matrix


def read_efficiency(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\{([^{}]+)\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Efficiency vector not found in {path}")
    return np.asarray([float(value.strip()) for value in match.group(1).split(",")], dtype=float)


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
    parser.add_argument("--energy-max", type=float, default=8.0, help="Maximum displayed energy in GeV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    glb_path = GLOBES_DIR / "DUNE_GLoBES.glb"
    true_edges = bin_edges(parse_globes_vector(glb_path, "sampling_stepsize"))
    reco_edges = bin_edges(parse_globes_vector(glb_path, "binsize"))

    names = [name for _, group in MATRIX_GROUPS for name in group]
    matrices = {
        name: read_smearing(GLOBES_DIR / "smr" / f"{name}.txt", len(true_edges) - 1)
        for name in names
    }

    positive = np.concatenate([matrix[matrix > 0.0] for matrix in matrices.values()])
    norm = LogNorm(vmin=max(1.0e-5, float(np.percentile(positive, 2.0))), vmax=1.0)

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
            ax.grid(False)
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
        ax.set_ylabel(r"Reconstructed energy $E_\mathrm{rec}$ (GeV)")

    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.07, top=0.97, wspace=0.12, hspace=0.22)
    colorbar_ax = fig.add_axes((0.93, 0.15, 0.018, 0.72))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"Migration weight $M(E_\mathrm{rec}, E_\mathrm{true})$")

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

    reco_centers = 0.5 * (reco_edges[:-1] + reco_edges[1:])
    energy_mask = (reco_centers >= 0.5) & (reco_centers <= args.energy_max)
    efficiency_files = sorted((GLOBES_DIR / "eff").glob("post_*.txt"))
    efficiency_values = []
    print(f"FD post-smearing efficiencies over 0.5-{args.energy_max:g} GeV:")
    for path in efficiency_files:
        values = read_efficiency(path)
        selected = values[: len(reco_centers)][energy_mask[: len(values)]]
        efficiency_values.append(selected)
        print(
            f"{path.stem}: min={selected.min():.6g}, "
            f"max={selected.max():.6g}, mean={selected.mean():.6g}"
        )
    all_efficiencies = np.concatenate(efficiency_values)
    print(
        f"All FD channels: min={all_efficiencies.min():.6g}, "
        f"max={all_efficiencies.max():.6g}, mean={all_efficiencies.mean():.6g}"
    )


if __name__ == "__main__":
    main()
