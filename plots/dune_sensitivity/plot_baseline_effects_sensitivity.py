from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


DATA_DIR = Path("data/dune/sensitivity")
DEFAULT_OUTPUT = Path("figures/dune_sensitivity/baseline_effects_analytic_3p1_point_maps.png")

DEFAULT_MAPS = [
    (
        DATA_DIR / "baseline_effects_analytic_3p1_theta14_uniform_dense_noshape.csv",
        r"$\sin^2\theta_{14}$",
        "sin2_theta14_eff",
        "Source uniforme",
    ),
    (
        DATA_DIR / "baseline_effects_analytic_3p1_theta14_point_dense_noshape.csv",
        r"$\sin^2\theta_{14}$",
        "sin2_theta14_eff",
        "Source ponctuelle",
    ),
    (
        DATA_DIR / "baseline_effects_analytic_3p1_theta24_uniform_dense_noshape.csv",
        r"$\sin^2\theta_{24}$",
        "sin2_theta24_eff",
        "Source uniforme",
    ),
    (
        DATA_DIR / "baseline_effects_analytic_3p1_theta24_point_dense_noshape.csv",
        r"$\sin^2\theta_{24}$",
        "sin2_theta24_eff",
        "Source ponctuelle",
    ),
]


def _finite_positive(df: pd.DataFrame, column: str, floor: float = 1.0e-30) -> np.ndarray:
    values = df[column].to_numpy(dtype=float)
    return np.where(np.isfinite(values) & (values > floor), values, np.nan)


def _load_map(path: Path, x_column: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"No rows in {path}")
    df = df.copy()
    df["_x"] = _finite_positive(df, x_column)
    df["_y"] = _finite_positive(df, "dm41_eV2", 1.0e-12)
    df["_chi2"] = df["chi2_min"].to_numpy(dtype=float)
    return df[np.isfinite(df["_x"]) & np.isfinite(df["_y"]) & np.isfinite(df["_chi2"]) & (df["_chi2"] > 0.0)], x_column


def _log_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(sorted(np.unique(centers)), dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two grid centers")
    logc = np.log10(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (logc[:-1] + logc[1:])
    edges[0] = logc[0] - 0.5 * (logc[1] - logc[0])
    edges[-1] = logc[-1] + 0.5 * (logc[-1] - logc[-2])
    return 10.0 ** edges


def _grid_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(sorted(df["_x"].unique()), dtype=float)
    ys = np.asarray(sorted(df["_y"].unique()), dtype=float)
    pivot = df.pivot_table(index="_y", columns="_x", values="_chi2", aggfunc="median")
    pivot = pivot.reindex(index=ys, columns=xs)
    return _log_edges(xs), _log_edges(ys), pivot.to_numpy(dtype=float)


def plot_point_maps(output_png: Path) -> None:
    loaded = []
    chi2_all = []
    for path, xlabel, x_column, title in DEFAULT_MAPS:
        df, _ = _load_map(path, x_column)
        x_edges, y_edges, z = _grid_from_df(df)
        loaded.append((x_edges, y_edges, z, xlabel, title))
        valid = z[np.isfinite(z) & (z > 0.0)]
        if len(valid):
            chi2_all.append(valid)

    if not chi2_all:
        raise RuntimeError("No valid chi2 values found")

    all_chi2 = np.concatenate(chi2_all)
    vmin = max(1.0e-6, float(np.nanpercentile(all_chi2, 1.0)))
    vmax = max(vmin * 10.0, float(np.nanpercentile(all_chi2, 99.0)))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.6), constrained_layout=True)
    axes = axes.ravel()
    mesh = None
    for ax, (x_edges, y_edges, z, xlabel, title) in zip(axes, loaded):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            z,
            cmap="viridis",
            norm=norm,
            shading="auto",
            linewidth=0.0,
            rasterized=True,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.94)
        cbar.set_label(r"$\chi^2_\mathrm{min}$")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DUNE sensitivity point maps from dense scan CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_point_maps(args.output)


if __name__ == "__main__":
    main()
