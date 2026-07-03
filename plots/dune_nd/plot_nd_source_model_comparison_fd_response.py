#!/usr/bin/env python
"""ND source-model comparison after applying the FD GLoBES response.

The input CSVs are the same as for plot_nd_source_model_comparison.py.  They
already contain event spectra binned at 0.25 GeV for each Fig.4 component.  This
script applies the FD migration matrix and post-smearing efficiency associated
with each component, then rebuilds the same N_ISS/N_3nu comparison plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots" / "dune_fd"))
sys.path.insert(0, str(ROOT / "plots" / "dune_nd"))

from plot_fig4_reconstructed_energy import BASE as FD_GLOBES_BASE  # noqa: E402
from plot_fig4_reconstructed_energy import CHANNELS, read_post_eff, read_smearing  # noqa: E402
from plot_nd_source_model_comparison import (  # noqa: E402
    DEFAULT_INPUTS,
    MODEL_COLORS,
    MODEL_STYLES,
    PANEL_COMPONENTS,
    PANEL_TITLES,
    labels_from_frames,
    padded_limits,
    read_inputs,
)


DEFAULT_OUT = ROOT / "figures" / "dune_nd" / "iss23" / "construct23_point70" / "source_models" / "nd_source_model_comparison_fd_response.png"
DEFAULT_CSV = ROOT / "data" / "dune_nd" / "minimal_onaxis" / "point_70" / "plots_validation" / "nd_source_model_comparison_fd_response.csv"

PANELS = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
E_MIN = 0.5
E_MAX = 8.0
BIN_WIDTH = 0.25


def aggregate_0125_to_025(values: np.ndarray, n_bins_025: int) -> np.ndarray:
    n = min(2 * n_bins_025, len(values))
    trimmed = np.asarray(values[:n], dtype=float)
    if len(trimmed) % 2:
        trimmed = trimmed[:-1]
    out = trimmed.reshape(-1, 2).sum(axis=1)
    if len(out) < n_bins_025:
        out = np.pad(out, (0, n_bins_025 - len(out)))
    return out[:n_bins_025]


def input_025_to_true0125(values: np.ndarray, n_true: int) -> np.ndarray:
    true_counts = np.zeros(n_true, dtype=float)
    for i, value in enumerate(np.asarray(values, dtype=float)):
        lo = 4 + 2 * i  # 0.5 GeV starts at the fifth 0.125-GeV bin.
        if lo < n_true:
            true_counts[lo] += 0.5 * value
        if lo + 1 < n_true:
            true_counts[lo + 1] += 0.5 * value
    return true_counts


def component_response_matrix(panel: str, component: str) -> np.ndarray:
    channels = CHANNELS[panel].get(component, [])
    matrices = []
    for channel in channels:
        smear = read_smearing(FD_GLOBES_BASE / "smr" / f"{channel.smear}.txt")
        eff = read_post_eff(FD_GLOBES_BASE / "eff" / f"{channel.eff}.txt")
        n_reco = min(smear.shape[0], len(eff))
        matrices.append(smear[:n_reco, :] * eff[:n_reco, None])
    if not matrices:
        raise ValueError(f"Pas de matrice FD pour {panel}/{component}")
    min_reco = min(matrix.shape[0] for matrix in matrices)
    min_true = min(matrix.shape[1] for matrix in matrices)
    stack = np.stack([matrix[:min_reco, :min_true] for matrix in matrices], axis=0)
    return np.mean(stack, axis=0)


def apply_fd_response_to_component(df: pd.DataFrame, panel: str, component: str) -> pd.DataFrame:
    selected = df[(df["panel"] == panel) & (df["component"] == component)].sort_values("Erec_GeV")
    if selected.empty:
        return selected.copy()

    response = component_response_matrix(panel, component)
    n_bins = len(selected)
    active_true = input_025_to_true0125(selected["globes_events"].to_numpy(dtype=float), response.shape[1])
    iss_true = input_025_to_true0125(selected["iss23_events"].to_numpy(dtype=float), response.shape[1])

    active_reco = aggregate_0125_to_025(response @ active_true, n_bins)
    iss_reco = aggregate_0125_to_025(response @ iss_true, n_bins)

    out = selected.copy()
    out["globes_events"] = active_reco
    out["iss23_events"] = iss_reco
    out["ratio_iss_over_3nu"] = np.divide(
        iss_reco,
        active_reco,
        out=np.full(n_bins, np.nan, dtype=float),
        where=np.abs(active_reco) > 1.0e-30,
    )
    out["rel_diff"] = out["ratio_iss_over_3nu"] - 1.0
    out["delta_events"] = iss_reco - active_reco
    out["fd_response_applied"] = 1
    return out


def apply_fd_response(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for panel in PANELS:
        for component in PANEL_COMPONENTS[panel]:
            if component == "tau" and df[(df["panel"] == panel) & (df["component"] == component)].empty:
                continue
            frames.append(apply_fd_response_to_component(df, panel, component))
    return pd.concat(frames, ignore_index=True)


def total_panel(df: pd.DataFrame, panel: str) -> pd.DataFrame:
    selected = df[(df["panel"] == panel) & (df["component"].isin(PANEL_COMPONENTS[panel]))]
    grouped = (
        selected.groupby("Erec_GeV", as_index=False)[["globes_events", "iss23_events"]]
        .sum()
        .sort_values("Erec_GeV")
    )
    grouped["ratio_iss_over_3nu"] = np.divide(
        grouped["iss23_events"],
        grouped["globes_events"],
        out=np.full(len(grouped), np.nan, dtype=float),
        where=grouped["globes_events"].to_numpy(dtype=float) > 0.0,
    )
    return grouped


def step(ax, data: pd.DataFrame, y_col: str, model: str, label: str | None = None) -> None:
    ax.step(
        data["Erec_GeV"],
        data[y_col],
        where="mid",
        color=MODEL_COLORS[model],
        linestyle=MODEL_STYLES[model],
        linewidth=1.7,
        label=label,
    )


def draw_panel(ax, panel: str, frames: dict[str, pd.DataFrame], labels: dict[str, str], show_xlabel: bool) -> dict[str, float]:
    totals = {model: total_panel(df, panel) for model, df in frames.items()}
    ratio_values: list[float] = []
    summary: dict[str, float] = {}

    dk2nu = totals["dk2nu"][["Erec_GeV", "iss23_events"]].rename(columns={"iss23_events": "dk2nu_iss"})
    for model in ["point", "uniform", "dk2nu"]:
        data = totals[model]
        ratio_values.extend(data["ratio_iss_over_3nu"].to_numpy(dtype=float).tolist())
        step(ax, data, "ratio_iss_over_3nu", model, labels[model])

        merged = data.merge(dk2nu, on="Erec_GeV", how="inner")
        rel = np.divide(
            merged["iss23_events"].to_numpy(dtype=float) - merged["dk2nu_iss"].to_numpy(dtype=float),
            merged["dk2nu_iss"].to_numpy(dtype=float),
            out=np.full(len(merged), np.nan, dtype=float),
            where=np.abs(merged["dk2nu_iss"].to_numpy(dtype=float)) > 1e-12,
        )
        summary[model] = float(100.0 * np.nanmax(np.abs(rel)))

    ax.set_title(PANEL_TITLES[panel], fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(*padded_limits(ratio_values, min_span=2.0e-4))
    ax.set_ylabel(r"$N_{ISS}/N_{3\nu}$")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    if show_xlabel:
        ax.set_xlabel("Energie reconstruite [GeV]")
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.grid(alpha=0.25)
    return summary


def plot_comparison(frames: dict[str, pd.DataFrame], out: Path) -> None:
    labels = labels_from_frames(frames)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes_by_panel = dict(zip(PANELS, axes.flat))

    summaries = {}
    for panel, ax in axes_by_panel.items():
        summaries[panel] = draw_panel(ax, panel, frames, labels, show_xlabel=panel.endswith("_dis"))

    axes_by_panel["FHC_app"].legend(loc="best", fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.97, hspace=0.24, wspace=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print("Max |relative difference in N_ISS vs dk2nu| per bin after FD response:")
    for panel, values in summaries.items():
        print(
            f"  {panel}: point={values['point']:.6g}% "
            f"uniform={values['uniform']:.6g}% dk2nu={values['dk2nu']:.6g}%"
        )
    print(f"Figure sauvegardee: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ND source model comparison with FD response applied.")
    parser.add_argument("--point", type=Path, default=DEFAULT_INPUTS["point"])
    parser.add_argument("--uniform", type=Path, default=DEFAULT_INPUTS["uniform"])
    parser.add_argument("--dk2nu", type=Path, default=DEFAULT_INPUTS["dk2nu"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_frames = read_inputs({"point": args.point, "uniform": args.uniform, "dk2nu": args.dk2nu})
    frames = {model: apply_fd_response(frame) for model, frame in raw_frames.items()}
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df.assign(model=model) for model, df in frames.items()], ignore_index=True).to_csv(args.out_csv, index=False)
    plot_comparison(frames, args.out)
    print(f"CSV sauvegarde: {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
