#!/usr/bin/env python
"""Plot the ND point/uniform/dk2nu source-model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


DEFAULT_BASE = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation")
DEFAULT_INPUTS = {
    "point": DEFAULT_BASE / "fig4_nd_point_source_iss23_vs_active3nu.csv",
    "uniform": DEFAULT_BASE / "fig4_nd_source_line_iss23_vs_active3nu.csv",
    "dk2nu": DEFAULT_BASE / "fig4_nd_dk2nu_iss23_vs_active3nu.csv",
}
DEFAULT_OUT = Path("figures/dune_nd/iss23/construct23_point70/source_models/nd_source_model_comparison.png")

PANELS = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
PANEL_COMPONENTS = {
    "FHC_app": ["nc", "numu", "beam", "signal"],
    "RHC_app": ["nc", "numu", "beam", "signal"],
    "FHC_dis": ["nc", "wrong_mu", "tau", "signal"],
    "RHC_dis": ["nc", "wrong_mu", "tau", "signal"],
}
PANEL_TITLES = {
    "FHC_app": r"FHC $\nu_e$ apparition",
    "RHC_app": r"RHC $\bar{\nu}_e$ apparition",
    "FHC_dis": r"FHC $\nu_\mu$ disparition",
    "RHC_dis": r"RHC $\bar{\nu}_\mu$ disparition",
}
MODEL_LABELS = {
    "point": "source ponctuelle",
    "uniform": "source uniforme",
    "dk2nu": "poids dk2nu",
}
MODEL_COLORS = {
    "point": "#6b6b6b",
    "uniform": "#1f77b4",
    "dk2nu": "#d62728",
}
MODEL_STYLES = {
    "point": ":",
    "uniform": "--",
    "dk2nu": "-",
}


def labels_from_frames(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    labels = dict(MODEL_LABELS)

    point = frames.get("point")
    if point is not None and not point.empty and "baseline_km" in point.columns:
        labels["point"] = f"source ponctuelle (L={1000.0 * float(point['baseline_km'].iloc[0]):.0f} m)"

    uniform = frames.get("uniform")
    if (
        uniform is not None
        and not uniform.empty
        and {"source_z_start_m", "decay_pipe_length_m"}.issubset(uniform.columns)
    ):
        z_min = float(uniform["source_z_start_m"].iloc[0])
        z_max = z_min + float(uniform["decay_pipe_length_m"].iloc[0])
        labels["uniform"] = (
            "source uniforme "
            f"(z=[{z_min:.0f},{z_max:.0f}] m)"
        )
    return labels


def read_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"CSV manquant pour {model}: {path}")
        df = pd.read_csv(path)
        required = {"panel", "component", "Erec_GeV", "globes_events", "iss23_events"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path}: colonnes manquantes: {sorted(missing)}")
        frames[model] = df
    return frames


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


def padded_limits(values: list[float], min_span: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(hi - lo, min_span)
    pad = 0.18 * span
    return lo - pad, hi + pad


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
    ax.set_xlim(0.5, 8.0)
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

    summaries: dict[str, dict[str, float]] = {}
    for panel, ax in axes_by_panel.items():
        summaries[panel] = draw_panel(ax, panel, frames, labels, show_xlabel=panel.endswith("_dis"))

    axes_by_panel["FHC_app"].legend(loc="best", fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.97, hspace=0.24, wspace=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print("Max |relative difference in N_ISS vs dk2nu| per bin:")
    for panel, values in summaries.items():
        print(
            f"  {panel}: point={values['point']:.6g}% "
            f"uniform={values['uniform']:.6g}% dk2nu={values['dk2nu']:.6g}%"
        )
    print(f"Figure sauvegardee: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ND comparaison modeles de sources")
    parser.add_argument("--point", type=Path, default=DEFAULT_INPUTS["point"])
    parser.add_argument("--uniform", type=Path, default=DEFAULT_INPUTS["uniform"])
    parser.add_argument("--dk2nu", type=Path, default=DEFAULT_INPUTS["dk2nu"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = read_inputs({"point": args.point, "uniform": args.uniform, "dk2nu": args.dk2nu})
    plot_comparison(frames, args.out)


if __name__ == "__main__":
    main()
