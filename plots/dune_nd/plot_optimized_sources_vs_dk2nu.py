#!/usr/bin/env python
"""Plot optimized point/line source models against dk2nu.

Curves:
- dotted: optimized point source, fixed L_eff = 525 m
- blue: optimized line source, uniform z in [0, 120] m
- red: dk2nu weighted source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("data/dune_nd/minimal_onaxis/point_70")
DEFAULT_POINT = BASE / "point_source_length_scan/point_L0525.00.csv"
DEFAULT_LINE = BASE / "source_model_scan/uniform_z000.00_l120.00.csv"
DEFAULT_DK2NU = BASE / "plots_validation/fig4_nd_dk2nu_iss23_vs_active3nu.csv"
DEFAULT_OUT = Path("figures/dune_nd/point_70/source_models/source_scan_best_vs_dk2nu.png")

PANELS = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
PANEL_COMPONENTS = {
    "FHC_app": ["nc", "numu", "beam", "signal"],
    "RHC_app": ["nc", "numu", "beam", "signal"],
    "FHC_dis": ["nc", "wrong_mu", "tau", "signal"],
    "RHC_dis": ["nc", "wrong_mu", "tau", "signal"],
}
PANEL_TITLES = {
    "FHC_app": r"FHC $\nu_e$ app.",
    "RHC_app": r"RHC $\bar{\nu}_e$ app.",
    "FHC_dis": r"FHC $\nu_\mu$ dis.",
    "RHC_dis": r"RHC $\bar{\nu}_\mu$ dis.",
}


def total_ratio(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"panel", "component", "Erec_GeV", "globes_events", "iss23_events"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: colonnes manquantes: {sorted(missing)}")

    out: dict[str, pd.DataFrame] = {}
    for panel in PANELS:
        selected = df[(df["panel"] == panel) & (df["component"].isin(PANEL_COMPONENTS[panel]))]
        grouped = (
            selected.groupby("Erec_GeV", as_index=False)[["globes_events", "iss23_events"]]
            .sum()
            .sort_values("Erec_GeV")
        )
        grouped["ratio"] = np.divide(
            grouped["iss23_events"].to_numpy(dtype=float),
            grouped["globes_events"].to_numpy(dtype=float),
            out=np.full(len(grouped), np.nan, dtype=float),
            where=grouped["globes_events"].to_numpy(dtype=float) > 0.0,
        )
        out[panel] = grouped
    return out


def max_abs_delta(candidate: pd.DataFrame, reference: pd.DataFrame, emin: float) -> float:
    merged = candidate[["Erec_GeV", "ratio"]].merge(
        reference[["Erec_GeV", "ratio"]],
        on="Erec_GeV",
        suffixes=("_candidate", "_reference"),
    )
    merged = merged[merged["Erec_GeV"] > emin]
    if merged.empty:
        return float("nan")
    return float(np.nanmax(np.abs(merged["ratio_candidate"] - merged["ratio_reference"])))


def plot(point: Path, line: Path, dk2nu: Path, out: Path, emin_score: float) -> None:
    curves = {
        "point source opt. L=525 m": total_ratio(point),
        "line source opt. [0,120] m": total_ratio(line),
        "dk2nu weighted": total_ratio(dk2nu),
    }
    styles = {
        "point source opt. L=525 m": dict(color="0.35", linestyle=":", linewidth=2.0),
        "line source opt. [0,120] m": dict(color="#1f77b4", linestyle="-", linewidth=1.9),
        "dk2nu weighted": dict(color="#d62728", linestyle="-", linewidth=1.9),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    for ax, panel in zip(axes.flat, PANELS):
        ref = curves["dk2nu weighted"][panel]
        for label, panel_curves in curves.items():
            data = panel_curves[panel]
            ax.step(data["Erec_GeV"], data["ratio"], where="mid", label=label, **styles[label])

        point_delta = max_abs_delta(curves["point source opt. L=525 m"][panel], ref, emin_score)
        line_delta = max_abs_delta(curves["line source opt. [0,120] m"][panel], ref, emin_score)
        ax.axvspan(emin_score, 8.0, color="0.92", alpha=0.55, zorder=-10)
        ax.set_title(
            f"{PANEL_TITLES[panel]}",#\n"
            # f"E>{emin_score:g} GeV: max |Delta| point={point_delta:.2e}, line={line_delta:.2e}",
            fontsize=9.5,
        )
        ax.set_xlim(0.5, 8.0)
        ax.set_ylabel(r"$N_{ISS}/N_{3\nu}$")
        ax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Energie reconstruite [GeV]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Modeles de source optimisés sur dk2nu",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print(f"Figure sauvegardee: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimized point/line source models vs dk2nu.")
    parser.add_argument("--point", type=Path, default=DEFAULT_POINT)
    parser.add_argument("--line", type=Path, default=DEFAULT_LINE)
    parser.add_argument("--dk2nu", type=Path, default=DEFAULT_DK2NU)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--emin-score-GeV", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.point, args.line, args.dk2nu, args.out, args.emin_score_GeV)


if __name__ == "__main__":
    main()
