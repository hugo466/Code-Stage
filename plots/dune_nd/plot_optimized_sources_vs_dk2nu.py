#!/usr/bin/env python
"""Fit point/uniform ND source models to the dk2nu source-averaged curve.

The fitted observable is the Fig.4-like total ratio N_ISS/N_3nu in each
panel.  Candidate spectra are generated with the C code, so the event
calculation, active-3nu benchmark, component definitions, fluxes and xsecs stay
identical to the production pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_nd_source_model_comparison import (
    DEFAULT_BASE,
    DEFAULT_INPUTS,
    MODEL_COLORS,
    MODEL_STYLES,
    PANEL_TITLES,
    PANELS,
    read_inputs,
    total_panel,
)


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "bin" / "app.exe"
TMP_CONFIG_DIR = ROOT / "config" / "presets" / "dune" / "nd" / "_fit_tmp"
TMP_DATA_DIR = DEFAULT_BASE / "_source_fit_tmp"

DEFAULT_OUT = Path("figures/dune_nd/iss23/construct23_point70/source_models/source_scan_best_vs_dk2nu_refit.png")
DEFAULT_OLD_OUT = Path("figures/dune_nd/iss23/construct23_point70/source_models/source_scan_best_vs_dk2nu_old_refit.png")
DEFAULT_CSV = DEFAULT_BASE / "source_scan_best_vs_dk2nu_old_refit.csv"
DEFAULT_SCORE_CSV = DEFAULT_BASE / "source_scan_best_vs_dk2nu_old_refit_scores.csv"

DETECTOR_DISTANCE_M = 574.0
POINT_COARSE = np.arange(450.0, 701.0, 5.0)
UNIFORM_Z_COARSE_STEP_M = 10.0
UNIFORM_Z_REFINE_HALF_WIDTH_M = 8.0
UNIFORM_Z_REFINE_STEP_M = 1.0
REFINE_HALF_WIDTH_M = 6.0
REFINE_STEP_M = 0.5
UNIFORM_Z_ALLOWED_MIN_M = 0.0
UNIFORM_Z_ALLOWED_MAX_M = 194.0
UNIFORM_Z_BINS = 80


def ensure_app() -> None:
    if APP.exists():
        return
    subprocess.run(["mingw32-make", "build"], cwd=ROOT, check=True)


def write_config(
    model: str,
    detector_distance_m: float,
    out_csv: Path,
    *,
    z_min_m: float = UNIFORM_Z_ALLOWED_MIN_M,
    z_max_m: float = UNIFORM_Z_ALLOWED_MAX_M,
) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rel_out = out_csv.resolve().relative_to(ROOT).as_posix()
    if model == "uniform":
        cfg = TMP_CONFIG_DIR / f"{model}_z{z_min_m:.3f}_{z_max_m:.3f}.ini"
    else:
        cfg = TMP_CONFIG_DIR / f"{model}_L{detector_distance_m:.3f}.ini"
    source_z_bins = 1 if model == "point" else UNIFORM_Z_BINS
    baseline_model = "fixed" if model == "point" else "source_line"
    source_model = "point" if model == "point" else "uniform"
    text = f"""include = ../fig4_point70_source_line.ini

[beam]
baseline_model = {baseline_model}
source_model = {source_model}
detector_distance_m = {detector_distance_m:.10g}
source_z_start_m = {z_min_m:.10g}
decay_pipe_length_m = {z_max_m - z_min_m:.10g}
source_z_bins = {source_z_bins}

[output]
spectrum_pred_csv = {rel_out}
"""
    cfg.write_text(text, encoding="utf-8")
    return cfg


def run_candidate(
    model: str,
    detector_distance_m: float,
    *,
    z_min_m: float = UNIFORM_Z_ALLOWED_MIN_M,
    z_max_m: float = UNIFORM_Z_ALLOWED_MAX_M,
) -> pd.DataFrame:
    TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if model == "uniform":
        out_csv = TMP_DATA_DIR / f"{model}_z{z_min_m:.3f}_{z_max_m:.3f}.csv"
    else:
        out_csv = TMP_DATA_DIR / f"{model}_L{detector_distance_m:.3f}.csv"
    if not out_csv.exists():
        cfg = write_config(model, detector_distance_m, out_csv, z_min_m=z_min_m, z_max_m=z_max_m)
        subprocess.run([str(APP), str(cfg)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(out_csv)


def target_curves(dk2nu_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {panel: total_panel(dk2nu_df, panel) for panel in PANELS}


def score_frame(candidate: pd.DataFrame, target: dict[str, pd.DataFrame]) -> float:
    pieces: list[np.ndarray] = []
    for panel in PANELS:
        cand = total_panel(candidate, panel)[["Erec_GeV", "ratio_iss_over_3nu"]]
        ref = target[panel][["Erec_GeV", "ratio_iss_over_3nu"]]
        merged = cand.merge(ref, on="Erec_GeV", suffixes=("_cand", "_dk2nu"))
        if merged.empty:
            continue
        delta = merged["ratio_iss_over_3nu_cand"].to_numpy(dtype=float) - merged["ratio_iss_over_3nu_dk2nu"].to_numpy(dtype=float)
        pieces.append(delta)
    if not pieces:
        return float("inf")
    residuals = np.concatenate(pieces)
    return float(np.sqrt(np.mean(residuals * residuals)))


def scan_model(model: str, grid: np.ndarray, target: dict[str, pd.DataFrame]) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_l = None
    best_score = float("inf")
    best_frame = None
    for distance in grid:
        frame = run_candidate(model, float(distance))
        score = score_frame(frame, target)
        rows.append({"model": model, "detector_distance_m": float(distance), "score_rms_ratio": score})
        if score < best_score:
            best_l = float(distance)
            best_score = score
            best_frame = frame

    if best_l is None or best_frame is None:
        raise RuntimeError(f"Aucun candidat valide pour {model}")

    refine = np.arange(
        best_l - REFINE_HALF_WIDTH_M,
        best_l + REFINE_HALF_WIDTH_M + 0.5 * REFINE_STEP_M,
        REFINE_STEP_M,
    )
    for distance in refine:
        frame = run_candidate(model, float(distance))
        score = score_frame(frame, target)
        rows.append({"model": model, "detector_distance_m": float(distance), "score_rms_ratio": score})
        if score < best_score:
            best_l = float(distance)
            best_score = score
            best_frame = frame

    return best_l, best_frame, pd.DataFrame(rows)


def valid_uniform_bounds(z_min_m: float, z_max_m: float) -> bool:
    return (
        UNIFORM_Z_ALLOWED_MIN_M <= z_min_m < z_max_m <= UNIFORM_Z_ALLOWED_MAX_M
        and z_max_m - z_min_m >= 1.0
    )


def uniform_coarse_grid() -> list[tuple[float, float]]:
    values = np.arange(
        UNIFORM_Z_ALLOWED_MIN_M,
        UNIFORM_Z_ALLOWED_MAX_M + 0.5 * UNIFORM_Z_COARSE_STEP_M,
        UNIFORM_Z_COARSE_STEP_M,
    )
    values = np.unique(np.append(values, UNIFORM_Z_ALLOWED_MAX_M))
    pairs: list[tuple[float, float]] = []
    for z_min in values:
        for z_max in values:
            if valid_uniform_bounds(float(z_min), float(z_max)):
                pairs.append((float(z_min), float(z_max)))
    return pairs


def uniform_refine_grid(z_min_best: float, z_max_best: float) -> list[tuple[float, float]]:
    z_min_values = np.arange(
        max(UNIFORM_Z_ALLOWED_MIN_M, z_min_best - UNIFORM_Z_REFINE_HALF_WIDTH_M),
        min(UNIFORM_Z_ALLOWED_MAX_M, z_min_best + UNIFORM_Z_REFINE_HALF_WIDTH_M) + 0.5 * UNIFORM_Z_REFINE_STEP_M,
        UNIFORM_Z_REFINE_STEP_M,
    )
    z_max_values = np.arange(
        max(UNIFORM_Z_ALLOWED_MIN_M, z_max_best - UNIFORM_Z_REFINE_HALF_WIDTH_M),
        min(UNIFORM_Z_ALLOWED_MAX_M, z_max_best + UNIFORM_Z_REFINE_HALF_WIDTH_M) + 0.5 * UNIFORM_Z_REFINE_STEP_M,
        UNIFORM_Z_REFINE_STEP_M,
    )
    pairs: list[tuple[float, float]] = []
    for z_min in z_min_values:
        for z_max in z_max_values:
            if valid_uniform_bounds(float(z_min), float(z_max)):
                pairs.append((float(z_min), float(z_max)))
    return pairs


def scan_uniform_bounds(target: dict[str, pd.DataFrame]) -> tuple[tuple[float, float], pd.DataFrame, pd.DataFrame]:
    rows = []
    best_bounds: tuple[float, float] | None = None
    best_score = float("inf")
    best_frame = None

    for z_min, z_max in uniform_coarse_grid():
        frame = run_candidate("uniform", DETECTOR_DISTANCE_M, z_min_m=z_min, z_max_m=z_max)
        score = score_frame(frame, target)
        rows.append(
            {
                "model": "uniform",
                "detector_distance_m": DETECTOR_DISTANCE_M,
                "z_min_m": z_min,
                "z_max_m": z_max,
                "score_rms_ratio": score,
            }
        )
        if score < best_score:
            best_bounds = (z_min, z_max)
            best_score = score
            best_frame = frame

    if best_bounds is None or best_frame is None:
        raise RuntimeError("Aucun candidat uniforme valide")

    for z_min, z_max in uniform_refine_grid(*best_bounds):
        frame = run_candidate("uniform", DETECTOR_DISTANCE_M, z_min_m=z_min, z_max_m=z_max)
        score = score_frame(frame, target)
        rows.append(
            {
                "model": "uniform",
                "detector_distance_m": DETECTOR_DISTANCE_M,
                "z_min_m": z_min,
                "z_max_m": z_max,
                "score_rms_ratio": score,
            }
        )
        if score < best_score:
            best_bounds = (z_min, z_max)
            best_score = score
            best_frame = frame

    return best_bounds, best_frame, pd.DataFrame(rows)


def labels(point_l: float, uniform_bounds: tuple[float, float]) -> dict[str, str]:
    z_min, z_max = uniform_bounds
    return {
        "point": f"source ponctuelle fit (L={point_l:.1f} m)",
        "uniform": f"source uniforme fit (L=574 m, z=[{z_min:.0f},{z_max:.0f}] m)",
        "dk2nu": "poids dk2nu",
    }


def plot_curves(frames: dict[str, pd.DataFrame], labels_by_model: dict[str, str], out: Path) -> None:
    curves = {model: {panel: total_panel(df, panel) for panel in PANELS} for model, df in frames.items()}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)

    styles = {
        "point": dict(color=MODEL_COLORS["point"], linestyle=MODEL_STYLES["point"], linewidth=2.0),
        "uniform": dict(color=MODEL_COLORS["uniform"], linestyle=MODEL_STYLES["uniform"], linewidth=1.9),
        "dk2nu": dict(color=MODEL_COLORS["dk2nu"], linestyle=MODEL_STYLES["dk2nu"], linewidth=1.9),
    }

    for ax, panel in zip(axes.flat, PANELS):
        for model in ["point", "uniform", "dk2nu"]:
            data = curves[model][panel]
            ax.step(data["Erec_GeV"], data["ratio_iss_over_3nu"], where="mid", label=labels_by_model[model], **styles[model])
        ax.set_title(PANEL_TITLES[panel], fontsize=10)
        ax.set_xlim(0.5, 8.0)
        ax.set_ylabel(r"$N_{\rm ISS}/N_{3\nu}$")
        ax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Energie reconstruite [GeV]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_curve_csv(frames: dict[str, pd.DataFrame], labels_by_model: dict[str, str], out: Path) -> None:
    rows = []
    for model, df in frames.items():
        for panel in PANELS:
            data = total_panel(df, panel)
            for _, row in data.iterrows():
                rows.append(
                    {
                        "model": model,
                        "label": labels_by_model[model],
                        "panel": panel,
                        "Erec_GeV": float(row["Erec_GeV"]),
                        "ratio_iss_over_3nu": float(row["ratio_iss_over_3nu"]),
                    }
                )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit point/uniform ND source models to dk2nu.")
    parser.add_argument("--dk2nu", type=Path, default=DEFAULT_INPUTS["dk2nu"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-out", type=Path, default=DEFAULT_OLD_OUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_app()
    dk2nu = read_inputs({"dk2nu": args.dk2nu})["dk2nu"]
    target = target_curves(dk2nu)

    point_l, point_frame, point_scores = scan_model("point", POINT_COARSE, target)
    uniform_bounds, uniform_frame, uniform_scores = scan_uniform_bounds(target)
    label_map = labels(point_l, uniform_bounds)
    frames = {"point": point_frame, "uniform": uniform_frame, "dk2nu": dk2nu}

    point_scores["z_min_m"] = np.nan
    point_scores["z_max_m"] = np.nan
    scores = pd.concat([point_scores, uniform_scores], ignore_index=True)
    args.score_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.score_csv, index=False)
    write_curve_csv(frames, label_map, args.out_csv)
    plot_curves(frames, label_map, args.out)
    if args.old_out != args.out:
        plot_curves(frames, label_map, args.old_out)

    print(f"best point source L = {point_l:.3f} m")
    print(f"best uniform detector distance = {DETECTOR_DISTANCE_M:.3f} m")
    print(f"best uniform source interval = [{uniform_bounds[0]:.3f}, {uniform_bounds[1]:.3f}] m")
    print(f"CSV sauvegarde: {args.out_csv.resolve()}")
    print(f"Scores sauvegardes: {args.score_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")
    if args.old_out != args.out:
        print(f"Figure legacy sauvegardee: {args.old_out.resolve()}")


if __name__ == "__main__":
    main()
