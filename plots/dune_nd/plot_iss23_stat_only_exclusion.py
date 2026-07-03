import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from plots.dune_nd.plot_iss23_scan_max_event_ratio_map import (
    AVOGADRO,
    EXPOSURE_YEARS,
    GLOBES,
    PANELS,
    POT_PER_YEAR,
    SOURCE_PROFILE_FHC,
    SOURCE_PROFILE_RHC,
    TARGET_MASS_KT,
    build_mixing,
    panel_components,
    read_dk2nu_flux_from_z,
    read_source_profile,
    read_xsec,
)


POINTS_CSV = ROOT / "data" / "inverse_seesaw" / "3p1" / "inverse_construct_23_kept_points" / "inverse_construct_23_kept_points.csv"
OUT_CSV = ROOT / "data" / "dune_nd" / "exclusion" / "iss23_stat_only_chi2.csv"
OUT_FIG = ROOT / "figures" / "dune_nd" / "iss23" / "exclusion" / "iss23_stat_only_exclusion.png"

CL_LEVELS_2DOF = {
    "90% CL": 4.61,
    "95% CL": 5.99,
}
M2_TO_CM2 = 1.0e4


def select_points(points_csv: Path, max_points: int = 0) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    for column in ("pmns_pass", "eta_pass"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    required = [
        "point_id",
        "dm21_calc_eV2",
        "dm31_calc_eV2",
        "dm41_calc_eV2",
        *[f"U_solver_{r}{c}" for r in range(1, 5) for c in range(1, 5)],
    ]
    df = df.dropna(subset=required)
    df = df[df["dm41_calc_eV2"] > 0.0].copy()
    if max_points > 0:
        df = df.head(max_points).copy()
    if df.empty:
        raise RuntimeError("Aucun point exploitable avec pmns_pass=1 et eta_pass=1.")
    df["Ue4_abs"] = df["U_solver_14"].abs()
    df["Umu4_abs"] = df["U_solver_24"].abs()
    df["Ue4Umu4_abs"] = df["Ue4_abs"] * df["Umu4_abs"]
    df["Umu4_sq"] = df["Umu4_abs"] ** 2
    return df


def compute_stat_only_chi2(df: pd.DataFrame, e_min: float, e_max: float, n_bins: int, epsilon: float) -> pd.DataFrame:
    u = build_mixing(df)
    masses4 = np.column_stack(
        [
            np.zeros(len(df), dtype=float),
            df["dm21_calc_eV2"].to_numpy(dtype=float),
            df["dm31_calc_eV2"].to_numpy(dtype=float),
            df["dm41_calc_eV2"].to_numpy(dtype=float),
        ]
    )
    masses3 = masses4[:, :3]
    energies = e_min + (np.arange(n_bins, dtype=float) + 0.5) * ((e_max - e_min) / n_bins)
    width = (e_max - e_min) / n_bins

    fluxes = {
        "FHC": read_dk2nu_flux_from_z(SOURCE_PROFILE_FHC),
        "RHC": read_dk2nu_flux_from_z(SOURCE_PROFILE_RHC),
    }
    profiles = {
        "FHC": read_source_profile(SOURCE_PROFILE_FHC),
        "RHC": read_source_profile(SOURCE_PROFILE_RHC),
    }
    cc = read_xsec(GLOBES / "xsec" / "xsec_cc.dat")
    nc = read_xsec(GLOBES / "xsec" / "xsec_nc.dat")
    scale = POT_PER_YEAR * EXPOSURE_YEARS * TARGET_MASS_KT * 1.0e9 * AVOGADRO

    chi2 = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}
    max_abs_rel = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}
    n3_totals = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}
    n4_totals = {panel: np.zeros(len(df), dtype=float) for panel in PANELS}

    for panel in PANELS:
        mode = "RHC" if panel.startswith("RHC") else "FHC"
        for energy in energies:
            comp3 = panel_components(panel, fluxes[mode], cc, nc, u, masses3, energy, width, profiles[mode], 3)
            comp4 = panel_components(panel, fluxes[mode], cc, nc, u, masses4, energy, width, profiles[mode], 4)
            n3 = sum(comp3.values()) * scale
            n4 = sum(comp4.values()) * scale
            diff = n4 - n3
            denom = n3 + epsilon
            chi2[panel] += np.divide(diff * diff, denom, out=np.zeros_like(diff), where=denom > 0.0)
            rel = np.divide(diff, n3, out=np.zeros_like(diff), where=np.abs(n3) > 1.0e-300)
            max_abs_rel[panel] = np.maximum(max_abs_rel[panel], np.abs(rel))
            n3_totals[panel] += n3
            n4_totals[panel] += n4

    out = df[
        [
            "point_id",
            "dm41_calc_eV2",
            "Ue4_abs",
            "Umu4_abs",
            "Ue4Umu4_abs",
            "Umu4_sq",
        ]
    ].copy()
    for panel in PANELS:
        out[f"chi2_{panel}"] = chi2[panel]
        out[f"{panel}_max_abs_rel"] = max_abs_rel[panel]
        out[f"{panel}_n3_total"] = n3_totals[panel]
        out[f"{panel}_n4_total"] = n4_totals[panel]
    out["chi2_appearance"] = out["chi2_FHC_app"] + out["chi2_RHC_app"]
    out["chi2_disappearance"] = out["chi2_FHC_dis"] + out["chi2_RHC_dis"]
    out["chi2_all"] = out["chi2_appearance"] + out["chi2_disappearance"]
    out["excluded_appearance_90cl_stat_only"] = out["chi2_appearance"] >= CL_LEVELS_2DOF["90% CL"]
    out["excluded_disappearance_90cl_stat_only"] = out["chi2_disappearance"] >= CL_LEVELS_2DOF["90% CL"]
    out["excluded_all_90cl_stat_only"] = out["chi2_all"] >= CL_LEVELS_2DOF["90% CL"]
    return out


def compute_stat_only_chi2_worker(payload: tuple[pd.DataFrame, float, float, int, float]) -> pd.DataFrame:
    df, e_min, e_max, n_bins, epsilon = payload
    return compute_stat_only_chi2(df, e_min, e_max, n_bins, epsilon)


def split_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    if chunk_size <= 0 or chunk_size >= len(df):
        return [df]
    return [df.iloc[start:start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def contour_if_possible(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, levels: list[float], n_bins: int = 30) -> None:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.0) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if len(x) < 10:
        return
    lx = np.log10(x)
    ly = np.log10(y)
    x_edges = np.linspace(float(np.nanmin(lx)), float(np.nanmax(lx)), n_bins + 1)
    y_edges = np.linspace(float(np.nanmin(ly)), float(np.nanmax(ly)), n_bins + 1)
    grid = np.full((n_bins, n_bins), np.nan, dtype=float)
    for ix in range(n_bins):
        x_mask = (lx >= x_edges[ix]) & (lx < x_edges[ix + 1])
        if not np.any(x_mask):
            continue
        for iy in range(n_bins):
            mask = x_mask & (ly >= y_edges[iy]) & (ly < y_edges[iy + 1])
            if np.count_nonzero(mask) >= 2:
                grid[iy, ix] = float(np.nanmedian(z[mask]))

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return
    kept_levels = [level for level in levels if float(np.nanmin(finite)) <= level <= float(np.nanmax(finite))]
    if not kept_levels:
        return
    x_centers = 10.0 ** (0.5 * (x_edges[:-1] + x_edges[1:]))
    y_centers = 10.0 ** (0.5 * (y_edges[:-1] + y_edges[1:]))
    xx, yy = np.meshgrid(x_centers, y_centers)
    masked_grid = np.ma.masked_invalid(grid)
    styles = ["-", "--", ":"]
    colors = ["black", "0.2", "0.35"]
    for idx, level in enumerate(kept_levels):
        ax.contour(
            xx,
            yy,
            masked_grid,
            levels=[level],
            colors=[colors[idx % len(colors)]],
            linestyles=[styles[idx % len(styles)]],
            linewidths=1.5,
        )


def draw_exclusion(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    specs = [
        (
            "chi2_appearance",
            "Ue4Umu4_abs",
            r"$|U_{e4}U_{\mu4}|$",
            r"Appearance: FHC $\nu_e$ + RHC $\bar{\nu}_e$",
        ),
        (
            "chi2_disappearance",
            "Umu4_sq",
            r"$|U_{\mu4}|^2$",
            r"Disappearance: FHC $\nu_\mu$ + RHC $\bar{\nu}_\mu$",
        ),
    ]
    levels = list(CL_LEVELS_2DOF.values())
    for ax, (chi2_col, y_col, y_label, title) in zip(axes, specs):
        x = summary["dm41_calc_eV2"].to_numpy(dtype=float)
        y = summary[y_col].to_numpy(dtype=float)
        z = summary[chi2_col].to_numpy(dtype=float)
        positive = z[np.isfinite(z) & (z > 0.0)]
        norm = LogNorm(vmin=max(float(np.nanmin(positive)), 1.0e-8), vmax=float(np.nanmax(positive))) if positive.size else None
        sc = ax.scatter(x, y, c=z, s=16, alpha=0.72, cmap="viridis", norm=norm, edgecolors="none")
        contour_if_possible(ax, x, y, z, levels)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]")
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, which="both")
        cbar = fig.colorbar(sc, ax=ax, pad=0.015)
        cbar.set_label(r"$\chi^2_\mathrm{stat}$")
        legend_lines = [
            plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="90% CL, 2 dof"),
            plt.Line2D([0], [0], color="0.2", linestyle="--", linewidth=1.5, label="95% CL, 2 dof"),
        ]
        ax.legend(handles=legend_lines, frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        r"DUNE ND ISS(2,3) stat-only exclusion proxy: "
        r"$\chi^2=\sum_\mathrm{channels,bins}(N_{4\nu}-N_{3\nu})^2/(N_{3\nu}+\epsilon)$",
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First stat-only DUNE ND ISS(2,3) exclusion proxy.")
    parser.add_argument("--points-csv", type=Path, default=POINTS_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out", type=Path, default=OUT_FIG)
    parser.add_argument("--e-min-GeV", type=float, default=0.5)
    parser.add_argument("--e-max-GeV", type=float, default=8.0)
    parser.add_argument("--e-bins", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = select_points(args.points_csv, args.max_points)
    if args.workers > 1 and len(points) > args.chunk_size:
        chunks = split_dataframe(points, args.chunk_size)
        print(f"Calcul parallele: {len(chunks)} blocs, {args.workers} workers, {len(points)} points")
        payloads = [(chunk, args.e_min_GeV, args.e_max_GeV, args.e_bins, args.epsilon) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            summary = pd.concat(pool.map(compute_stat_only_chi2_worker, payloads), ignore_index=True)
    else:
        summary = compute_stat_only_chi2(points, args.e_min_GeV, args.e_max_GeV, args.e_bins, args.epsilon)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    draw_exclusion(summary, args.out)
    print(f"Points utilises: {len(summary)}")
    print(f"CSV sauvegarde: {args.out_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")
    for col in ("chi2_appearance", "chi2_disappearance", "chi2_all"):
        print(
            f"{col}: min={summary[col].min():.6g}, "
            f"median={summary[col].median():.6g}, max={summary[col].max():.6g}"
        )


if __name__ == "__main__":
    main()
