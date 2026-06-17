#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_rebuild_masses.png"


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")


def _finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _finite_triplet(x, y, m):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=bool)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], m[mask]


def _plot_scatter_panel(ax, x, y, eta_mask, title, xlabel, ylabel, log_scale=False):
    x, y, eta_mask = _finite_triplet(x, y, eta_mask)
    if x.size == 0:
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        return

    if log_scale:
        pos = (x > 0.0) & (y > 0.0)
        x = x[pos]
        y = y[pos]
        eta_mask = eta_mask[pos]
        if x.size == 0:
            ax.set_title(f"{title} (no positive data)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            return

    # Base: PMNS-pass points
    ax.scatter(x, y, s=5, alpha=0.20, color="#1f77b4", edgecolors="none", label="pmns_pass=1")

    # Overlay: eta-pass subset
    x_eta = x[eta_mask]
    y_eta = y[eta_mask]
    if x_eta.size > 0:
        ax.scatter(x_eta, y_eta, s=7, alpha=0.35, color="#d62728", edgecolors="none", label="eta_pass=1")

    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if np.isclose(lo, hi):
        span = max(abs(lo), 1.0) * 0.1
        lo -= span
        hi += span

    if log_scale:
        lo = max(lo, np.finfo(float).tiny)
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point trouvé.")

    required = [
        "solve_ok", "pmns_pass", "eta_pass",
        "dm21_target_eV2", "dm21_calc_eV2",
        "dm31_target_eV2", "dm31_calc_eV2",
        "dm41_target_eV2", "dm41_calc_eV2",
        "M1_GeV", "M2_GeV",
    ]

    _require_columns(df, required)

    for col in ["solve_ok", "pmns_pass", "eta_pass"]:
        df[col] = df[col].fillna(0).astype(int)

    df_sel = df[(df["solve_ok"] == 1) & (df["pmns_pass"] == 1)].copy()
    if df_sel.empty:
        raise RuntimeError("Aucun point solve_ok=1 et pmns_pass=1.")

    eta_mask = (df_sel["eta_pass"].to_numpy(int) == 1)

    n = len(df_sel)

    masses = [
        ("dm21", df_sel["dm21_target_eV2"].to_numpy(float), df_sel["dm21_calc_eV2"].to_numpy(float), r"$\Delta m^2_{21}$ [eV$^2$]"),
        ("dm31", df_sel["dm31_target_eV2"].to_numpy(float), df_sel["dm31_calc_eV2"].to_numpy(float), r"$\Delta m^2_{31}$ [eV$^2$]"),
        ("dm41", df_sel["dm41_target_eV2"].to_numpy(float), df_sel["dm41_calc_eV2"].to_numpy(float), r"$\Delta m^2_{41}$ [eV$^2$]"),
    ]

    # has_m1_reco = "M1_calc_GeV" in df_sel.columns
    # has_m2_reco = "M2_calc_GeV" in df_sel.columns

    # m1_target = df_sel["M1_GeV"].to_numpy(float)
    # m2_target = df_sel["M2_GeV"].to_numpy(float)

    # if has_m1_reco:
    #     m1_reco = df_sel["M1_calc_GeV"].to_numpy(float)
    # else:
    #     m1_reco = m1_target.copy()
    #     print("[warn] M1_calc_GeV absent: using M1_GeV as fallback for reconstructed axis.")

    # if has_m2_reco:
    #     m2_reco = df_sel["M2_calc_GeV"].to_numpy(float)
    # else:
    #     m2_reco = m2_target.copy()
    #     print("[warn] M2_calc_GeV absent: using M2_GeV as fallback for reconstructed axis.")

    # heavy_masses = [
    #     ("M1", m1_target, m1_reco, r"$M_1$ [GeV]", has_m1_reco),
    #     ("M2", m2_target, m2_reco, r"$M_2$ [GeV]", has_m2_reco),
    # ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()

    # Row 1: mass splittings (fixed target vs reconstructed)
    for idx, (_, target, reco, label) in enumerate(masses):
        _plot_scatter_panel(
            axes[idx],
            target,
            reco,
            eta_mask,
            title=f"{label}",
            xlabel=f"Fixé (target) {label}",
            ylabel=f"Reconstruit {label}",
            log_scale=True,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle(
        f"masses fixées vs reconstruites",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
