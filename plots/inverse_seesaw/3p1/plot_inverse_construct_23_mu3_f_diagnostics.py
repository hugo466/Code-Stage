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
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_mu3_f.png"


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")


def _safe_log10(values, floor=1e-30):
    x = np.asarray(values, dtype=float)
    return np.log10(np.clip(x, floor, None))


def _finite_xyb(x, y, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=bool)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m], b[m]


def _scatter_with_categories(ax, x, y, eta_mask, title, xlabel, ylabel):
    x, y, eta = _finite_xyb(x, y, eta_mask)
    if x.size == 0:
        ax.set_title(f"{title} (no data)")
        ax.grid(alpha=0.25)
        return

    ax.scatter(x, y, s=7, alpha=0.20, color="#1f77b4", edgecolors="none", label="pmns_pass=1")

    if np.any(eta):
        ax.scatter(x[eta], y[eta], s=9, alpha=0.35, color="#d62728", edgecolors="none", label="eta_pass=1")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point disponible.")

    required = [
        "solve_ok", "pmns_pass", "eta_pass",
        "mu3_11_eV", "mu3_12_eV", "mu3_13_eV",
        "mu3_21_eV", "mu3_22_eV", "mu3_23_eV",
        "mu3_31_eV", "mu3_32_eV", "mu3_33_eV",
        "kappa_f", "det_f", "mL_rel_frob_error",
    ]
    _require_columns(df, required)

    for c in ["solve_ok", "pmns_pass", "eta_pass"]:
        df[c] = df[c].fillna(0).astype(int)

    sel = (df["solve_ok"] == 1) & (df["pmns_pass"] == 1)
    d = df.loc[sel].copy()
    if d.empty:
        raise RuntimeError("Aucun point solve_ok=1 et pmns_pass=1")

    mu_cols = [
        "mu3_11_eV", "mu3_12_eV", "mu3_13_eV",
        "mu3_21_eV", "mu3_22_eV", "mu3_23_eV",
        "mu3_31_eV", "mu3_32_eV", "mu3_33_eV",
    ]
    mu_arr = np.abs(d[mu_cols].to_numpy(float))
    mu_max = np.nanmax(mu_arr, axis=1)
    log_mu_max = _safe_log10(mu_max)

    eta_mask = d["eta_pass"].to_numpy(int) == 1

    kappa = d["kappa_f"].to_numpy(float)
    detf = np.abs(d["det_f"].to_numpy(float))
    ml_err = d["mL_rel_frob_error"].to_numpy(float)

    log_kappa = _safe_log10(kappa)
    log_ml_err = _safe_log10(ml_err + 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes.flatten()

    _scatter_with_categories(ax[0], log_kappa, log_mu_max, eta_mask,
                             "mu3 vs kappa(f)", "log10(kappa_f)", "log10(max |mu3_ij| [eV])")

    _scatter_with_categories(ax[1], -_safe_log10(detf), log_mu_max, eta_mask,
                             "mu3 vs |det(f)|", "-log10(|det(f)|)", "log10(max |mu3_ij| [eV])")

    _scatter_with_categories(ax[2], log_ml_err, log_mu_max, eta_mask,
                             "mu3 vs erreur bloc mL", "log10(mL_rel_frob_error)", "log10(max |mu3_ij| [eV])")

    handles, labels = ax[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    n = len(d)
    n_eta = int(np.sum(eta_mask))
    fig.suptitle(
        (
            "N={n}, eta_pass={n_eta}"
        ),
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
