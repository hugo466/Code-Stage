from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_construct_24_kept_points import load_kept_points_dataframe

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_parameter_covariance.png"


def parameter_schema():
    columns = [
        "dm41_target_eV2",
        "dm51_target_eV2",
        "s1",
        "s2",
        "V_angle_deg",
        "W_angle_deg",
        "V_alpha_deg",
        "V_beta_deg",
        "V_gamma_deg",
        "W_alpha_deg",
        "W_beta_deg",
        "W_gamma_deg",
        "majorana_alpha21_deg",
        "majorana_alpha31_deg",
        "M1_GeV",
        "M2_GeV",
        "f11_abs",
        "f12_abs",
        "f21_abs",
        "f22_abs",
        "f11_phase_deg",
        "f12_phase_deg",
        "f21_phase_deg",
        "f22_phase_deg",
    ]
    labels = [
        r"$\Delta m^2_{41}$",
        r"$\Delta m^2_{51}$",
        r"$s_1$",
        r"$s_2$",
        r"$\theta_V$",
        r"$\theta_W$",
        r"$\alpha_V$",
        r"$\beta_V$",
        r"$\gamma_V$",
        r"$\alpha_W$",
        r"$\beta_W$",
        r"$\gamma_W$",
        r"$\alpha_{21}$",
        r"$\alpha_{31}$",
        r"$M_1$",
        r"$M_2$",
        r"$|f_{11}|$",
        r"$|f_{12}|$",
        r"$|f_{21}|$",
        r"$|f_{22}|$",
        r"$\phi_{f_{11}}$",
        r"$\phi_{f_{12}}$",
        r"$\phi_{f_{21}}$",
        r"$\phi_{f_{22}}$",
    ]
    return columns, labels


def covariance_to_correlation(data: pd.DataFrame) -> np.ndarray:
    matrix = data.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if matrix.shape[0] < 2:
        raise RuntimeError("Pas assez de points valides pour calculer la covariance.")
    cov = np.cov(matrix, rowvar=False)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, matrix.shape[0]


def plot_corr(ax, corr: np.ndarray, labels: list[str], title: str):
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr[i, j]
            color = "white" if abs(value) > 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.4, color=color)
    ax.set_title(title, fontweight="bold")
    return im


def main() -> None:
    df = load_kept_points_dataframe()
    if df.empty:
        raise RuntimeError("CSV construct_24 introuvable ou vide.")

    coherent = df.get("coherence_pass", 1).astype(int) == 1
    pmns = (df.get("pmns_pass", 0).astype(int) == 1) & coherent
    eta = df.get("eta_pass", 0).astype(int) == 1

    columns, labels = parameter_schema()
    available = [(c, l) for c, l in zip(columns, labels) if c in df.columns]
    columns = [x[0] for x in available]
    labels = [x[1] for x in available]

    df_pmns = df.loc[pmns, columns].copy()
    df_eta = df.loc[pmns & eta, columns].copy()

    corr_pmns, n_pmns = covariance_to_correlation(df_pmns)
    corr_eta, n_eta = covariance_to_correlation(df_eta)

    fig, axes = plt.subplots(2, 1, figsize=(18.0, 32.0))
    im1 = plot_corr(axes[0], corr_pmns, labels, f"Matrice de correlation - PMNS OK ({n_pmns} points)")
    fig.colorbar(im1, ax=axes[0], shrink=0.85, label="Correlation")
    im2 = plot_corr(axes[1], corr_eta, labels, f"Matrice de correlation - PMNS+eta OK ({n_eta} points)")
    fig.colorbar(im2, ax=axes[1], shrink=0.85, label="Correlation")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
