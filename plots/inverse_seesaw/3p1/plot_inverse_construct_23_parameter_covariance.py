import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_3p1.csv")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_construct_23_parameter_covariance.png")

FREE_PARAMETER_COLUMNS = [
    "dm41_target_eV2",
    "zeta_norm",
    "zeta_direction_deg",
    "f11",
    "f12",
    "f21",
    "f22",
    "M1_GeV",
    "M2_GeV",
]

DISPLAY_LABELS = [
    r"$\Delta m_{41}^2$",
    r"$\|\zeta\|$",
    r"$\phi_\zeta$",
    r"$f_{11}$",
    r"$f_{12}$",
    r"$f_{21}$",
    r"$f_{22}$",
    r"$M_1$",
    r"$M_2$",
]


def load_free_parameter_matrices(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")

    rows_all = []
    rows_eta_pass = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row.get("solve_ok", "0"))) != 1:
                    continue
            except ValueError:
                continue

            values = []
            valid = True
            for col in FREE_PARAMETER_COLUMNS:
                raw = row.get(col, "")
                try:
                    value = float(raw)
                except ValueError:
                    valid = False
                    break
                if not np.isfinite(value):
                    valid = False
                    break
                values.append(value)

            if valid:
                try:
                    pmns_pass = int(float(row.get("pmns_pass", "0")))
                except ValueError:
                    pmns_pass = 0
                if pmns_pass != 1:
                    continue
                rows_all.append(values)
                try:
                    eta_pass = int(float(row.get("eta_pass", "0")))
                except ValueError:
                    eta_pass = 0
                if eta_pass == 1:
                    rows_eta_pass.append(values)

    if len(rows_all) < 2:
        raise RuntimeError("Pas assez de points valides (solve_ok=1, pmns_pass=1) pour calculer une covariance.")

    matrix_all = np.array(rows_all, dtype=float)
    matrix_eta_pass = np.array(rows_eta_pass, dtype=float) if len(rows_eta_pass) >= 2 else None
    return matrix_all, matrix_eta_pass


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(covariance))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def plot_correlation_matrix(ax, correlation: np.ndarray, title: str):
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(DISPLAY_LABELS)))
    ax.set_yticks(np.arange(len(DISPLAY_LABELS)))
    ax.set_xticklabels(DISPLAY_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_LABELS)

    for i in range(len(DISPLAY_LABELS)):
        for j in range(len(DISPLAY_LABELS)):
            text = f"{correlation[i, j]:.2f}"
            color = "white" if abs(correlation[i, j]) > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title)
    return im


def main():
    matrix_all, matrix_eta_pass = load_free_parameter_matrices(DATA_PATH)
    covariance_all = np.cov(matrix_all, rowvar=False)
    correlation_all = covariance_to_correlation(covariance_all)
    point_count_all = matrix_all.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 20))

    im_all = plot_correlation_matrix(
        axes[0],
        correlation_all,
        rf"Matrice de covariance (points PMNS OK: {point_count_all})"
    )
    cbar_all = fig.colorbar(im_all, ax=axes[0], shrink=0.9)
    cbar_all.set_label("Corrélation")

    if matrix_eta_pass is not None:
        covariance_eta_pass = np.cov(matrix_eta_pass, rowvar=False)
        correlation_eta_pass = covariance_to_correlation(covariance_eta_pass)
        point_count_eta_pass = matrix_eta_pass.shape[0]
        im_eta = plot_correlation_matrix(
            axes[1],
            correlation_eta_pass,
            rf"Matrice de covariance (PMNS OK + η OK: {point_count_eta_pass})"
        )
        cbar_eta = fig.colorbar(im_eta, ax=axes[1], shrink=0.9)
        cbar_eta.set_label("Corrélation")
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Pas assez de points eta_pass=1\npour calculer une covariance",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[1].transAxes,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")

    plt.close(fig)


if __name__ == "__main__":
    main()
