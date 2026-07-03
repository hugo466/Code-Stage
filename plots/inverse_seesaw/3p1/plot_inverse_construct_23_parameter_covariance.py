from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_parameter_covariance.png"
F_PARAMETER_REPRESENTATION = "abs_phase"  # "abs_phase" or "re_im"

def build_parameter_schema(representation: str):
    base_columns = [
        "dm41_target_eV2",
        "zeta_norm",
        "zeta_direction_deg",
        "zeta_phase_deg",
        "majorana_alpha21_deg",
        "majorana_alpha31_deg",
    ]
    base_labels = [
        r"$\Delta m_{41}^2$",
        r"$\|\zeta\|$",
        r"$\phi_\zeta$",
        r"$\varphi_\zeta$",
        r"$\alpha_{21}$",
        r"$\alpha_{31}$",
    ]

    if representation == "abs_phase":
        f_columns = ["f11", "f12", "f21", "f22", "f11_phase_deg", "f12_phase_deg", "f21_phase_deg", "f22_phase_deg"]
        f_labels = [
            r"$|f_{11}|$",
            r"$|f_{12}|$",
            r"$|f_{21}|$",
            r"$|f_{22}|$",
            r"$\phi_{f_{11}}$",
            r"$\phi_{f_{12}}$",
            r"$\phi_{f_{21}}$",
            r"$\phi_{f_{22}}$",
        ]
    elif representation == "re_im":
        f_columns = ["f11_re", "f11_im", "f12_re", "f12_im", "f21_re", "f21_im", "f22_re", "f22_im"]
        f_labels = [
            r"$\Re(f_{11})$",
            r"$\Im(f_{11})$",
            r"$\Re(f_{12})$",
            r"$\Im(f_{12})$",
            r"$\Re(f_{21})$",
            r"$\Im(f_{21})$",
            r"$\Re(f_{22})$",
            r"$\Im(f_{22})$",
        ]
    else:
        raise ValueError("F_PARAMETER_REPRESENTATION must be 'abs_phase' or 're_im'.")

    end_columns = ["M1_GeV", "M2_GeV"]
    end_labels = [r"$M_1$", r"$M_2$"]
    return base_columns + f_columns + end_columns, base_labels + f_labels + end_labels


def extract_row_values(row, columns, representation: str):
    values = []
    for col in columns:
        if col.endswith("_re") or col.endswith("_im"):
            fij = col[:3]
            mag = row.get(fij, np.nan)
            phase_deg = row.get(f"{fij}_phase_deg", np.nan)
            if not (np.isfinite(mag) and np.isfinite(phase_deg)):
                return None
            phase_rad = np.deg2rad(float(phase_deg))
            val = float(mag) * (np.cos(phase_rad) if col.endswith("_re") else np.sin(phase_rad))
        else:
            if col in {"zeta_phase_deg", "majorana_alpha21_deg", "majorana_alpha31_deg"} and col not in row:
                val = 0.0
                values.append(val)
                continue
            val = row.get(col, np.nan)
            if not np.isfinite(val):
                return None
            val = float(val)
        values.append(val)
    return values


def load_free_parameter_matrices(data_dir: Path, columns, representation: str):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de points conservés introuvable: {data_dir}")

    df = load_kept_points_dataframe(data_dir)
    if df.empty:
        raise RuntimeError("Aucun point conservé trouvé.")

    rows_all = []
    rows_eta_pass = []
    for _, row in df.iterrows():
        values = extract_row_values(row, columns, representation)
        if values is not None:
            rows_all.append(values)
            if int(row.get("eta_pass", 0)) == 1:
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


def plot_correlation_matrix(ax, correlation: np.ndarray, title: str, labels):
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{correlation[i, j]:.2f}"
            color = "white" if abs(correlation[i, j]) > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title)
    return im


def main():
    free_parameter_columns, display_labels = build_parameter_schema(F_PARAMETER_REPRESENTATION)
    matrix_all, matrix_eta_pass = load_free_parameter_matrices(
        DATA_DIR,
        free_parameter_columns,
        F_PARAMETER_REPRESENTATION,
    )
    covariance_all = np.cov(matrix_all, rowvar=False)
    correlation_all = covariance_to_correlation(covariance_all)
    point_count_all = matrix_all.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 20))

    im_all = plot_correlation_matrix(
        axes[0],
        correlation_all,
        rf"Matrice de covariance (points PMNS OK: {point_count_all})",
        display_labels,
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
            rf"Matrice de covariance (PMNS OK + η OK: {point_count_eta_pass})",
            display_labels,
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
