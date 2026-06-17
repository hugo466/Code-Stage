#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

DATA_DIR = get_inverse_kept_points_dir()
CONFIG_PATH = Path("config/presets/inverse_seesaw/3p1/construct_23.txt")
OUTPUT_PNG = Path("figures/inverse_seesaw/3p1/inverse_construct_23_f.png")
OUTPUT_PDF = Path("figures/inverse_seesaw/3p1/inverse_construct_23_f_diagnostics.pdf")


def deg_to_rad(x):
    return np.pi * x / 180.0


def parse_config_value(path: Path, key: str, default_value: float):
    if not path.exists():
        return default_value

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw or "=" not in raw:
            continue
        lhs, rhs = raw.split("=", 1)
        if lhs.strip() == key:
            try:
                return float(rhs.strip())
            except ValueError:
                return default_value
    return default_value


def build_p_matrix_from_nufit(cfg_path: Path):
    t12_deg = parse_config_value(cfg_path, "inverse_nufit_theta12_deg", 33.68)
    t23_deg = parse_config_value(cfg_path, "inverse_nufit_theta23_deg", 43.3)
    t13_deg = parse_config_value(cfg_path, "inverse_nufit_theta13_deg", 8.56)

    t12 = deg_to_rad(t12_deg)
    t23 = deg_to_rad(t23_deg)
    t13 = deg_to_rad(t13_deg)

    c12 = np.cos(t12)
    s12 = np.sin(t12)
    c13 = np.cos(t13)
    s13 = np.sin(t13)
    c23 = np.cos(t23)
    s23 = np.sin(t23)

    y = np.array([
        c12 * c13,
        -s12 * c23 - c12 * s23 * s13,
        s12 * s23 - c12 * c23 * s13,
    ], dtype=float)
    p1 = np.array([
        s12 * c13,
        c12 * c23 - s12 * s23 * s13,
        -c12 * s23 - s12 * c23 * s13,
    ], dtype=float)
    p2 = np.array([
        s13,
        s23 * c13,
        c23 * c13,
    ], dtype=float)

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 0.0 else v

    y = normalize(y)
    p1 = normalize(p1)
    p2 = normalize(p2)

    # P has shape (3,2): columns are p1 and p2
    p = np.column_stack((p1, p2))
    return p


def load_accepted_points(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de points conservés introuvable: {data_dir}")

    df = load_kept_points_dataframe(data_dir)
    required_cols = [
        "solve_ok", "pmns_pass", "eta_pass",
        "f11", "f12", "f21", "f22",
        "f11_phase_deg", "f12_phase_deg", "f21_phase_deg", "f22_phase_deg",
        "det_f",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans le CSV: {missing}")

    # Experimental constraints: solved + PMNS + eta
    sel = (df["solve_ok"] == 1) & (df["pmns_pass"] == 1) & (df["eta_pass"] == 1)
    df_ok = df.loc[sel].copy()
    if df_ok.empty:
        raise RuntimeError("Aucun point ne passe solve_ok=1, pmns_pass=1, eta_pass=1.")

    return df_ok


def compute_matrices_and_spectra(df_ok: pd.DataFrame, p: np.ndarray):
    f11_abs = df_ok["f11"].to_numpy(dtype=float)
    f12_abs = df_ok["f12"].to_numpy(dtype=float)
    f21_abs = df_ok["f21"].to_numpy(dtype=float)
    f22_abs = df_ok["f22"].to_numpy(dtype=float)
    p11 = deg_to_rad(df_ok["f11_phase_deg"].to_numpy(dtype=float))
    p12 = deg_to_rad(df_ok["f12_phase_deg"].to_numpy(dtype=float))
    p21 = deg_to_rad(df_ok["f21_phase_deg"].to_numpy(dtype=float))
    p22 = deg_to_rad(df_ok["f22_phase_deg"].to_numpy(dtype=float))

    f11 = f11_abs * np.exp(1j * p11)
    f12 = f12_abs * np.exp(1j * p12)
    f21 = f21_abs * np.exp(1j * p21)
    f22 = f22_abs * np.exp(1j * p22)

    n = len(df_ok)
    s_f_max = np.empty(n, dtype=float)
    s_f_min = np.empty(n, dtype=float)
    kappa_f = np.empty(n, dtype=float)
    s_F1 = np.empty(n, dtype=float)
    s_F2 = np.empty(n, dtype=float)
    eta_h = np.empty((n, 3, 3), dtype=float)

    for i in range(n):
        f = np.array([
            [f11[i], f12[i]],
            [f21[i], f22[i]],
        ], dtype=complex)

        sv_f = np.linalg.svd(f, compute_uv=False)
        s_f_max[i] = sv_f[0]
        s_f_min[i] = sv_f[-1]
        kappa_f[i] = sv_f[0] / max(sv_f[-1], 1e-15)

        F = p @ f
        sv_F = np.linalg.svd(F, compute_uv=False)
        # F is (3x2), so 2 singular values
        s_F1[i] = sv_F[0]
        s_F2[i] = sv_F[1]

        FFh = F @ F.conj().T
        eta_h[i] = 0.5 * np.real(FFh)

    return {
        "f11_abs": f11_abs,
        "f12_abs": f12_abs,
        "f21_abs": f21_abs,
        "f22_abs": f22_abs,
        "f11_phase_deg": np.rad2deg(p11),
        "f12_phase_deg": np.rad2deg(p12),
        "f21_phase_deg": np.rad2deg(p21),
        "f22_phase_deg": np.rad2deg(p22),
        "f11": f11,
        "f12": f12,
        "f21": f21,
        "f22": f22,
        "det_f_abs": np.abs(df_ok["det_f"].to_numpy(dtype=float)),
        "s_f_max": s_f_max,
        "s_f_min": s_f_min,
        "kappa_f": kappa_f,
        "s_F1": s_F1,
        "s_F2": s_F2,
        "eta_h": eta_h,
    }


def hist_with_imag_overlay(ax, arr, title, xlabel):
    re = np.real(arr)
    im = np.imag(arr)
    ax.hist(re, bins=70, density=True, alpha=0.75, label="Re", color="#1f77b4")
    ax.hist(im, bins=70, density=True, alpha=0.55, label="Im", color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densite")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def log_bins_from_data(values, nbins=80):
    finite_pos = np.asarray(values, dtype=float)
    finite_pos = finite_pos[np.isfinite(finite_pos) & (finite_pos > 0.0)]
    if finite_pos.size == 0:
        return nbins

    vmin = finite_pos.min()
    vmax = finite_pos.max()
    if np.isclose(vmin, vmax):
        vmin *= 0.9
        vmax *= 1.1

    return np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)


def main():
    df_ok = load_accepted_points(DATA_DIR)
    p = build_p_matrix_from_nufit(CONFIG_PATH)
    d = compute_matrices_and_spectra(df_ok, p)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    hist_with_imag_overlay(axes[0], d["f11"], r"$f_{11}$", r"$f_{11}$")
    hist_with_imag_overlay(axes[1], d["f12"], r"$f_{12}$", r"$f_{12}$")
    hist_with_imag_overlay(axes[2], d["f21"], r"$f_{21}$", r"$f_{21}$")
    hist_with_imag_overlay(axes[3], d["f22"], r"$f_{22}$", r"$f_{22}$")

    axes[4].hist(d["det_f_abs"], bins=80, density=True, color="#e31616", alpha=0.8)
    axes[4].set_title(r"Histogramme $|\det(f)|$")
    axes[4].set_xlabel(r"$|\det(f)|$")
    axes[4].set_ylabel("Densite")
    axes[4].grid(alpha=0.25)

    kappa_bins = log_bins_from_data(d["kappa_f"], nbins=80)
    axes[5].hist(d["kappa_f"], bins=kappa_bins, density=True, color="#e31616", alpha=0.8)
    axes[5].set_title(r"$\kappa(f)=\sigma_{max}/\sigma_{min}$")
    axes[5].set_xlabel(r"$\kappa(f)$")
    axes[5].set_ylabel("Densite")
    axes[5].set_xscale("log")
    axes[5].grid(alpha=0.25)

    axes[6].hist(d["s_f_max"], bins=80, density=True, alpha=0.7, label=r"$\sigma_1(f)$", color="#1f77b4")
    axes[6].hist(d["s_f_min"], bins=80, density=True, alpha=0.7, label=r"$\sigma_2(f)$", color="#ff7f0e")
    axes[6].set_title(r"Valeurs singulieres de $f$")
    axes[6].set_xlabel(r"$\sigma_i(f)$")
    axes[6].set_ylabel("Densite")
    axes[6].grid(alpha=0.25)
    axes[6].legend(fontsize=8)

    axes[7].axis("off")

    fig.suptitle(
        f"Points pmns_pass = 1 (N={len(df_ok)})",
        fontsize=16,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=220)
    # fig.savefig(OUTPUT_PDF)
    print(f"Figure sauvegardee: {OUTPUT_PNG}")
    # print(f"Figure sauvegardee: {OUTPUT_PDF}")

    plt.close(fig)


if __name__ == "__main__":
    main()
