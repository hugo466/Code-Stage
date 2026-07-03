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

from inverse_construct_24_observables import build_mu4_abs, finite_xy_eta, load_construct24_points, require_columns

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_mu4_f.png"


def safe_log10(values, floor=1e-300):
    values = np.asarray(values, dtype=float)
    return np.log10(np.clip(values, floor, None))


def scatter(ax, x, y, eta, title, xlabel):
    x, y, eta = finite_xy_eta(x, y, eta)
    ax.scatter(x, y, s=7, alpha=0.22, color="tab:blue", edgecolors="none", label="PMNS")
    if np.any(eta):
        ax.scatter(x[eta], y[eta], s=9, alpha=0.42, color="tab:red", edgecolors="none", label="PMNS+eta")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10}(\max |\mu_4|\,/\,{\rm eV})$")
    ax.grid(alpha=0.25)


def main() -> None:
    df = load_construct24_points()
    required = ["eta_pass", "f11_abs", "f12_abs", "f21_abs", "f22_abs", "mL_rel_frob_error", "U5_abs_rms_error"]
    require_columns(df, required)

    mu4 = build_mu4_abs(df)
    mu4_max = np.nanmax(mu4.reshape(len(df), -1), axis=1)
    y = safe_log10(mu4_max)
    eta = df["eta_pass"].astype(int).to_numpy() == 1

    f11 = df["f11_abs"].to_numpy(float)
    f12 = df["f12_abs"].to_numpy(float)
    f21 = df["f21_abs"].to_numpy(float)
    f22 = df["f22_abs"].to_numpy(float)
    detf = np.abs(f11 * f22 - f12 * f21)
    sv = np.linalg.svd(np.stack([f11, f12, f21, f22], axis=1).reshape(-1, 2, 2), compute_uv=False)
    kappa = sv[:, 0] / np.maximum(sv[:, -1], 1e-30)

    panels = [
        (safe_log10(kappa), "mu4 vs kappa(f)", r"$\log_{10}\kappa(f)$"),
        (-safe_log10(detf), "mu4 vs determinant f", r"$-\log_{10}|\det f|$"),
        (safe_log10(df["mL_rel_frob_error"].to_numpy(float)), "mu4 vs mL reconstruction", r"$\log_{10}$ mL relative error"),
        (safe_log10(df["U5_abs_rms_error"].to_numpy(float)), "mu4 vs U5 reconstruction", r"$\log_{10}$ U5 abs RMS error"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for ax, (x, title, xlabel) in zip(axes.ravel(), panels):
        scatter(ax, x, y, eta, title, xlabel)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"ISS(2,4) construct_24 - correlations mu4/f ({len(df)} points)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
