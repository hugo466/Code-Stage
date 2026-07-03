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

from inverse_construct_24_observables import build_mu4_abs, load_construct24_points, log_bins

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_mu4.png"


def hist_many(ax, arrays, labels, title, xlabel, colors=None, logx=True):
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    all_values = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    bins = log_bins(all_values, 80) if logx else 80
    for arr, label, color in zip(arrays, labels, colors):
        ax.hist(arr, bins=bins, histtype="step", linewidth=1.4, label=label, color=color)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, frameon=False)


def scatter(ax, x, y, title, xlabel, ylabel, log=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if log:
        mask &= (x > 0.0) & (y > 0.0)
    ax.scatter(x[mask], y[mask], s=7, alpha=0.22, color="tab:purple", edgecolors="none")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, which="both")


def plot_dataset(axes, label, df):
    mu4 = build_mu4_abs(df)
    sv = np.linalg.svd(mu4, compute_uv=False)
    kappa = sv[:, 0] / np.maximum(sv[:, -1], 1e-30)

    mu00 = mu4[:, 0:2, 0:2]
    muH0 = mu4[:, 2:4, 0:2]
    muH = mu4[:, 2:4, 2:4]

    mu00_00 = mu00[:, 0, 0]
    mu00_01 = mu00[:, 0, 1]
    mu00_11 = mu00[:, 1, 1]
    muH0_flat = muH0.reshape(len(df), 4)
    muH_11 = muH[:, 0, 0]
    muH_12 = muH[:, 0, 1]
    muH_22 = muH[:, 1, 1]

    mu00_max = np.nanmax(mu00.reshape(len(df), -1), axis=1)
    muH0_max = np.nanmax(muH0_flat, axis=1)
    muH_max = np.nanmax(muH.reshape(len(df), -1), axis=1)
    mu4_max = np.nanmax(mu4.reshape(len(df), -1), axis=1)

    hist_many(
        axes[0],
        [mu00_00, mu00_01, mu00_11],
        [r"$|\mu_{00,00}|$", r"$|\mu_{00,01}|$", r"$|\mu_{00,11}|$"],
        f"Bloc mu00 - {label}",
        r"$|\mu_{00,ij}|$ [eV]",
        colors=["tab:blue", "tab:orange", "tab:green"],
    )
    hist_many(
        axes[1],
        [muH0_flat[:, 0], muH0_flat[:, 1], muH0_flat[:, 2], muH0_flat[:, 3]],
        [r"00", r"01", r"10", r"11"],
        f"Bloc muH0 - {label}",
        r"$|\mu_{H0,ij}|$ [eV]",
    )
    hist_many(
        axes[2],
        [muH_11, muH_12, muH_22],
        [r"$|\mu_{H,00}|$", r"$|\mu_{H,01}|$", r"$|\mu_{H,11}|$"],
        f"Bloc muH - {label}",
        r"$|\mu_{H,ij}|$ [eV]",
        colors=["tab:blue", "tab:orange", "tab:green"],
    )
    hist_many(
        axes[3],
        [sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]],
        [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"],
        f"Valeurs singulieres mu4 - {label}",
        r"$\sigma_i(\mu_4)$ [eV]",
    )

    axes[4].hist(kappa, bins=log_bins(kappa, 90), histtype="stepfilled", color="tab:red", alpha=0.75)
    axes[4].set_xscale("log")
    axes[4].set_title(f"Conditionnement mu4 - {label}")
    axes[4].set_xlabel(r"$\kappa(\mu_4)$")
    axes[4].set_ylabel("count")
    axes[4].grid(alpha=0.25, which="both")

    hist_many(
        axes[5],
        [mu00_max, muH0_max, muH_max, mu4_max],
        [r"$\max|\mu_{00}|$", r"$\max|\mu_{H0}|$", r"$\max|\mu_H|$", r"$\max|\mu_4|$"],
        f"Echelles des blocs - {label}",
        r"$|\mu|$ [eV]",
    )

    trace_muH = muH[:, 0, 0] + muH[:, 1, 1]
    det_muH = muH[:, 0, 0] * muH[:, 1, 1] - muH[:, 0, 1] * muH[:, 1, 0]
    scatter(axes[6], trace_muH, np.abs(det_muH), f"Trace/det muH - {label}", r"Tr $|\mu_H|$ [eV]", r"$|\det \mu_H|$ [eV$^2$]")
    scatter(axes[7], mu00_00, mu00_11, f"Diagonales mu00 - {label}", r"$|\mu_{00,00}|$ [eV]", r"$|\mu_{00,11}|$ [eV]")


def main() -> None:
    df_pmns = load_construct24_points()
    df_eta = load_construct24_points(eta_only=True)

    fig, axes = plt.subplots(4, 4, figsize=(19.0, 17.0))
    plot_dataset(axes[:2, :].ravel(), f"PMNS (N={len(df_pmns)})", df_pmns)
    plot_dataset(axes[2:, :].ravel(), f"PMNS+eta (N={len(df_eta)})", df_eta)

    fig.suptitle("ISS(2,4) construct_24 - diagnostics du bloc mu4", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=190)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
