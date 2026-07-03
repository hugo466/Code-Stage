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

from inverse_construct_24_observables import load_construct24_points, log_bins, require_columns

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_f.png"


def main() -> None:
    df = load_construct24_points(eta_only=True)
    required = ["f11_abs", "f12_abs", "f21_abs", "f22_abs"]
    require_columns(df, required)
    f11 = df["f11_abs"].to_numpy(float)
    f12 = df["f12_abs"].to_numpy(float)
    f21 = df["f21_abs"].to_numpy(float)
    f22 = df["f22_abs"].to_numpy(float)
    f_stack = np.stack([f11, f12, f21, f22], axis=1)
    det = np.abs(f11 * f22 - f12 * f21)
    sv = np.linalg.svd(f_stack.reshape(-1, 2, 2), compute_uv=False)
    kappa = sv[:, 0] / np.maximum(sv[:, -1], 1e-30)

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.0))
    ax = axes.ravel()
    for idx, (values, title) in enumerate(
        [
            (f11, r"$|f_{11}|$"),
            (f12, r"$|f_{12}|$"),
            (f21, r"$|f_{21}|$"),
            (f22, r"$|f_{22}|$"),
        ]
    ):
        ax[idx].hist(values, bins=70, histtype="stepfilled", alpha=0.75, color="tab:blue")
        ax[idx].set_title(title)
        ax[idx].set_xlabel(title)
        ax[idx].set_ylabel("density")
        ax[idx].grid(alpha=0.25)

    ax[4].hist(det, bins=log_bins(det, 80), histtype="stepfilled", alpha=0.75, color="tab:red")
    ax[4].set_xscale("log")
    ax[4].set_title(r"$|\det f|$")
    ax[4].set_xlabel(r"$|\det f|$")
    ax[4].grid(alpha=0.25, which="both")

    ax[5].hist(kappa, bins=log_bins(kappa, 80), histtype="stepfilled", alpha=0.75, color="tab:red")
    ax[5].set_xscale("log")
    ax[5].set_title(r"$\kappa(f)$")
    ax[5].set_xlabel(r"$\kappa(f)$")
    ax[5].grid(alpha=0.25, which="both")

    ax[6].hist(sv[:, 0], bins=70, histtype="step", linewidth=1.5, color="tab:blue", label=r"$\sigma_1(f)$")
    ax[6].hist(sv[:, 1], bins=70, histtype="step", linewidth=1.5, color="tab:orange", label=r"$\sigma_2(f)$")
    ax[6].set_title("Valeurs singulieres")
    ax[6].set_xlabel(r"$\sigma_i(f)$")
    ax[6].grid(alpha=0.25)
    ax[6].legend(frameon=False, fontsize=9)

    ax[7].axis("off")

    fig.suptitle(f"ISS(2,4) construct_24 - diagnostics du bloc f ({len(df)} points PMNS+eta)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
