"""
Heatmap ACP = P(numu->nue) - P(antinumu->antinue) en 3+1 neutrinos
Axes :  x = énergie (GeV)
        y = delta_41 (deg)
Couleur : ACP  (rouge = violation asymétrique, bleu = asymétrie inverse)
Deux colonnes : détecteur proche (574 m) | détecteur lointain (1300 km)
Un groupe de colonnes par valeur de dm41.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import os
from collections import defaultdict

CSV_PATH = os.path.join("data", "oscillations", "3p1", "cp_heatmap_3p1.csv")
OUT_DIR  = os.path.join("figures", "oscillations", "3p1")
OUT_FILE = os.path.join(OUT_DIR, "cp_heatmap_3p1.png")

os.makedirs(OUT_DIR, exist_ok=True)

data = defaultdict(lambda: defaultdict(list))

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dm41 = float(row["dm41_eV2"])
        base = float(row["baseline_km"])
        logE = float(row["log10_energy_GeV"])
        d41 = float(row["delta41_deg"])
        acp = float(row["ACP"])
        data[dm41][base].append((logE, d41, acp))

dm41_values = sorted(data.keys())
baseline_set = sorted({b for dm in data.values() for b in dm.keys()})
n_dm41 = len(dm41_values)
n_baselines = len(baseline_set)


def build_grid(pts):
    e_set = sorted({e for e, d, a in pts})
    delta_set = sorted({d for e, d, a in pts})

    e_idx = {e: i for i, e in enumerate(e_set)}
    d_idx = {d: i for i, d in enumerate(delta_set)}

    z = np.zeros((len(delta_set), len(e_set)))
    for e_val, d_val, a_val in pts:
        z[d_idx[d_val], e_idx[e_val]] = a_val

    return np.array(e_set), np.array(delta_set), z


def detector_label(base_km):
    if base_km < 1.0:
        return f"Détecteur proche — $L = {base_km * 1e3:.0f}$ m"
    return f"Détecteur lointain — $L = {base_km:.0f}$ km"


fig, axes = plt.subplots(
    n_dm41, n_baselines,
    figsize=(5.5 * n_baselines, 4.8 * n_dm41),
    squeeze=False,
    constrained_layout=True,
)

cmap = "RdBu_r"

for row_i, dm41 in enumerate(dm41_values):
    for col_j, base in enumerate(baseline_set):
        ax = axes[row_i, col_j]
        pts = data[dm41].get(base, [])

        if not pts:
            ax.set_visible(False)
            continue

        egrid, dgrid, z = build_grid(pts)
        vmax = max(abs(z.min()), abs(z.max()), 1e-12)

        im = ax.pcolormesh(
            egrid, dgrid, z,
            cmap=cmap,
            vmin=-vmax, vmax=vmax,
            shading="auto",
            rasterized=True,
        )

        loge_min, loge_max = float(egrid.min()), float(egrid.max())
        major_ticks = np.arange(np.ceil(loge_min), np.floor(loge_max) + 0.5, 1.0)
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"$10^{{{x:.4g}}}$")
        )
        ax.set_box_aspect(1)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            r"$A_{CP} = P(\nu_\mu\!\to\!\nu_e) - P(\bar\nu_\mu\!\to\!\bar\nu_e)$",
            fontsize=12,
        )

        ax.set_xlabel(r"Énergie $E$ (GeV)", fontsize=10)
        ax.set_ylabel(r"$\delta_{41}$ (deg)", fontsize=10)
        ax.set_ylim(dgrid.min(), dgrid.max())
        ax.set_xlim(loge_min, loge_max)

        title = detector_label(base)
        if n_dm41 > 1:
            if dm41 >= 1.0:
                title += f"\n$\\Delta m^2_{{41}} = {dm41:.3g}$ eV$^2$"
            else:
                title += f"\n$\\Delta m^2_{{41}} = {dm41:.2e}$ eV$^2$"
        ax.set_title(title, fontsize=10)

fig.suptitle(
    r"Asymétrie CP $A_{CP}(E,\,\delta_{41})$ (3+1 neutrinos)"
    r" — $\Delta m^2_{41}=1\,\mathrm{eV}^2$, $\theta_{14}=\theta_{24}=25°$",
    fontsize=12,
)

plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Figure sauvegardée : {OUT_FILE}")
