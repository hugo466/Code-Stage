"""
Plot des probabilités d'oscillation νμ→νμ et νμ→νe en fonction de la distance L (km),
à énergie fixe, pour différentes valeurs de Δm²₄₁.
Affiche aussi les probabilités avec la matrice PMNS 3x3 standard (sans neutrinos stériles).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("data/oscillations/3p1/nd_3p1_vs_distance.csv")
OUTPUT_PATH = Path("figures/oscillations/3p1/nd_3p1_vs_distance.png")

COLORS = {
    0.1: "green",
    1.0: "red",
    10.0: "blue",
    100.0: "orange",
}


def load_data(path: Path):
    grouped = defaultdict(lambda: {"L": [], "Pmumu": [], "Pmue": [], "Pmumu_3nu": [], "Pmue_3nu": []})
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dm = float(row["dm41_eV2"])
            grouped[dm]["L"].append(float(row["baseline_km"]))
            grouped[dm]["Pmumu"].append(float(row["P_mumu_disappearance"]))
            grouped[dm]["Pmue"].append(float(row["P_mue_appearance"]))
            grouped[dm]["Pmumu_3nu"].append(float(row["P_mumu_3nu"]))
            grouped[dm]["Pmue_3nu"].append(float(row["P_mue_3nu"]))
    return dict(sorted(grouped.items()))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {DATA_PATH}\nLance d'abord le programme C.")

    data = load_data(DATA_PATH)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Courbes 3+1 (une par valeur de dm41)
    for dm41, values in data.items():
        color = COLORS.get(dm41)
        label_3p1 = rf"$\Delta m_{{41}}^2 = {dm41:g}\,\mathrm{{eV}}^2$"
        ax_l.plot(values["L"], values["Pmumu"], color=color, lw=1.8, label=label_3p1, linestyle='-')
        ax_r.plot(values["L"], values["Pmue"],  color=color, lw=1.8, label=label_3p1, linestyle='-')

    # Courbe 3ν standard — identique pour tous les dm41, tracée une seule fois
    first = next(iter(data.values()))
    ax_l.plot(first["L"], first["Pmumu_3nu"], color="black", lw=1.8, label=r"3$\nu$ sans stérile", linestyle='--')
    ax_r.plot(first["L"], first["Pmue_3nu"],  color="black", lw=1.8, label=r"3$\nu$ sans stérile", linestyle='--')

    ax_l.set_title(r"$\nu_\mu \to \nu_\mu$ (disparition)")
    ax_r.set_title(r"$\nu_\mu \to \nu_e$ (apparition)")

    for ax in (ax_l, ax_r):
        ax.set_xlabel("Distance L [km]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    ax_l.set_ylabel("Probabilité")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée : {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
