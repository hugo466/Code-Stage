import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("data/oscillations/3p1/nd_3p1_probabilities.csv")
OUTPUT_PATH = Path("figures/oscillations/3p1/nd_3p1_probabilities.png")

COLORS = {
    0.1: "green",
    1.0: "red",
    10.0: "blue",
    100.0: "orange",
}


def load_data(path: Path):
    grouped = defaultdict(lambda: {"E": [], "Pmumu": [], "Pmue": []})

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dm = float(row["dm41_eV2"])
            grouped[dm]["E"].append(float(row["energy_GeV"]))
            grouped[dm]["Pmumu"].append(float(row["P_mumu_disappearance"]))
            grouped[dm]["Pmue"].append(float(row["P_mue_appearance"]))

    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "CSV introuvable. Exécute d'abord le programme C pour générer data/oscillations/3p1/nd_3p1_probabilities.csv"
        )

    data = load_data(DATA_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax_left, ax_right = axes

    for dm41, values in data.items():
        color = COLORS.get(dm41, None)
        label = rf"$\Delta m_{{41}}^2 = {dm41:g}\,\mathrm{{eV}}^2$"

        ax_left.plot(values["E"], values["Pmumu"], color=color, lw=2, label=label)
        ax_right.plot(values["E"], values["Pmue"], color=color, lw=2, label=label)

    ax_left.set_title(r"$\nu_\mu \to \nu_\mu$ (disparition)")
    ax_right.set_title(r"$\nu_\mu \to \nu_e$ (apparition)")

    ax_left.set_ylabel("Probabilité")
    ax_left.set_xlabel("Énergie [GeV]")
    ax_right.set_xlabel("Énergie [GeV]")

    ax_left.grid(alpha=0.25)
    ax_right.grid(alpha=0.25)

    ax_left.legend(fontsize=9)
    ax_right.legend(fontsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
