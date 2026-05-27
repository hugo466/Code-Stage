import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("data/inverse_seesaw/3p1/inverse_seesaw_3p1.csv")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_seesaw_3p1_pmue_vs_energy.png")


def load_data(path: Path):
    grouped = defaultdict(lambda: {"E": [], "Pmue": []})

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            mu00 = float(row["mu00_eV"])
            grouped[mu00]["E"].append(float(row["energy_GeV"]))
            grouped[mu00]["Pmue"].append(float(row["P_mue_appearance"]))

    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "CSV introuvable. Exécute d'abord le scan C inverse seesaw pour générer data/inverse_seesaw/3p1/inverse_seesaw_3p1.csv"
        )

    data = load_data(DATA_PATH)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.0))

    for mu00, values in data.items():
        label = rf"$\mu_{{00}} = {mu00:g}\,\mathrm{{eV}}$"
        ax.plot(values["E"], values["Pmue"], lw=2.0, label=label)

    ax.set_title(r"Inverse seesaw 3+1 : $P(\nu_\mu \rightarrow \nu_e)$")
    ax.set_xlabel("Énergie [GeV]")
    ax.set_ylabel("Probabilité")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
