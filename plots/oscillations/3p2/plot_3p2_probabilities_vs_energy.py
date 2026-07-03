import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/oscillations/3p2/nd_3p2_probabilities.csv")
OUTPUT_PATH = Path("figures/oscillations/3p2/nd_3p2_probabilities_vs_energy.png")

TARGET_BASELINE_KM = 0.574

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
            baseline_km = float(row.get("baseline_km", TARGET_BASELINE_KM))
            if abs(baseline_km - TARGET_BASELINE_KM) > 1e-12:
                continue

            dm41 = float(row["dm41_eV2"])
            dm54 = float(row["dm54_eV2"])
            dm51 = float(row["dm51_eV2"])
            key = (dm41, dm54, dm51)

            grouped[key]["E"].append(float(row["energy_GeV"]))
            grouped[key]["Pmumu"].append(float(row["P_mumu_disappearance"]))
            grouped[key]["Pmue"].append(float(row["P_mue_appearance"]))

    return dict(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])))


def parse_args():
    parser = argparse.ArgumentParser(description="Trace les probabilites 3+2 ND en fonction de l'energie.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dm41", type=float, default=None, help="Valeur de Delta m41^2 a selectionner dans le CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"CSV introuvable: {args.input}. Execute d'abord le programme C pour le generer.")

    data = load_data(args.input)
    if not data:
        raise RuntimeError("Le CSV 3+2 est vide.")

    dm41_values = sorted({key[0] for key in data.keys()})
    target_dm41 = args.dm41 if args.dm41 is not None else dm41_values[0]

    filtered = {
        key: values
        for key, values in data.items()
        if abs(key[0] - target_dm41) <= max(1.0e-12, 1.0e-12 * abs(target_dm41))
    }
    if not filtered:
        raise RuntimeError(f"Aucune courbe trouvee pour dm41={target_dm41:g} eV^2 dans {args.input}.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    ax_left, ax_right = axes

    for (dm41, dm54, dm51), values in filtered.items():
        label = rf"$\Delta m_{{54}}^2 = {dm54:g}\,\mathrm{{eV}}^2$"
        color = COLORS.get(dm54, None)
        ax_left.plot(values["E"], values["Pmumu"], color=color, lw=2, label=label)
        ax_right.plot(values["E"], values["Pmue"], color=color, lw=2, label=label)

    ax_left.set_title(r"$\nu_\mu \to \nu_\mu$ (3+2, disparition)")
    ax_right.set_title(r"$\nu_\mu \to \nu_e$ (3+2, apparition)")

    ax_left.set_ylabel("Probabilite")
    ax_left.set_xlabel("Energie [GeV]")
    ax_right.set_xlabel("Energie [GeV]")

    ax_left.grid(alpha=0.25)
    ax_right.grid(alpha=0.25)

    ax_left.legend(fontsize=8)
    ax_right.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Figure sauvegardee: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
