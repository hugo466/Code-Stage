import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("data/oscillations/3p2/nd_3p2_probabilities.csv")
OUTPUT_PATH = Path("figures/oscillations/3p2/nd_3p2_probabilities_vs_LE.png")

TARGET_BASELINE_KM = 0.574

COLORS = {
    0.1: "green",
    1.0: "red",
    10.0: "blue",
    100.0: "orange",
}


def load_data(path: Path):
    grouped = defaultdict(lambda: {"LE": [], "Pmumu": [], "Pmue": []})

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            baseline_km = float(row.get("baseline_km", 0.574))
            if abs(baseline_km - TARGET_BASELINE_KM) > 1e-12:
                continue

            energy_g = float(row["energy_GeV"])
            dm41 = float(row["dm41_eV2"])
            dm54 = float(row["dm54_eV2"])
            dm51 = float(row["dm51_eV2"])
            key = (dm41, dm54, dm51)

            grouped[key]["LE"].append(baseline_km / energy_g)
            grouped[key]["Pmumu"].append(float(row["P_mumu_disappearance"]))
            grouped[key]["Pmue"].append(float(row["P_mue_appearance"]))

    return dict(sorted(grouped.items(), key=lambda item: item[0][1]))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "CSV introuvable. Exécute d'abord le programme C pour générer data/oscillations/3p2/nd_3p2_probabilities.csv"
        )

    data = load_data(DATA_PATH)
    if not data:
        raise RuntimeError("Le CSV 3+2 est vide.")

    dm41_values = sorted({key[0] for key in data.keys()})
    target_dm41 = dm41_values[0]

    filtered = {
        key: values
        for key, values in data.items()
        if abs(key[0] - target_dm41) < 1e-15
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    ax_left, ax_right = axes

    for (dm41, dm54, dm51), values in filtered.items():
        x = values["LE"]
        order = sorted(range(len(x)), key=lambda i: x[i])
        xs = [x[i] for i in order]
        y_mumu = [values["Pmumu"][i] for i in order]
        y_mue = [values["Pmue"][i] for i in order]

        label = rf"$\Delta m_{{54}}^2 = {dm54:g}\,\mathrm{{eV}}^2$"
        color = COLORS.get(dm54, None)
        ax_left.plot(xs, y_mumu, color=color, lw=2, label=label)
        ax_right.plot(xs, y_mue, color=color, lw=2, label=label)

    ax_left.set_title(r"$\nu_\mu \to \nu_\mu$ vs $L/E$")
    ax_right.set_title(r"$\nu_\mu \to \nu_e$ vs $L/E$")
    ax_left.set_ylabel("Probabilité")
    ax_left.set_xlabel(r"$L/E$ [km/GeV]")
    ax_right.set_xlabel(r"$L/E$ [km/GeV]")

    ax_left.grid(alpha=0.25)
    ax_right.grid(alpha=0.25)

    ax_left.legend(fontsize=8)
    ax_right.legend(fontsize=8)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
