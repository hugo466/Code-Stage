import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_pmns_filter_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_pmns_filter_parameter_correlations.png")
MIN_POINT_ID = 1
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_scalar(line: str) -> float:
    return float(line.split("=", 1)[1].strip())


def parse_all_floats(text: str):
    rhs = text.split("=", 1)[1] if "=" in text else text
    return [float(x) for x in FLOAT_PATTERN.findall(rhs)]


def load_parameter_table(data_dir: Path, min_point_id: int):
    labels = [
        "mu00", "muH11", "muH12", "muH21", "muH22", "muH01", "muH02",
        "mD11", "mD12", "mD21", "mD22", "mD31", "mD32",
        "M11", "M12", "M21", "M22",
    ]

    rows = []
    selected_files = sorted(
        [p for p in data_dir.glob("*.txt") if p.stem.isdigit() and int(p.stem) >= min_point_id],
        key=lambda p: int(p.stem),
    )

    for point_file in selected_files:
        row = {label: None for label in labels}
        lines = point_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("mu00_eV"):
                row["mu00"] = parse_scalar(line)
            elif line.startswith("M_2x2_GeV"):
                values = parse_all_floats(line)
                row["M11"], row["M12"], row["M21"], row["M22"] = values
            elif line.startswith("mD_3x2_GeV"):
                values = parse_all_floats(line)
                row["mD11"], row["mD12"], row["mD21"], row["mD22"], row["mD31"], row["mD32"] = values
            elif line.startswith("mu_H_2x2_eV"):
                values = parse_all_floats(line)
                row["muH11"], row["muH12"], row["muH21"], row["muH22"] = values
            elif line.startswith("mu_H0_2x1_eV"):
                values = parse_all_floats(line)
                row["muH01"], row["muH02"] = values

        if all(row[label] is not None for label in labels):
            rows.append([row[label] for label in labels])

    if not rows:
        raise RuntimeError(f"Aucun point exploitable trouvé à partir de l'identifiant {min_point_id}.")

    return selected_files, labels, np.array(rows, dtype=float)


def print_top_correlations(labels, corr, top_n=15):
    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(corr[i, j]), corr[i, j], labels[i], labels[j]))

    pairs.sort(reverse=True)
    # print(f"Top {top_n} corrélations |r| :")
    # for _, signed_r, a, b in pairs[:top_n]:
    #     print(f"  {a:>5s} vs {b:<5s} : r = {signed_r:+.4f}")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable. Vérifie inverse_kept_points_dir dans la config: {DATA_DIR}"
        )

    selected_files, labels, values = load_parameter_table(DATA_DIR, MIN_POINT_ID)
    point_count = len(selected_files)
    corr = np.corrcoef(values, rowvar=False)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if abs(corr[i, j]) > 0.55 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title(rf"Matrice de corrélation des paramètres ({point_count} points)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Corrélation")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    print_top_correlations(labels, corr, top_n=15)
    plt.close(fig)


if __name__ == "__main__":
    main()
