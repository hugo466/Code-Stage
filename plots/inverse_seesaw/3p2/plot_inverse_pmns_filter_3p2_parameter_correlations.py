import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_pmns_filter_3p2_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p2/inverse_pmns_filter_3p2_parameter_correlations.png")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

ALL_KEYS = [
    "M11", "M12", "M21", "M22",
    "mD11", "mD12", "mD21", "mD22", "mD31", "mD32",
    "mu00_11", "mu00_12", "mu00_22",
    "muH_11", "muH_12", "muH_22",
    "muH0_11", "muH0_12", "muH0_21", "muH0_22",
]

TICK_LABELS = [
    r"$M_{11}$", r"$M_{12}$", r"$M_{21}$", r"$M_{22}$",
    r"$m_{D,11}$", r"$m_{D,12}$", r"$m_{D,21}$", r"$m_{D,22}$",
    r"$m_{D,31}$", r"$m_{D,32}$",
    r"$\mu_{0,11}$", r"$\mu_{0,12}$", r"$\mu_{0,22}$",
    r"$\mu_{H,11}$", r"$\mu_{H,12}$", r"$\mu_{H,22}$",
    r"$\mu_{H0,11}$", r"$\mu_{H0,12}$", r"$\mu_{H0,21}$", r"$\mu_{H0,22}$",
]


def parse_all_floats(text: str):
    rhs = text.split("=", 1)[1] if "=" in text else text
    return [float(x) for x in FLOAT_PATTERN.findall(rhs)]


def load_table(data_dir: Path):
    rows = []

    for point_file in sorted(data_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        row = {k: None for k in ALL_KEYS}
        for line in point_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("M_2x2_GeV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    row["M11"] = abs(v[0])
                    row["M12"] = abs(v[1])
                    row["M21"] = abs(v[2])
                    row["M22"] = abs(v[3])
            elif line.startswith("mD_3x2_GeV"):
                v = parse_all_floats(line)
                if len(v) >= 6:
                    row["mD11"] = abs(v[0])
                    row["mD12"] = abs(v[1])
                    row["mD21"] = abs(v[2])
                    row["mD22"] = abs(v[3])
                    row["mD31"] = abs(v[4])
                    row["mD32"] = abs(v[5])
            elif line.startswith("mu00_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    row["mu00_11"] = abs(v[0])
                    row["mu00_12"] = abs(v[1])
                    row["mu00_22"] = abs(v[3])
            elif line.startswith("mu_H0_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    row["muH0_11"] = abs(v[0])
                    row["muH0_12"] = abs(v[1])
                    row["muH0_21"] = abs(v[2])
                    row["muH0_22"] = abs(v[3])
            elif line.startswith("mu_H_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    row["muH_11"] = abs(v[0])
                    row["muH_12"] = abs(v[1])
                    row["muH_22"] = abs(v[3])

        if all(row[k] is not None for k in ALL_KEYS):
            rows.append([row[k] for k in ALL_KEYS])

    if not rows:
        raise RuntimeError("Aucun point exploitable.")

    return np.array(rows, dtype=float)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    values = load_table(DATA_DIR)
    corr = np.corrcoef(values, rowvar=False)
    n = len(ALL_KEYS)

    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(TICK_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(TICK_LABELS, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr[i, j]) > 0.55 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("ISS 3+2 — Matrice de corrélation", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Corrélation")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
