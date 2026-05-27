from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_pmns_filter_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_pmns_filter_dm21_vs_dm32.png")

# Valeurs experimentales (NuFIT 6.0, NO, valeurs centrales)
DM21_EXP_EV2 = 7.42e-5
DM31_EXP_EV2 = 2.517e-3
DM32_EXP_EV2 = DM31_EXP_EV2 - DM21_EXP_EV2


def load_points(path: Path):
    dm21 = []
    dm32 = []
    eta_dm21 = []
    eta_dm32 = []

    for point_file in sorted(path.glob("*.txt"), key=lambda p: int(p.stem)):
        dm21_value = None
        dm32_value = None
        eta_pass = 0

        with point_file.open("r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("dm21_eV2"):
                    dm21_value = float(line.split("=", 1)[1].strip())
                elif line.startswith("dm32_eV2"):
                    dm32_value = float(line.split("=", 1)[1].strip())
                elif line.startswith("eta_pass"):
                    eta_pass = int(float(line.split("=", 1)[1].strip()))

        if dm21_value is None or dm32_value is None:
            continue

        if dm21_value > 0.0 and dm32_value > 0.0:
            dm21.append(dm21_value)
            dm32.append(dm32_value)
            if eta_pass == 1:
                eta_dm21.append(dm21_value)
                eta_dm32.append(dm32_value)

    return dm21, dm32, eta_dm21, eta_dm32


def apply_log_jitter(values, jitter_decades=0.008, seed=12345):
    rng = random.Random(seed)
    jittered = []
    for value in values:
        factor = 10.0 ** rng.uniform(-jitter_decades, jitter_decades)
        jittered.append(value * factor)
    return jittered


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable. Vérifie inverse_kept_points_dir dans la config: {DATA_DIR}"
        )

    dm21, dm32, eta_dm21, eta_dm32 = load_points(DATA_DIR)

    dm21_plot = apply_log_jitter(dm21, jitter_decades=0.008, seed=12345)
    dm32_plot = apply_log_jitter(dm32, jitter_decades=0.008, seed=54321)
    eta_dm21_plot = apply_log_jitter(eta_dm21, jitter_decades=0.008, seed=22222)
    eta_dm32_plot = apply_log_jitter(eta_dm32, jitter_decades=0.008, seed=33333)
    unique_count = len(set(zip(dm32, dm21)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    ax.scatter(dm32_plot, dm21_plot, s=10, alpha=0.25, color="#1f77b4")
    if eta_dm21_plot and eta_dm32_plot:
        ax.scatter(eta_dm32_plot, eta_dm21_plot, s=10, alpha=0.5, color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(DM32_EXP_EV2, color="black", lw=0.7, alpha=0.6)
    ax.axhline(DM21_EXP_EV2, color="black", lw=0.7, alpha=0.6)

    eta_count = len(eta_dm21)
    ax.set_title(rf"ISS 3$+$1 neutrinos légers: PMNS ({unique_count}) et PMNS$+$eta ({eta_count})")
    ax.set_xlabel(r"$\Delta m_{32}^2\;[\mathrm{eV}^2]$")
    ax.set_ylabel(r"$\Delta m_{21}^2\;[\mathrm{eV}^2]$")
    ax.grid(alpha=0.25)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
