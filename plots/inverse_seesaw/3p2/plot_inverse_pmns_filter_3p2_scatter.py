from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_pmns_filter_3p2_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p2/inverse_pmns_filter_3p2_dm21_vs_dm32.png")

# PDG / NuFIT 5.3 best-fit values (normal ordering)
DM21_BF   = 7.53e-5   # eV²
DM21_LOW  = 7.53e-5 - 0.18e-5
DM21_HIGH = 7.53e-5 + 0.18e-5
DM32_BF   = 2.453e-3  # eV²
DM32_LOW  = 2.453e-3 - 0.030e-3
DM32_HIGH = 2.453e-3 + 0.026e-3


def apply_log_jitter(values, jitter_decades=0.008, seed=12345):
    rng = random.Random(seed)
    return [v * (10.0 ** rng.uniform(-jitter_decades, jitter_decades)) for v in values]

def load_points(path: Path):
    dm21, dm32 = [], []
    eta_dm21, eta_dm32 = [], []

    for point_file in sorted(path.glob("*.txt"), key=lambda p: int(p.stem)):
        dm21_value = None
        dm31_value = None
        eta_pass = 0

        for line in point_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("dm21_eV2"):
                dm21_value = float(line.split("=", 1)[1].strip())
            elif line.startswith("dm31_eV2"):
                dm31_value = float(line.split("=", 1)[1].strip())
            elif line.startswith("eta_pass"):
                eta_pass = int(float(line.split("=", 1)[1].strip()))

        if dm21_value is None or dm31_value is None:
            continue

        dm32_value = dm31_value - dm21_value

        if dm21_value > 0.0 and dm32_value > 0.0:
            dm21.append(dm21_value)
            dm32.append(dm32_value)
            if eta_pass == 1:
                eta_dm21.append(dm21_value)
                eta_dm32.append(dm32_value)

    return dm21, dm32, eta_dm21, eta_dm32


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    dm21, dm32, eta_dm21, eta_dm32 = load_points(DATA_DIR)
    if not dm21:
        raise RuntimeError("Aucun point valide.")

    dm21_plot = apply_log_jitter(dm21, seed=111)
    dm32_plot = apply_log_jitter(dm32, seed=222)
    eta_dm21_plot = apply_log_jitter(eta_dm21, seed=333)
    eta_dm32_plot = apply_log_jitter(eta_dm32, seed=444)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(dm32_plot, dm21_plot, s=10, alpha=0.25, color="#1f77b4", label="PMNS")
    if eta_dm21_plot and eta_dm32_plot:
        ax.scatter(eta_dm32_plot, eta_dm21_plot, s=10, alpha=0.55, color="red", label=r"PMNS + $\eta$")

    # ── Theoretical lines (PDG/NuFIT 5.3 best fit) — même style que 3p1 ──────
    ax.axhline(DM21_BF, color="black", lw=0.7, alpha=0.6)
    ax.axvline(DM32_BF, color="black", lw=0.7, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta m_{32}^2\;[\mathrm{eV}^2]$")
    ax.set_ylabel(r"$\Delta m_{21}^2\;[\mathrm{eV}^2]$")
    ax.set_title(rf"ISS 3+2 — PMNS ({len(dm21)}) et PMNS+η ({len(eta_dm21)})")
    ax.grid(alpha=0.25)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
