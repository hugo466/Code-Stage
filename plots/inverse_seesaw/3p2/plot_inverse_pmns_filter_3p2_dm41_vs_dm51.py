"""Scatter plot: dm41^2 vs dm51^2 — same style as dm21_vs_dm32 (kept_points files)."""
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_pmns_filter_3p2_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p2/inverse_pmns_filter_3p2_dm41_vs_dm51.png")


def apply_log_jitter(values, jitter_decades=0.008, seed=0):
    rng = random.Random(seed)
    return [v * (10.0 ** rng.uniform(-jitter_decades, jitter_decades)) for v in values]


def load_points(path: Path):
    dm41, dm51 = [], []
    eta_dm41, eta_dm51 = [], []

    for point_file in sorted(path.glob("*.txt"), key=lambda p: int(p.stem)):
        dm41_value = None
        dm51_value = None
        eta_pass = 0

        for line in point_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("dm41_eV2"):
                dm41_value = float(line.split("=", 1)[1].strip())
            elif line.startswith("dm51_eV2"):
                dm51_value = float(line.split("=", 1)[1].strip())
            elif line.startswith("eta_pass"):
                eta_pass = int(float(line.split("=", 1)[1].strip()))

        if dm41_value is None or dm51_value is None:
            continue
        if dm41_value <= 0.0 or dm51_value <= 0.0:
            continue

        dm41.append(dm41_value)
        dm51.append(dm51_value)
        if eta_pass == 1:
            eta_dm41.append(dm41_value)
            eta_dm51.append(dm51_value)

    return dm41, dm51, eta_dm41, eta_dm51


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    dm41, dm51, eta_dm41, eta_dm51 = load_points(DATA_DIR)
    if not dm41:
        raise RuntimeError("Aucun point valide.")

    dm41_plot     = apply_log_jitter(dm41,     seed=111)
    dm51_plot     = apply_log_jitter(dm51,     seed=222)
    eta_dm41_plot = apply_log_jitter(eta_dm41, seed=333)
    eta_dm51_plot = apply_log_jitter(eta_dm51, seed=444)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(dm41_plot, dm51_plot, s=10, alpha=0.25, color="#1f77b4")
    if eta_dm41_plot:
        ax.scatter(eta_dm41_plot, eta_dm51_plot, s=10, alpha=0.55, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta m_{41}^2\;[\mathrm{eV}^2]$")
    ax.set_ylabel(r"$\Delta m_{51}^2\;[\mathrm{eV}^2]$")
    ax.set_title(rf"ISS(2,3+2) — PMNS ({len(dm41)}) et PMNS+η ({len(eta_dm41)})")
    ax.grid(alpha=0.25)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
