from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path("figures/oscillations/3p2/formule_ana_3p2.png")

L_KM = 0.574
E_GEV = np.linspace(0.5, 6.0, 500)
FACTOR = 2.534
SIGMA_PHASE_REL = 0.05

DM41 = 1000000.0
DM54_LIST = [0.1, 1.0, 10.0, 100.0]

UE4 = 0.316
UMU4 = UE4
UE5 = UE4
UMU5 = UE4


def gaussian_weighted_sin2(phase: np.ndarray, sigma_rel: float = SIGMA_PHASE_REL) -> np.ndarray:
    raw = np.sin(phase / 2) ** 2
    sigma_phi = np.abs(phase) * sigma_rel
    weight = 1.0 - np.exp(-0.5 * sigma_phi**2)
    return (1.0 - weight) * raw + weight * 0.5


def p_mumu(energy_gev: np.ndarray, dm41: float, dm54: float) -> np.ndarray:
    phi41 = FACTOR * dm41 * L_KM / energy_gev
    phi51 = FACTOR * (dm54 + dm41) * L_KM / energy_gev
    phi54 = FACTOR * dm54 * L_KM / energy_gev
    prob = (
        1
        - 4 * UMU4**2 * (1 - UMU4**2) * gaussian_weighted_sin2(phi41)
        - 4 * UMU5**2 * (1 - UMU5**2) * gaussian_weighted_sin2(phi51)
        + 4 * UMU4**2 * UMU5**2 * (gaussian_weighted_sin2(phi41) + gaussian_weighted_sin2(phi51) - gaussian_weighted_sin2(phi54))
    )
    return np.clip(prob, 0.0, 1.0)


def p_mue(energy_gev: np.ndarray, dm41: float, dm54: float) -> np.ndarray:
    phi54 = FACTOR * dm54 * L_KM / energy_gev
    prob = (
        4 * UE4**2 * UMU4**2 * 0.5
        + 4 * UE5**2 * UMU5**2 * 0.5
        + 4 * UE4**4 * (1 - gaussian_weighted_sin2(phi54))
    )
    return np.clip(prob, 0.0, 1.0)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax_left, ax_right = axes

    for dm54 in DM54_LIST:
        label = rf"$\Delta m_{{54}}^2 = {dm54:g}\,\mathrm{{eV}}^2$"
        ax_left.plot(E_GEV, p_mumu(E_GEV, DM41, dm54), lw=2, label=label)
        ax_right.plot(E_GEV, p_mue(E_GEV, DM41, dm54), lw=2, label=label)

    ax_left.set_title(r"$\nu_\mu \to \nu_\mu$ (3+2, disparition)")
    ax_right.set_title(r"$\nu_\mu \to \nu_e$ (3+2, apparition)")

    ax_left.set_ylabel("Probabilité")
    ax_left.set_xlabel("Énergie [GeV]")
    ax_right.set_xlabel("Énergie [GeV]")

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
