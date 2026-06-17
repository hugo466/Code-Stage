import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POINT_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_kept_points/70.txt")
CSV_PATH = Path("data/inverse_seesaw/3p1/iss23_point70_dune_probabilities.csv")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/iss23_point70_dune_probabilities.png")

BASELINE_KM = 0.574
ENERGY_MIN_GEV = 0.5
ENERGY_MAX_GEV = 6.0
ENERGY_STEP_GEV = 0.01


def _read_scalar(text: str, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Champ manquant dans {POINT_PATH}: {key}")
    return cast(match.group(1).strip())


def _read_u4x4_solver(text: str) -> np.ndarray:
    marker = "U4x4_solver ="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Matrice U4x4_solver introuvable dans {POINT_PATH}")

    rows = []
    for line in text[start + len(marker) :].splitlines():
        if not line.strip():
            continue
        if not line.lstrip().startswith("["):
            if rows:
                break
            continue
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)]
        if len(values) != 4:
            raise ValueError(f"Ligne U4x4_solver invalide: {line}")
        rows.append(values)
        if len(rows) == 4:
            break

    if len(rows) != 4:
        raise ValueError(f"Matrice U4x4_solver incomplete dans {POINT_PATH}")
    return np.array(rows, dtype=float)


def load_point(path: Path):
    text = path.read_text(encoding="utf-8")
    point_id = _read_scalar(text, "point_id", int)
    pmns_pass = _read_scalar(text, "pmns_pass", int)
    eta_pass = _read_scalar(text, "eta_pass", int)
    if point_id != 70:
        raise ValueError(f"Point inattendu: {point_id}")
    if pmns_pass != 1 or eta_pass != 1:
        raise ValueError(f"Le point 70 n'est pas valide: pmns_pass={pmns_pass}, eta_pass={eta_pass}")

    dm21 = _read_scalar(text, "dm21_calc_eV2")
    dm31 = _read_scalar(text, "dm31_calc_eV2")
    dm41 = _read_scalar(text, "dm41_calc_eV2")
    mixing = _read_u4x4_solver(text)
    masses2 = np.array([0.0, dm21, dm31, dm41], dtype=float)
    return {
        "point_id": point_id,
        "pmns_pass": pmns_pass,
        "eta_pass": eta_pass,
        "dm21": dm21,
        "dm31": dm31,
        "dm41": dm41,
        "mixing": mixing,
        "masses2": masses2,
    }


def probability_vacuum(point, alpha: int, beta: int, energy_gev: float, baseline_km: float) -> float:
    mixing = point["mixing"]
    masses2 = point["masses2"]
    p = 1.0 if alpha == beta else 0.0

    for i in range(4):
        for j in range(i + 1, 4):
            phase = 1.267 * (masses2[j] - masses2[i]) * baseline_km / energy_gev
            amp = mixing[alpha, i] * mixing[beta, i] * mixing[alpha, j] * mixing[beta, j]
            p -= 4.0 * amp * math.sin(phase) ** 2

    if -1e-12 < p < 0.0:
        p = 0.0
    if 1.0 < p < 1.0 + 1e-12:
        p = 1.0
    return p


def write_csv(energies, pmumu, pmue, point):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "point_id",
                "baseline_km",
                "energy_GeV",
                "dm41_eV2",
                "P_mumu_disappearance",
                "P_mue_appearance",
            ]
        )
        for energy, p_mumu, p_mue in zip(energies, pmumu, pmue):
            writer.writerow(
                [
                    point["point_id"],
                    f"{BASELINE_KM:.10g}",
                    f"{energy:.10g}",
                    f"{point['dm41']:.12e}",
                    f"{p_mumu:.12e}",
                    f"{p_mue:.12e}",
                ]
            )


def plot_probabilities(energies, pmumu, pmue, point):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax_left, ax_right = axes

    label = (
        rf"point 70, $\Delta m_{{41}}^2={point['dm41']:.3g}\,\mathrm{{eV}}^2$"
        "\n"
        rf"$\mathrm{{pmns\_pass}}={point['pmns_pass']}$, "
        rf"$\mathrm{{eta\_pass}}={point['eta_pass']}$"
    )

    ax_left.plot(energies, pmumu, color="tab:red", lw=2.0, label=label)
    ax_right.plot(energies, pmue, color="tab:red", lw=2.0, label=label)

    ax_left.set_title(r"ISS(2,3): $\nu_\mu \to \nu_\mu$ (disparition)")
    ax_right.set_title(r"ISS(2,3): $\nu_\mu \to \nu_e$ (apparition)")
    ax_left.set_ylabel("Probabilite")
    ax_left.set_xlabel("Energie [GeV]")
    ax_right.set_xlabel("Energie [GeV]")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(ENERGY_MIN_GEV, ENERGY_MAX_GEV)
        ax.legend(fontsize=8.5)

    fig.suptitle(rf"DUNE ND, $L={BASELINE_KM:.3f}\,\mathrm{{km}}$ - probabilites exactes en vide", y=1.02)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    point = load_point(POINT_PATH)
    energies = np.arange(ENERGY_MIN_GEV, ENERGY_MAX_GEV + 0.5 * ENERGY_STEP_GEV, ENERGY_STEP_GEV)
    pmumu = np.array([probability_vacuum(point, 1, 1, energy, BASELINE_KM) for energy in energies])
    pmue = np.array([probability_vacuum(point, 1, 0, energy, BASELINE_KM) for energy in energies])

    write_csv(energies, pmumu, pmue, point)
    plot_probabilities(energies, pmumu, pmue, point)

    print(f"CSV sauvegarde: {CSV_PATH}")
    print(f"Figure sauvegardee: {OUTPUT_PATH}")
    print(
        "Point 70 valide: "
        f"pmns_pass={point['pmns_pass']}, eta_pass={point['eta_pass']}, "
        f"dm41={point['dm41']:.6g} eV^2"
    )
    print(f"P_mumu range: [{pmumu.min():.6g}, {pmumu.max():.6g}]")
    print(f"P_mue range: [{pmue.min():.6g}, {pmue.max():.6g}]")


if __name__ == "__main__":
    main()
