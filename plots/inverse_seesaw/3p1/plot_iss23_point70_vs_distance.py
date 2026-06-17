import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POINT_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_kept_points/70.txt")
CSV_PATH = Path("data/inverse_seesaw/3p1/iss23_point70_vs_distance.csv")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/iss23_point70_vs_distance.png")

DIST_MIN_KM = 0.0
DIST_MAX_KM = 1550.0
DIST_STEP_KM = 2.0
FIXED_ENERGY_GEV = 2.5

THETA12_DEG = 33.44
THETA13_DEG = 8.57
THETA23_DEG = 49.20
DELTA_CP_DEG = 230.0
DM21_3NU_EV2 = 7.42e-5
DM31_3NU_EV2 = 2.517e-3
GAUSSIAN_FILTER_ENABLED = False
SIGMAE_OVER_E = 0.05
OSCILLATION_PHASE_FACTOR = 1.267 * 2.0


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
    return np.array(rows, dtype=complex)


def load_iss23_point(path: Path):
    text = path.read_text(encoding="utf-8")
    point_id = _read_scalar(text, "point_id", int)
    pmns_pass = _read_scalar(text, "pmns_pass", int)
    eta_pass = _read_scalar(text, "eta_pass", int)
    if point_id != 70:
        raise ValueError(f"Point inattendu: {point_id}")
    if pmns_pass != 1 or eta_pass != 1:
        raise ValueError(f"Le point 70 n'est pas valide: pmns_pass={pmns_pass}, eta_pass={eta_pass}")

    return {
        "point_id": point_id,
        "pmns_pass": pmns_pass,
        "eta_pass": eta_pass,
        "dm21": _read_scalar(text, "dm21_calc_eV2"),
        "dm31": _read_scalar(text, "dm31_calc_eV2"),
        "dm41": _read_scalar(text, "dm41_calc_eV2"),
        "mixing": _read_u4x4_solver(text),
    }


def pmns_3nu() -> np.ndarray:
    t12 = math.radians(THETA12_DEG)
    t13 = math.radians(THETA13_DEG)
    t23 = math.radians(THETA23_DEG)
    delta = math.radians(DELTA_CP_DEG)

    c12, s12 = math.cos(t12), math.sin(t12)
    c13, s13 = math.cos(t13), math.sin(t13)
    c23, s23 = math.cos(t23), math.sin(t23)
    exp_pos = complex(math.cos(delta), math.sin(delta))
    exp_neg = complex(math.cos(delta), -math.sin(delta))

    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * exp_neg],
            [-s12 * c23 - c12 * s23 * s13 * exp_pos, c12 * c23 - s12 * s23 * s13 * exp_pos, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * exp_pos, -c12 * s23 - s12 * c23 * s13 * exp_pos, c23 * c13],
        ],
        dtype=complex,
    )


def probability(mixing: np.ndarray, masses2: np.ndarray, alpha: int, beta: int, energy_gev: float, baseline_km: float) -> float:
    # Meme convention de phase que probability_with_gaussian_filter_n() cote C,
    # mais sans amortissement gaussien pour visualiser les oscillations brutes.
    p = 0.0
    n_states = len(masses2)
    for i in range(n_states):
        p += abs(mixing[beta, i]) ** 2 * abs(mixing[alpha, i]) ** 2

    for i in range(n_states):
        for j in range(i):
            delta_phi = OSCILLATION_PHASE_FACTOR * (masses2[i] - masses2[j]) * baseline_km / energy_gev
            coeff = mixing[beta, i] * np.conj(mixing[alpha, i]) * np.conj(mixing[beta, j]) * mixing[alpha, j]
            damping = 1.0
            if GAUSSIAN_FILTER_ENABLED:
                sigma_delta_phi = delta_phi * SIGMAE_OVER_E
                damping = math.exp(-0.5 * sigma_delta_phi * sigma_delta_phi)
            p += 2.0 * damping * float(np.real(coeff * np.exp(-1j * delta_phi)))

    if -1e-12 < p < 0.0:
        p = 0.0
    if 1.0 < p < 1.0 + 1e-12:
        p = 1.0
    return p


def write_csv(distances, p_iss_mumu, p_iss_mue, p_3nu_mumu, p_3nu_mue, point):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "point_id",
                "energy_GeV",
                "baseline_km",
                "dm41_eV2",
                "P_mumu_iss23",
                "P_mue_iss23",
                "P_mumu_3nu",
                "P_mue_3nu",
            ]
        )
        for row in zip(distances, p_iss_mumu, p_iss_mue, p_3nu_mumu, p_3nu_mue):
            baseline, iss_mumu, iss_mue, nu3_mumu, nu3_mue = row
            writer.writerow(
                [
                    point["point_id"],
                    f"{FIXED_ENERGY_GEV:.10g}",
                    f"{baseline:.10g}",
                    f"{point['dm41']:.12e}",
                    f"{iss_mumu:.12e}",
                    f"{iss_mue:.12e}",
                    f"{nu3_mumu:.12e}",
                    f"{nu3_mue:.12e}",
                ]
            )


def plot_distance(distances, p_iss_mumu, p_iss_mue, p_3nu_mumu, p_3nu_mue, point):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

    iss_label = (
        rf"ISS(2,3) point 70, $\Delta m_{{41}}^2={point['dm41']:.3g}\,\mathrm{{eV}}^2$"
        "\n"
        rf"$\mathrm{{pmns\_pass}}={point['pmns_pass']}$, "
        rf"$\mathrm{{eta\_pass}}={point['eta_pass']}$"
    )
    nu3_label = r"3$\nu$ sans sterile"

    ax_l.plot(distances, p_iss_mumu, color="tab:red", lw=1.6, label=iss_label)
    ax_r.plot(distances, p_iss_mue, color="tab:red", lw=1.6, label=iss_label)
    ax_l.plot(distances, p_3nu_mumu, color="black", lw=1.8, ls="--", label=nu3_label)
    ax_r.plot(distances, p_3nu_mue, color="black", lw=1.8, ls="--", label=nu3_label)

    ax_l.set_title(r"$\nu_\mu \to \nu_\mu$ (disparition)")
    ax_r.set_title(r"$\nu_\mu \to \nu_e$ (apparition)")
    ax_l.set_ylabel("Probabilite")

    for ax in (ax_l, ax_r):
        ax.set_xlabel("Distance L [km]")
        ax.set_xlim(DIST_MIN_KM, DIST_MAX_KM)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle(
        rf"Point 70 ISS(2,3), $E_\nu={FIXED_ENERGY_GEV:g}\,\mathrm{{GeV}}$ - "
        "oscillations en vide sans filtre gaussien",
        y=1.02,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    point = load_iss23_point(POINT_PATH)
    distances = np.arange(DIST_MIN_KM, DIST_MAX_KM + 0.5 * DIST_STEP_KM, DIST_STEP_KM)

    iss_masses2 = np.array([0.0, point["dm21"], point["dm31"], point["dm41"]], dtype=float)
    nu3_masses2 = np.array([0.0, DM21_3NU_EV2, DM31_3NU_EV2], dtype=float)
    nu3_mixing = pmns_3nu()

    p_iss_mumu = np.array([probability(point["mixing"], iss_masses2, 1, 1, FIXED_ENERGY_GEV, L) for L in distances])
    p_iss_mue = np.array([probability(point["mixing"], iss_masses2, 1, 0, FIXED_ENERGY_GEV, L) for L in distances])
    p_3nu_mumu = np.array([probability(nu3_mixing, nu3_masses2, 1, 1, FIXED_ENERGY_GEV, L) for L in distances])
    p_3nu_mue = np.array([probability(nu3_mixing, nu3_masses2, 1, 0, FIXED_ENERGY_GEV, L) for L in distances])

    write_csv(distances, p_iss_mumu, p_iss_mue, p_3nu_mumu, p_3nu_mue, point)
    plot_distance(distances, p_iss_mumu, p_iss_mue, p_3nu_mumu, p_3nu_mue, point)

    print(f"CSV sauvegarde: {CSV_PATH}")
    print(f"Figure sauvegardee: {OUTPUT_PATH}")
    print(
        "Point 70 valide: "
        f"pmns_pass={point['pmns_pass']}, eta_pass={point['eta_pass']}, "
        f"dm41={point['dm41']:.6g} eV^2"
    )
    print(f"P_mumu ISS range: [{p_iss_mumu.min():.6g}, {p_iss_mumu.max():.6g}]")
    print(f"P_mue ISS range: [{p_iss_mue.min():.6g}, {p_iss_mue.max():.6g}]")


if __name__ == "__main__":
    main()
