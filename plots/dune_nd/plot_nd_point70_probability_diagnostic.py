"""Point-source ND probability diagnostic for ISS(2,3) point 70.

This intentionally avoids fluxes, cross sections, smearing, efficiencies and
stacked event categories.  It is meant to reveal the fast sterile oscillation
pattern before source-line and detector/analysis averaging.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


POINT_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_kept_points/70.txt")
OUT_CSV = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/nd_point70_probability_diagnostic_point_source.csv")
OUT_FIG = Path("figures/dune_nd/iss23/construct23_point70/probabilities/nd_point70_probability_diagnostic_point_source.png")

BASELINE_KM = 0.574
E_MIN_GEV = 0.5
E_MAX_GEV = 8.0
E_STEP_GEV = 0.005
PHASE_COEFF = 1.267


def read_scalar(text: str, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {key} in {POINT_PATH}")
    return cast(match.group(1).strip())


def read_u4x4_solver(text: str) -> np.ndarray:
    marker = "U4x4_solver ="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Missing U4x4_solver in {POINT_PATH}")

    rows = []
    for line in text[start + len(marker) :].splitlines():
        if not line.strip():
            continue
        if not line.lstrip().startswith("["):
            if rows:
                break
            continue
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)]
        if len(values) == 4:
            rows.append(values)
        if len(rows) == 4:
            break

    if len(rows) != 4:
        raise ValueError(f"Incomplete U4x4_solver in {POINT_PATH}")
    return np.array(rows, dtype=complex)


def load_point() -> tuple[np.ndarray, np.ndarray]:
    text = POINT_PATH.read_text(encoding="utf-8")
    point_id = read_scalar(text, "point_id", int)
    pmns_pass = read_scalar(text, "pmns_pass", int)
    eta_pass = read_scalar(text, "eta_pass", int)
    if point_id != 70 or pmns_pass != 1 or eta_pass != 1:
        raise ValueError(f"Unexpected point metadata: point={point_id}, pmns_pass={pmns_pass}, eta_pass={eta_pass}")

    masses2 = np.array(
        [
            0.0,
            read_scalar(text, "dm21_calc_eV2"),
            read_scalar(text, "dm31_calc_eV2"),
            read_scalar(text, "dm41_calc_eV2"),
        ],
        dtype=float,
    )
    return read_u4x4_solver(text), masses2


def probability_vacuum(mixing: np.ndarray, masses2: np.ndarray, alpha: int, beta: int, energy_gev: float) -> float:
    p = 1.0 if alpha == beta else 0.0
    n = len(masses2)
    for i in range(n):
        for j in range(i + 1, n):
            phase = PHASE_COEFF * (masses2[j] - masses2[i]) * BASELINE_KM / energy_gev
            amp = mixing[alpha, i] * np.conjugate(mixing[beta, i]) * np.conjugate(mixing[alpha, j]) * mixing[beta, j]
            p -= 4.0 * float(np.real(amp)) * np.sin(phase) ** 2
            p += 2.0 * float(np.imag(amp)) * np.sin(2.0 * phase)
    return float(np.clip(p, 0.0, 1.0))


def build_rows() -> list[dict[str, float]]:
    u4, m4 = load_point()
    u3 = u4[:3, :3]
    m3 = m4[:3]
    energies = np.arange(E_MIN_GEV, E_MAX_GEV + 0.5 * E_STEP_GEV, E_STEP_GEV)

    rows: list[dict[str, float]] = []
    channels = [
        ("mue", 1, 0),
        ("mumu", 1, 1),
    ]
    for energy in energies:
        for name, alpha, beta in channels:
            p3 = probability_vacuum(u3, m3, alpha, beta, float(energy))
            piss = probability_vacuum(u4, m4, alpha, beta, float(energy))
            rows.append(
                {
                    "channel": name,
                    "energy_GeV": float(energy),
                    "P_3nu_active_point70": p3,
                    "P_ISS23_point70": piss,
                    "delta_P": piss - p3,
                }
            )
    return rows


def write_csv(rows: list[dict[str, float]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "channel",
                "energy_GeV",
                "P_3nu_active_point70",
                "P_ISS23_point70",
                "delta_P",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]]) -> None:
    labels = {
        "mue": r"$P_{\mu e}$",
        "mumu": r"$P_{\mu\mu}$",
    }
    fig = plt.figure(figsize=(10.5, 6.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.08, wspace=0.28)

    for col, channel in enumerate(["mue", "mumu"]):
        ax = fig.add_subplot(grid[0, col])
        rax = fig.add_subplot(grid[1, col], sharex=ax)
        data = [row for row in rows if row["channel"] == channel]
        energy = np.array([row["energy_GeV"] for row in data])
        p3 = np.array([row["P_3nu_active_point70"] for row in data])
        piss = np.array([row["P_ISS23_point70"] for row in data])
        delta = piss - p3

        ax.plot(energy, p3, color="black", lw=1.8, label=r"$3\nu$ active point 70")
        ax.plot(energy, piss, color="#d62728", lw=1.0, label=r"ISS(2,3) point 70")
        rax.plot(energy, delta, color="#d62728", lw=0.9)
        rax.axhline(0.0, color="black", lw=0.7)

        ax.set_title(labels[channel], fontsize=13)
        ax.set_ylabel("Probability")
        ax.set_xlim(E_MIN_GEV, E_MAX_GEV)
        ax.tick_params(labelbottom=False, direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        ax.grid(alpha=0.18)

        rax.set_xlabel(r"True neutrino energy $E_\nu$ [GeV]")
        rax.set_ylabel(r"$P_{\rm ISS}-P_{3\nu}$")
        rax.tick_params(direction="in", top=True, right=True)
        rax.minorticks_on()
        rax.tick_params(which="minor", direction="in", top=True, right=True)
        rax.grid(alpha=0.18)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.suptitle(
        rf"ND point-source probability diagnostic, $L={BASELINE_KM * 1000:.0f}$ m, "
        rf"$\Delta E={E_STEP_GEV:g}$ GeV, no flux/xsec/smearing",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=240)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    plot(rows)
    print(f"CSV sauvegarde: {OUT_CSV.resolve()}")
    print(f"Figure sauvegardee: {OUT_FIG.resolve()}")


if __name__ == "__main__":
    main()
