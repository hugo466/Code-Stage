"""True-energy FD oscillation probabilities for the Fig. 4 setup.

This plot uses no flux, no cross section, no smearing and no efficiency.
It compares:
  - 3nu probabilities built from the active 3x3 block of the chosen ISS(2,3) point,
  - the same 3nu probabilities with the C constant-density matter convention,
  - ISS(2,3) point-70 probabilities with the same C matter convention.
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
OUT_CSV = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fd_true_energy_probabilities_matter.csv")
OUT_FIG = Path("figures/dune_fd/point_70/probabilities/fd_true_energy_probabilities_matter.png")

BASELINE_KM = 1284.9
RHO_G_CM3 = 2.848
ACC_COEFF = 7.6e-5
ANC_OVER_ACC = 0.5
PHASE_FACTOR = 2.0 * 1.267

E_MIN_GEV = 0.5
E_MAX_GEV = 8.0
N_ENERGY = 900

def c_matter_probability(
    mixing: np.ndarray,
    masses2: np.ndarray,
    alpha: int,
    beta: int,
    energy_gev: float,
    anti: bool,
    sterile_nc: bool,
) -> float:
    """Python mirror of the C constant-density diagonalization convention."""
    u = np.conjugate(mixing) if anti else mixing
    h = u @ np.diag(masses2) @ np.conjugate(u).T
    sign = -1.0 if anti else 1.0
    a_cc = ACC_COEFF * RHO_G_CM3 * energy_gev
    h[0, 0] += sign * a_cc
    if sterile_nc and len(masses2) > 3:
        a_nc = ANC_OVER_ACC * a_cc
        for sterile in range(3, len(masses2)):
            h[sterile, sterile] += sign * a_nc

    eigvals, eigvecs = np.linalg.eigh(h)
    phases = np.exp(-1j * PHASE_FACTOR * eigvals * BASELINE_KM / energy_gev)
    amp = eigvecs @ np.diag(phases) @ np.conjugate(eigvecs).T
    return float(np.clip(abs(amp[beta, alpha]) ** 2, 0.0, 1.0))


def read_scalar(text: str, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing field in {POINT_PATH}: {key}")
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
        if len(values) != 4:
            raise ValueError(f"Invalid U4x4_solver row: {line}")
        rows.append(values)
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError(f"Incomplete U4x4_solver in {POINT_PATH}")
    return np.array(rows, dtype=complex)


def load_iss_point() -> tuple[np.ndarray, np.ndarray]:
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


def build_table() -> list[dict[str, float]]:
    energies = np.linspace(E_MIN_GEV, E_MAX_GEV, N_ENERGY)
    u_iss, m_iss = load_iss_point()
    u3 = u_iss[:3, :3]
    m3 = m_iss[:3]

    channels = [
        ("mue", 1, 0, False),
        ("muebar", 1, 0, True),
        ("mumu", 1, 1, False),
        ("mumubar", 1, 1, True),
    ]

    rows: list[dict[str, float]] = []
    for energy in energies:
        for name, alpha, beta, anti in channels:
            rows.append(
                {
                    "channel": name,
                    "energy_GeV": float(energy),
                    "P_3nu_point70_active": c_matter_probability(u3, m3, alpha, beta, float(energy), anti, sterile_nc=False),
                    "P_3nu_C_matter": c_matter_probability(u3, m3, alpha, beta, float(energy), anti, sterile_nc=False),
                    "P_ISS_C_matter": c_matter_probability(u_iss, m_iss, alpha, beta, float(energy), anti, sterile_nc=True),
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
                "P_3nu_point70_active",
                "P_3nu_C_matter",
                "P_ISS_C_matter",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]]) -> None:
    labels = {
        "mue": r"$P_{\mu e}$",
        "muebar": r"$P_{\bar{\mu}\bar{e}}$",
        "mumu": r"$P_{\mu\mu}$",
        "mumubar": r"$P_{\bar{\mu}\bar{\mu}}$",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), sharex=True)
    for ax, channel in zip(axes.flat, ["mue", "muebar", "mumu", "mumubar"]):
        data = [row for row in rows if row["channel"] == channel]
        energy = np.array([row["energy_GeV"] for row in data])
        ax.plot(energy, [row["P_3nu_point70_active"] for row in data], color="black", lw=1.9, label=r"$3\nu$ active block point 70")
        ax.plot(energy, [row["P_3nu_C_matter"] for row in data], color="#1f77b4", lw=1.4, ls="--", label=r"$3\nu$ C matter")
        ax.plot(energy, [row["P_ISS_C_matter"] for row in data], color="#d62728", lw=1.5, ls="-.", label=r"ISS(2,3) C matter point 70")
        ax.set_title(labels[channel], fontsize=12)
        ax.set_xlim(E_MIN_GEV, E_MAX_GEV)
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"True neutrino energy $E_\nu$ [GeV]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        rf"FD true-energy probabilities, $L={BASELINE_KM:g}$ km, "
        rf"$\rho={RHO_G_CM3:g}$ g/cm$^3$, $A_{{CC}}=7.6\times10^{{-5}}\rho E$ eV$^2$",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = build_table()
    write_csv(rows)
    plot(rows)
    print(f"CSV sauvegarde: {OUT_CSV.resolve()}")
    print(f"Figure sauvegardee: {OUT_FIG.resolve()}")


if __name__ == "__main__":
    main()
