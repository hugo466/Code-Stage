from __future__ import annotations

import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POINT_PATH = ROOT / "data" / "inverse_seesaw" / "3p1" / "inverse_construct_23_kept_points" / "70.txt"
SOURCE_FHC = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_FHC_FD.csv"
SOURCE_RHC = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_RHC_FD.csv"
OUT_CSV = ROOT / "data" / "dune_nd" / "minimal_onaxis" / "point_70" / "plots_validation" / "fd_dk2nu_true_energy_probabilities.csv"
OUT_FIG = ROOT / "figures" / "dune_fd" / "iss23" / "construct23_point70" / "fig4" / "fd_dk2nu_true_energy_probabilities.png"

FD_DISTANCE_M = 1_284_900.0
RHO_G_CM3 = 2.848
ACC_COEFF = 7.6e-5
ANC_OVER_ACC = 0.5
PHASE_FACTOR = 2.0 * 1.267


def read_scalar(text: str, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Champ absent dans {POINT_PATH}: {key}")
    return cast(match.group(1).strip())


def read_matrix_block(text: str, marker: str, n: int) -> np.ndarray:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Bloc absent dans {POINT_PATH}: {marker}")
    rows = []
    for line in text[start + len(marker) :].splitlines():
        if not line.strip():
            continue
        if not line.lstrip().startswith("["):
            if rows:
                break
            continue
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)]
        if len(values) != n:
            raise ValueError(f"Ligne invalide pour {marker}: {line}")
        rows.append(values)
        if len(rows) == n:
            break
    if len(rows) != n:
        raise ValueError(f"Bloc incomplet pour {marker}")
    return np.array(rows, dtype=float)


def load_point70() -> tuple[np.ndarray, np.ndarray]:
    text = POINT_PATH.read_text(encoding="utf-8")
    point_id = read_scalar(text, "point_id", int)
    if point_id != 70:
        raise ValueError(f"Point inattendu: {point_id}")
    re_part = read_matrix_block(text, "U4x4_solver_re =", 4)
    im_part = read_matrix_block(text, "U4x4_solver_im =", 4)
    masses2 = np.array(
        [
            0.0,
            read_scalar(text, "dm21_calc_eV2"),
            read_scalar(text, "dm31_calc_eV2"),
            read_scalar(text, "dm41_calc_eV2"),
        ],
        dtype=float,
    )
    return re_part + 1j * im_part, masses2


def c_matter_probability(
    mixing: np.ndarray,
    masses2: np.ndarray,
    alpha: int,
    beta: int,
    energy_gev: float,
    baseline_km: float,
    anti: bool,
    sterile_nc: bool,
) -> float:
    u = np.conjugate(mixing) if anti else mixing
    h = u @ np.diag(masses2) @ np.conjugate(u).T
    sign = -1.0 if anti else 1.0
    a_cc = ACC_COEFF * RHO_G_CM3 * energy_gev
    h[0, 0] += sign * a_cc
    if sterile_nc and len(masses2) > 3:
        for sterile in range(3, len(masses2)):
            h[sterile, sterile] += sign * ANC_OVER_ACC * a_cc
    eigvals, eigvecs = np.linalg.eigh(h)
    phases = np.exp(-1j * PHASE_FACTOR * eigvals * baseline_km / energy_gev)
    amp = eigvecs @ np.diag(phases) @ np.conjugate(eigvecs).T
    return float(np.clip(abs(amp[beta, alpha]) ** 2, 0.0, 1.0))


def profile_average_probability(profile: pd.DataFrame, flavor: str, energy_low: float, energy_high: float, prob_fn) -> float:
    rows = profile[
        (profile["flavor"] == flavor)
        & np.isclose(profile["E_GeV_bin_low"], energy_low)
        & np.isclose(profile["E_GeV_bin_high"], energy_high)
    ]
    if rows.empty:
        energy = 0.5 * (energy_low + energy_high)
        return prob_fn((FD_DISTANCE_M / 1000.0), energy)
    weights = rows["weight"].to_numpy(dtype=float)
    z = 0.5 * (rows["z_decay_m_bin_low"].to_numpy(dtype=float) + rows["z_decay_m_bin_high"].to_numpy(dtype=float))
    energy = 0.5 * (energy_low + energy_high)
    total = float(np.sum(weights))
    if total <= 0.0:
        return prob_fn((FD_DISTANCE_M / 1000.0), energy)
    probs = np.array([prob_fn((FD_DISTANCE_M - zi) / 1000.0, energy) for zi in z], dtype=float)
    return float(np.sum(weights * probs) / total)


def build_rows() -> list[dict[str, float | str]]:
    u4, m4 = load_point70()
    u3 = u4[:3, :3]
    m3 = m4[:3]
    profiles = {"FHC": pd.read_csv(SOURCE_FHC), "RHC": pd.read_csv(SOURCE_RHC)}
    channels = [
        ("FHC_app", "FHC", "numu", 1, 0, False, r"$P_{\mu e}$"),
        ("RHC_app", "RHC", "numubar", 1, 0, True, r"$P_{\bar{\mu}\bar{e}}$"),
        ("FHC_dis", "FHC", "numu", 1, 1, False, r"$P_{\mu\mu}$"),
        ("RHC_dis", "RHC", "numubar", 1, 1, True, r"$P_{\bar{\mu}\bar{\mu}}$"),
    ]
    rows: list[dict[str, float | str]] = []
    for panel, mode, flavor, alpha, beta, anti, label in channels:
        profile = profiles[mode]
        energy_bins = (
            profile.loc[profile["flavor"] == flavor, ["E_GeV_bin_low", "E_GeV_bin_high"]]
            .drop_duplicates()
            .sort_values(["E_GeV_bin_low", "E_GeV_bin_high"])
        )
        for e_low, e_high in energy_bins.to_numpy(dtype=float):
            if e_low < 0.5 or e_high > 8.0:
                continue
            p3 = profile_average_probability(
                profile,
                flavor,
                e_low,
                e_high,
                lambda baseline_km, energy: c_matter_probability(u3, m3, alpha, beta, energy, baseline_km, anti, sterile_nc=False),
            )
            p4 = profile_average_probability(
                profile,
                flavor,
                e_low,
                e_high,
                lambda baseline_km, energy: c_matter_probability(u4, m4, alpha, beta, energy, baseline_km, anti, sterile_nc=True),
            )
            rows.append(
                {
                    "point_id": 70,
                    "detector": "FD",
                    "source_model": "dk2nu",
                    "panel": panel,
                    "channel_label": label,
                    "flavor": flavor,
                    "E_GeV": 0.5 * (e_low + e_high),
                    "benchmark3nu_probability": p3,
                    "iss23_probability": p4,
                    "delta_probability": p4 - p3,
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    df = pd.DataFrame(rows)
    panels = [
        ("FHC_app", r"$P_{\mu e}$"),
        ("RHC_app", r"$P_{\bar{\mu}\bar{e}}$"),
        ("FHC_dis", r"$P_{\mu\mu}$"),
        ("RHC_dis", r"$P_{\bar{\mu}\bar{\mu}}$"),
    ]
    fig = plt.figure(figsize=(11.0, 8.8))
    grid = fig.add_gridspec(4, 2, height_ratios=[3.0, 1.0, 3.0, 1.0], hspace=0.16, wspace=0.28)
    axes = {
        "FHC_app": (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])),
        "RHC_app": (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])),
        "FHC_dis": (fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[3, 0])),
        "RHC_dis": (fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[3, 1])),
    }
    for panel, label in panels:
        ax, rax = axes[panel]
        sub = df[df["panel"] == panel].sort_values("E_GeV")
        x = sub["E_GeV"].to_numpy(dtype=float)
        p3 = sub["benchmark3nu_probability"].to_numpy(dtype=float)
        p4 = sub["iss23_probability"].to_numpy(dtype=float)
        rel = np.divide(p4 - p3, p3, out=np.full_like(p4, np.nan), where=np.abs(p3) > 1.0e-18)

        ax.plot(x, p3, color="black", lw=1.8, label=r"$3\nu$ active")
        ax.plot(x, p4, color="tab:blue", lw=1.4, ls="--", label="ISS(2,3)")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0.5, 8.0)
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)

        rax.axhline(0.0, color="black", lw=0.8)
        rax.plot(x, rel, color="0.35", lw=1.1)
        rax.set_xlim(0.5, 8.0)
        rax.set_ylabel(r"$\Delta P/P_{3\nu}$", fontsize=9)
        rax.grid(alpha=0.25)
        rax.tick_params(direction="in", top=True, right=True)
        rax.minorticks_on()
        rax.tick_params(which="minor", direction="in", top=True, right=True)
        finite = rel[np.isfinite(rel)]
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            span = max(hi - lo, 1.0e-8)
            rax.set_ylim(lo - 0.18 * span, hi + 0.18 * span)

    axes["FHC_app"][0].legend(frameon=False, fontsize=8)
    for panel in ("FHC_app", "RHC_app"):
        axes[panel][1].tick_params(labelbottom=False)
    for panel in ("FHC_dis", "RHC_dis"):
        axes[panel][1].set_xlabel(r"$E_\nu$ [GeV]")
    fig.suptitle("dk2nu source averaged probability - ISS(2,3) point 70", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    plot(rows)
    print(f"CSV sauvegarde: {OUT_CSV}")
    print(f"Figure sauvegardee: {OUT_FIG}")


if __name__ == "__main__":
    main()
