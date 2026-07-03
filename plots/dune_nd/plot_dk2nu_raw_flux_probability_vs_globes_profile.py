#!/usr/bin/env python
"""Compare dk2nu absolute flux-weighted probabilities with GLoBES flux times dk2nu source profiles.

For each channel this plots, as a function of true neutrino energy,
  I_raw(E) = sum_z Phi_dk2nu_raw(E,z,flavor) P(L-z,E)
and
  I_globes_profile(E) = Phi_GLoBES(E,flavor) sum_z p_dk2nu(z|E,flavor) P(L-z,E).

No cross section, efficiency or smearing is applied: this isolates the flux/source-profile convention.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POINT = ROOT / "data" / "inverse_seesaw" / "3p1" / "inverse_construct_23_kept_points" / "70.txt"
FHC_RAW = ROOT / "data" / "dune" / "dk2nu" / "flux_z_FHC_ND_raw.csv"
RHC_RAW = ROOT / "data" / "dune" / "dk2nu" / "flux_z_RHC_ND_raw.csv"
FHC_PROFILE = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_FHC_ND.csv"
RHC_PROFILE = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_RHC_ND.csv"
FHC_GLOBES = ROOT / "data" / "dune" / "flux" / "flux_dune_neutrino_ND_globes.txt"
RHC_GLOBES = ROOT / "data" / "dune" / "flux" / "flux_dune_antineutrino_ND_globes.txt"
OUT = ROOT / "figures" / "dune_nd" / "flux" / "dk2nu_raw_flux_probability_vs_globes_profile_point70.png"
OUT_CSV = ROOT / "data" / "dune" / "dk2nu" / "dk2nu_raw_flux_probability_vs_globes_profile_point70.csv"

BASELINE_KM = 0.574
PHASE_COEFF = 1.2669328
FLUX_COLUMNS = ["E_GeV", "nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
CHANNELS = [
    ("FHC", "numu", 1, 0, False, r"FHC $\nu_\mu\to\nu_e$"),
    ("RHC", "numubar", 1, 0, True, r"RHC $\bar\nu_\mu\to\bar\nu_e$"),
    ("FHC", "numu", 1, 1, False, r"FHC $\nu_\mu\to\nu_\mu$"),
    ("RHC", "numubar", 1, 1, True, r"RHC $\bar\nu_\mu\to\bar\nu_\mu$"),
]


def extract_scalar(text: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([+-]?[0-9.eE+-]+)", text, flags=re.M)
    if not match:
        raise ValueError(f"Missing scalar {name}")
    return float(match.group(1))


def extract_matrix(text: str, label: str, n: int = 4) -> np.ndarray:
    start = text.index(label)
    chunk = text[start:].split("\n", 1)[1].split("\n")
    rows = []
    for line in chunk:
        if not line.strip():
            if rows:
                break
            continue
        if not line.lstrip().startswith("["):
            if rows:
                break
            continue
        values = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", line)]
        if values:
            rows.append(values[:n])
        if len(rows) == n:
            break
    if len(rows) != n:
        raise ValueError(f"Could not parse {label}")
    return np.asarray(rows, dtype=float)


def load_point(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    u_re = extract_matrix(text, "U4x4_solver_re =")
    u_im = extract_matrix(text, "U4x4_solver_im =")
    masses2 = np.array(
        [
            0.0,
            extract_scalar(text, "dm21_calc_eV2"),
            extract_scalar(text, "dm31_calc_eV2"),
            extract_scalar(text, "dm41_calc_eV2"),
        ],
        dtype=float,
    )
    return u_re + 1j * u_im, masses2


def probability(u: np.ndarray, masses2: np.ndarray, alpha: int, beta: int, energy: float, baseline_km: float, anti: bool) -> float:
    if energy <= 0.0:
        return 0.0
    p = 1.0 if alpha == beta else 0.0
    im_sign = -1.0 if anti else 1.0
    for i in range(4):
        for j in range(i + 1, 4):
            phase = PHASE_COEFF * (masses2[j] - masses2[i]) * baseline_km / energy
            a = u[alpha, i] * np.conjugate(u[beta, i]) * np.conjugate(u[alpha, j]) * u[beta, j]
            p -= 4.0 * np.real(a) * np.sin(phase) ** 2
            p += im_sign * 2.0 * np.imag(a) * np.sin(2.0 * phase)
    return float(np.clip(p, 0.0, 1.0))


def read_globes(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=FLUX_COLUMNS,
        decimal=",",
        comment="#",
        engine="python",
    ).apply(pd.to_numeric, errors="coerce").dropna(subset=["E_GeV"])


def read_z_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for col in ["E_GeV_bin_low", "E_GeV_bin_high", "z_decay_m_bin_low", "z_decay_m_bin_high", "weight"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=["flavor", "E_GeV_bin_low", "E_GeV_bin_high", "z_decay_m_bin_low", "z_decay_m_bin_high", "weight"])
    table["E_GeV"] = 0.5 * (table["E_GeV_bin_low"] + table["E_GeV_bin_high"])
    table["z_m"] = 0.5 * (table["z_decay_m_bin_low"] + table["z_decay_m_bin_high"])
    return table


def channel_integrals(raw: pd.DataFrame, profile: pd.DataFrame, globes: pd.DataFrame, flavor: str, alpha: int, beta: int, anti: bool, u: np.ndarray, masses2: np.ndarray) -> pd.DataFrame:
    rows = []
    raw_sub = raw[raw["flavor"] == flavor]
    prof_sub = profile[profile["flavor"] == flavor]
    keys = sorted(set(zip(raw_sub["E_GeV_bin_low"], raw_sub["E_GeV_bin_high"])))
    for e_low, e_high in keys:
        e = 0.5 * (float(e_low) + float(e_high))
        r = raw_sub[(raw_sub["E_GeV_bin_low"] == e_low) & (raw_sub["E_GeV_bin_high"] == e_high)]
        q = prof_sub[(prof_sub["E_GeV_bin_low"] == e_low) & (prof_sub["E_GeV_bin_high"] == e_high)]
        if r.empty or q.empty:
            continue
        r_probs = np.array([probability(u, masses2, alpha, beta, e, max(BASELINE_KM - z * 1e-3, 0.0), anti) for z in r["z_m"]])
        q_probs = np.array([probability(u, masses2, alpha, beta, e, max(BASELINE_KM - z * 1e-3, 0.0), anti) for z in q["z_m"]])
        raw_integral = float(np.sum(r["weight"].to_numpy(dtype=float) * r_probs))
        prof_weights = q["weight"].to_numpy(dtype=float)
        prof_sum = float(np.sum(prof_weights))
        if prof_sum <= 0.0:
            continue
        pbar = float(np.sum(prof_weights * q_probs) / prof_sum)
        globes_flux = float(np.interp(e, globes["E_GeV"].to_numpy(dtype=float), globes[flavor].to_numpy(dtype=float), left=0.0, right=0.0))
        globes_profile_integral = globes_flux * pbar
        ratio = raw_integral / globes_profile_integral if globes_profile_integral > 0.0 else np.nan
        rows.append(
            {
                "E_GeV": e,
                "flavor": flavor,
                "raw_flux_times_probability": raw_integral,
                "globes_flux_times_profile_probability": globes_profile_integral,
                "ratio_raw_over_globes_profile": ratio,
                "relative_difference": ratio - 1.0 if np.isfinite(ratio) else np.nan,
                "pbar_profile": pbar,
                "sum_raw_flux_weight": float(np.sum(r["weight"])),
                "globes_flux": globes_flux,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--emax-GeV", type=float, default=8.0)
    args = parser.parse_args()

    u, masses2 = load_point(POINT)
    inputs = {
        "FHC": (read_z_table(FHC_RAW), read_z_table(FHC_PROFILE), read_globes(FHC_GLOBES)),
        "RHC": (read_z_table(RHC_RAW), read_z_table(RHC_PROFILE), read_globes(RHC_GLOBES)),
    }

    all_rows = []
    fig = plt.figure(figsize=(13, 9.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[3.0, 1.0, 3.0, 1.0], hspace=0.12, wspace=0.24)
    panel_axes = [
        (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])),
        (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])),
        (fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[3, 0])),
        (fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[3, 1])),
    ]

    for (mode, flavor, alpha, beta, anti, title), (ax, rax) in zip(CHANNELS, panel_axes):
        raw, profile, globes = inputs[mode]
        data = channel_integrals(raw, profile, globes, flavor, alpha, beta, anti, u, masses2)
        data.insert(0, "channel", title.replace("$", ""))
        all_rows.append(data)
        data = data[data["E_GeV"] <= args.emax_GeV]
        ax.step(data["E_GeV"], data["raw_flux_times_probability"], where="mid", color="#d62728", linewidth=1.6, label=r"$\sum_z \Phi^{dk2nu}_{raw}(E,z)P(L-z,E)$")
        ax.step(data["E_GeV"], data["globes_flux_times_profile_probability"], where="mid", color="black", linestyle="--", linewidth=1.5, label=r"$\Phi^{GLoBES}(E)\sum_z p^{dk2nu}(z|E)P(L-z,E)$")
        rax.axhline(0.0, color="0.2", linewidth=0.8)
        rax.step(data["E_GeV"], data["relative_difference"], where="mid", color="#1f77b4", linewidth=1.3)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlim(0.5, args.emax_GeV)
        rax.set_xlim(0.5, args.emax_GeV)
        ax.set_ylabel(r"flux $\times$ prob.")
        rax.set_ylabel(r"ratio $-1$")
        rax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.grid(alpha=0.25)
        rax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False, direction="in", top=True, right=True)
        rax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on(); rax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        rax.tick_params(which="minor", direction="in", top=True, right=True)
        if mode == "FHC" and beta == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")

    out_df = pd.concat(all_rows, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"CSV sauvegarde: {args.out_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")
    for channel, grp in out_df.groupby("channel"):
        sub = grp[(grp["E_GeV"] >= 0.5) & (grp["E_GeV"] <= args.emax_GeV)]
        print(channel, "median ratio", float(np.nanmedian(sub["ratio_raw_over_globes_profile"])), "min", float(np.nanmin(sub["ratio_raw_over_globes_profile"])), "max", float(np.nanmax(sub["ratio_raw_over_globes_profile"])))


if __name__ == "__main__":
    main()
