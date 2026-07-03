from __future__ import annotations

import argparse
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


POINT_ID = 16


def point_path_for(point_id: int) -> Path:
    base = ROOT / "data" / "inverse_seesaw" / "3p2"
    candidates = [
        base / "inverse_construct_24_kept_points" / f"{point_id}.txt",
        base / "inverse_construct_24_kept_points_logdm" / f"{point_id}.txt",
        base / "inverse_pmns_filter_kept_points_9x9" / f"{point_id}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


POINT_PATH = point_path_for(POINT_ID)
SOURCE_FHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_FHC_ND_raw.csv"
SOURCE_RHC = ROOT / "data" / "dune" / "dk2nu" / "flux_z_RHC_ND_raw.csv"
XSEC_CC = ROOT / "data" / "dune" / "2103.04797v2" / "dune_globes" / "xsec" / "xsec_cc.dat"
XSEC_NC = ROOT / "data" / "dune" / "2103.04797v2" / "dune_globes" / "xsec" / "xsec_nc.dat"

OUT_DIR = ROOT / "figures" / "dune_nd" / "iss24" / f"construct24_point{POINT_ID}" / "fig4"
DATA_DIR = ROOT / "data" / "dune_nd" / "iss24" / f"construct24_point{POINT_ID}" / "fig4"
OUT_FIG = OUT_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu.png"
OUT_PROB_FIG = OUT_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu_raw_probabilities.png"
OUT_CSV = DATA_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu.csv"
OUT_PROB_CSV = DATA_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu_raw_probabilities.csv"

L_ND_KM = 0.574
SOURCE_Z_START_M = 0.0
DECAY_PIPE_LENGTH_M = 194.0
SOURCE_Z_BINS = 80
E_MIN = 0.5
E_MAX = 8.0
N_BINS = 30
POT_PER_YEAR = 11.0e20
EXPOSURE_YEARS = 6.5
TARGET_MASS_KT = 0.067
AVOGADRO = 6.02214076e23
M2_TO_CM2 = 1.0e4
XSEC_SCALE_CM2 = 1.0e-38
PHASE_FACTOR = 2.0 * 1.267

FLAVORS = ["nue", "numu", "nutau", "nuebar", "numubar", "nutaubar"]
FLAVOR_INDEX = {"nue": 0, "numu": 1, "nuebar": 0, "numubar": 1}
FLUX_COLORS = {"black": "0.35", "blue": "deepskyblue", "limegreen": "forestgreen", "red": "tomato"}


def configure_point(point_id: int) -> None:
    global POINT_ID, POINT_PATH, OUT_DIR, DATA_DIR, OUT_FIG, OUT_PROB_FIG, OUT_CSV, OUT_PROB_CSV
    POINT_ID = int(point_id)
    POINT_PATH = point_path_for(POINT_ID)
    OUT_DIR = ROOT / "figures" / "dune_nd" / "iss24" / f"construct24_point{POINT_ID}" / "fig4"
    DATA_DIR = ROOT / "data" / "dune_nd" / "iss24" / f"construct24_point{POINT_ID}" / "fig4"
    OUT_FIG = OUT_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu.png"
    OUT_PROB_FIG = OUT_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu_raw_probabilities.png"
    OUT_CSV = DATA_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu.csv"
    OUT_PROB_CSV = DATA_DIR / "fig4_nd_dk2nu_iss24_vs_active3nu_raw_probabilities.csv"


def read_scalar(text: str, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Champ absent dans {POINT_PATH}: {key}")
    return cast(match.group(1).strip())


def has_scalar(text: str, key: str) -> bool:
    return re.search(rf"^{re.escape(key)}\s*=", text, re.MULTILINE) is not None


def read_first_scalar(text: str, keys: list[str], cast=float):
    for key in keys:
        if has_scalar(text, key):
            return read_scalar(text, key, cast)
    raise ValueError(f"Champs absents dans {POINT_PATH}: {', '.join(keys)}")


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


def load_point() -> tuple[np.ndarray, np.ndarray]:
    text = POINT_PATH.read_text(encoding="utf-8")
    if has_scalar(text, "pmns_pass") and read_scalar(text, "pmns_pass", int) != 1:
        raise ValueError(f"Le point {POINT_ID} ne verifie pas pmns_pass=1 et eta_pass=1")
    if has_scalar(text, "eta_pass") and read_scalar(text, "eta_pass", int) != 1:
        raise ValueError(f"Le point {POINT_ID} ne verifie pas pmns_pass=1 et eta_pass=1")
    if "U5_solver_re =" in text and "U5_solver_im =" in text:
        re_part = read_matrix_block(text, "U5_solver_re =", 5)
        im_part = read_matrix_block(text, "U5_solver_im =", 5)
        mixing = re_part + 1j * im_part
    else:
        mixing = read_matrix_block(text, "mixing_9x9 =", 9)[:5, :5].astype(complex)
    masses2 = np.array(
        [
            0.0,
            read_first_scalar(text, ["dm21_calc_eV2", "dm21_eV2"]),
            read_first_scalar(text, ["dm31_calc_eV2", "dm31_eV2"]),
            read_first_scalar(text, ["dm41_calc_eV2", "dm41_eV2"]),
            read_first_scalar(text, ["dm51_calc_eV2", "dm51_eV2"]),
        ],
        dtype=float,
    )
    return mixing, masses2


def read_flux(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["E_GeV", *FLAVORS],
        decimal=",",
        comment="#",
        engine="python",
    ).apply(pd.to_numeric, errors="coerce")


def read_dk2nu_flux_from_z(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    grouped = (
        table.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high"], as_index=False)["weight"]
        .sum()
        .assign(E_GeV=lambda df: 0.5 * (df["E_GeV_bin_low"] + df["E_GeV_bin_high"]))
    )
    flux = grouped.pivot_table(index="E_GeV", columns="flavor", values="weight", aggfunc="sum").reset_index()
    for flavor in FLAVORS:
        if flavor not in flux.columns:
            flux[flavor] = 0.0
    return flux[["E_GeV", *FLAVORS]].sort_values("E_GeV").reset_index(drop=True)


def read_xsec(path: Path) -> pd.DataFrame:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean:
            tokens.extend(clean.split())
    values = [float(token.replace(",", ".")) for token in tokens]
    rows = np.asarray(values, dtype=float).reshape((-1, 7))
    df = pd.DataFrame(rows, columns=["log10_E_GeV", *FLAVORS])
    df.insert(0, "E_GeV", np.power(10.0, df.pop("log10_E_GeV")))
    return df


def interp(df: pd.DataFrame, flavor: str, energy: float) -> float:
    return float(np.interp(energy, df["E_GeV"].to_numpy(), df[flavor].to_numpy(), left=0.0, right=0.0))


def oscillation_probability(mixing: np.ndarray, masses2: np.ndarray, alpha: int, beta: int, energy: float, baseline_km: float, anti: bool) -> float:
    if energy <= 0.0:
        return 0.0
    u = np.conjugate(mixing) if anti else mixing
    phases = np.exp(-1j * PHASE_FACTOR * masses2 * baseline_km / energy)
    amp = np.sum(u[beta, :] * np.conjugate(u[alpha, :]) * phases)
    return float(np.clip(abs(amp) ** 2, 0.0, 1.0))


def source_average_probability(
    profile: pd.DataFrame,
    flavor: str,
    energy_low: float,
    energy_high: float,
    mixing: np.ndarray,
    masses2: np.ndarray,
    alpha: int,
    beta: int,
    anti: bool,
) -> float:
    rows = profile[
        (profile["flavor"] == flavor)
        & np.isclose(profile["E_GeV_bin_low"], energy_low)
        & np.isclose(profile["E_GeV_bin_high"], energy_high)
    ]
    energy = 0.5 * (energy_low + energy_high)
    if rows.empty:
        return oscillation_probability(mixing, masses2, alpha, beta, energy, L_ND_KM, anti)
    weights = rows["weight"].to_numpy(dtype=float)
    z = 0.5 * (rows["z_decay_m_bin_low"].to_numpy(dtype=float) + rows["z_decay_m_bin_high"].to_numpy(dtype=float))
    total = float(np.sum(weights))
    if total <= 0.0:
        return oscillation_probability(mixing, masses2, alpha, beta, energy, L_ND_KM, anti)
    probs = np.array(
        [oscillation_probability(mixing, masses2, alpha, beta, energy, max(L_ND_KM - zi * 1.0e-3, 0.0), anti) for zi in z],
        dtype=float,
    )
    return float(np.sum(weights * probs) / total)


def active_probability(profile, flavor, e_low, e_high, mixing, masses2, alpha, anti) -> float:
    return float(
        np.clip(
            sum(source_average_probability(profile, flavor, e_low, e_high, mixing, masses2, alpha, beta, anti) for beta in range(3)),
            0.0,
            1.0,
        )
    )


def event_rate(flux, xsec, profile, mixing, masses2, initial_flavor, final_flavor, alpha, beta, e_low, e_high, anti, active=False) -> float:
    energy = 0.5 * (e_low + e_high)
    width = e_high - e_low
    phi = interp(flux, initial_flavor, energy) / M2_TO_CM2
    sigma_hat = interp(xsec, final_flavor, energy)
    sigma = sigma_hat * energy * XSEC_SCALE_CM2
    if active:
        p = active_probability(profile, initial_flavor, e_low, e_high, mixing, masses2, alpha, anti)
    else:
        p = source_average_probability(profile, initial_flavor, e_low, e_high, mixing, masses2, alpha, beta, anti)
    return phi * sigma * p * width


def muon_flavor(anti: bool) -> str:
    return "numubar" if anti else "numu"


def electron_flavor(anti: bool) -> str:
    return "nuebar" if anti else "nue"


def component_rate(flux, cc, nc, profile, mixing, masses2, panel: str, component: str, e_low: float, e_high: float) -> float:
    is_rhc = panel.startswith("RHC")
    is_app = "_app" in panel
    right_anti = is_rhc
    wrong_anti = not right_anti
    right_mu = muon_flavor(right_anti)
    wrong_mu = muon_flavor(wrong_anti)
    right_e = electron_flavor(right_anti)
    wrong_e = electron_flavor(wrong_anti)

    if component == "nc":
        selection = 0.025 if is_app else 0.015
        total = 0.0
        for flavor, alpha, anti in [("nue", 0, False), ("numu", 1, False), ("nuebar", 0, True), ("numubar", 1, True)]:
            total += event_rate(flux, nc, profile, mixing, masses2, flavor, flavor, alpha, 0, e_low, e_high, anti, active=True)
        return selection * total

    if is_app and component == "numu":
        mis_id = 0.015
        return mis_id * (
            event_rate(flux, cc, profile, mixing, masses2, right_mu, right_mu, 1, 1, e_low, e_high, right_anti)
            + event_rate(flux, cc, profile, mixing, masses2, wrong_mu, wrong_mu, 1, 1, e_low, e_high, wrong_anti)
        )

    if is_app and component == "beam":
        return event_rate(flux, cc, profile, mixing, masses2, right_e, right_e, 0, 0, e_low, e_high, right_anti) + event_rate(
            flux, cc, profile, mixing, masses2, wrong_e, wrong_e, 0, 0, e_low, e_high, wrong_anti
        )

    if is_app and component == "signal":
        return event_rate(flux, cc, profile, mixing, masses2, right_mu, right_e, 1, 0, e_low, e_high, right_anti) + event_rate(
            flux, cc, profile, mixing, masses2, wrong_mu, wrong_e, 1, 0, e_low, e_high, wrong_anti
        )

    if (not is_app) and component == "wrong_mu":
        return event_rate(flux, cc, profile, mixing, masses2, wrong_mu, wrong_mu, 1, 1, e_low, e_high, wrong_anti)

    if (not is_app) and component == "signal":
        return event_rate(flux, cc, profile, mixing, masses2, right_mu, right_mu, 1, 1, e_low, e_high, right_anti)

    return 0.0


def panel_stacks(app=True, anti=False):
    if app:
        return [
            ("nc", ["nc"], "red", "NC"),
            ("numu", ["nc", "numu"], "limegreen", r"$(\nu_\mu+\bar{\nu}_\mu)$ CC"),
            ("beam", ["nc", "numu", "beam"], "blue", r"Beam $(\nu_e+\bar{\nu}_e)$ CC"),
            ("signal", ["nc", "numu", "beam", "signal"], "black", r"Signal $(\nu_e+\bar{\nu}_e)$ CC"),
        ]
    return [
        ("nc", ["nc"], "red", "NC"),
        ("wrong_mu", ["nc", "wrong_mu"], "limegreen", r"$\nu_\mu$ CC" if anti else r"$\bar{\nu}_\mu$ CC"),
        ("signal", ["nc", "wrong_mu", "signal"], "black", r"Signal $\bar{\nu}_\mu$ CC" if anti else r"Signal $\nu_\mu$ CC"),
    ]


def combined(df: pd.DataFrame, panel: str, components: list[str], col: str) -> pd.DataFrame:
    out = None
    for component in components:
        sub = df[(df["panel"] == panel) & (df["component"] == component)].sort_values("Erec_GeV")
        if out is None:
            out = sub[["Erec_GeV", col]].copy()
        else:
            out[col] = out[col].to_numpy() + sub[col].to_numpy()
    return out if out is not None else pd.DataFrame(columns=["Erec_GeV", col])


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    u5, m5 = load_point()
    u3 = u5[:3, :3]
    m3 = m5[:3]
    profiles = {"FHC": pd.read_csv(SOURCE_FHC), "RHC": pd.read_csv(SOURCE_RHC)}
    fluxes = {"FHC": read_dk2nu_flux_from_z(SOURCE_FHC), "RHC": read_dk2nu_flux_from_z(SOURCE_RHC)}
    cc = read_xsec(XSEC_CC)
    nc = read_xsec(XSEC_NC)
    scale = POT_PER_YEAR * EXPOSURE_YEARS * TARGET_MASS_KT * 1.0e9 * AVOGADRO

    panels = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
    app_components = ["nc", "numu", "beam", "signal"]
    dis_components = ["nc", "wrong_mu", "signal"]
    rows = []
    prob_rows = []
    width = (E_MAX - E_MIN) / N_BINS

    for panel in panels:
        mode = "RHC" if panel.startswith("RHC") else "FHC"
        profile = profiles[mode]
        flux = fluxes[mode]
        is_app = "_app" in panel
        right_anti = mode == "RHC"
        wrong_anti = not right_anti
        components = app_components if is_app else dis_components
        for b in range(N_BINS):
            e_low = E_MIN + b * width
            e_high = e_low + width
            energy = 0.5 * (e_low + e_high)
            for component in components:
                g = component_rate(flux, cc, nc, profile, u3, m3, panel, component, e_low, e_high) * scale
                y = component_rate(flux, cc, nc, profile, u5, m5, panel, component, e_low, e_high) * scale
                rows.append(
                    {
                        "point_id": POINT_ID,
                        "detector": "ND",
                        "model": "iss24",
                        "source_model": "dk2nu",
                        "panel": panel,
                        "component": component,
                        "Erec_GeV": energy,
                        "active3nu_events": g,
                        "iss24_events": y,
                        "ratio_iss24_over_3nu": y / g if g > 0.0 else 0.0,
                        "rel_diff": (y - g) / g if g > 0.0 else 0.0,
                        "delta_events": y - g,
                    }
                )

            for channel, flavor, alpha, beta, anti in [
                ("right_mu_to_e", muon_flavor(right_anti), 1, 0, right_anti),
                ("right_mu_to_mu", muon_flavor(right_anti), 1, 1, right_anti),
                ("right_mu_active", muon_flavor(right_anti), 1, -1, right_anti),
                ("right_e_to_e", electron_flavor(right_anti), 0, 0, right_anti),
                ("wrong_mu_to_e", muon_flavor(wrong_anti), 1, 0, wrong_anti),
                ("wrong_mu_to_mu", muon_flavor(wrong_anti), 1, 1, wrong_anti),
                ("wrong_mu_active", muon_flavor(wrong_anti), 1, -1, wrong_anti),
            ]:
                if beta >= 0:
                    p3 = source_average_probability(profile, flavor, e_low, e_high, u3, m3, alpha, beta, anti)
                    p5 = source_average_probability(profile, flavor, e_low, e_high, u5, m5, alpha, beta, anti)
                else:
                    p3 = active_probability(profile, flavor, e_low, e_high, u3, m3, alpha, anti)
                    p5 = active_probability(profile, flavor, e_low, e_high, u5, m5, alpha, anti)
                prob_rows.append(
                    {
                        "point_id": POINT_ID,
                        "detector": "ND",
                        "model": "iss24",
                        "source_model": "dk2nu",
                        "panel": panel,
                        "channel": channel,
                        "initial_flavor": flavor,
                        "alpha": alpha,
                        "beta": beta,
                        "E_GeV": energy,
                        "benchmark3nu_probability": p3,
                        "iss24_probability": p5,
                        "delta_probability": p5 - p3,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prob_rows)


def plot_fig4(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.4, 10.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[3.2, 0.95, 3.2, 0.95], hspace=0.06, wspace=0.34)
    specs = [
        ("FHC_app", r"$\nu_e$ Appearance", True, False, grid[0, 0], grid[1, 0]),
        ("RHC_app", r"$\bar{\nu}_e$ Appearance", True, True, grid[0, 1], grid[1, 1]),
        ("FHC_dis", r"$\nu_\mu$ Disappearance", False, False, grid[2, 0], grid[3, 0]),
        ("RHC_dis", r"$\bar{\nu}_\mu$ Disappearance", False, True, grid[2, 1], grid[3, 1]),
    ]
    for panel, title, app, anti, main_cell, res_cell in specs:
        ax = fig.add_subplot(main_cell)
        rax = fig.add_subplot(res_cell)
        diffs = []
        ymax = 0.0
        for _, components, color, label in panel_stacks(app=app, anti=anti):
            ref = combined(df, panel, components, "active3nu_events")
            iss = combined(df, panel, components, "iss24_events")
            ax.step(ref["Erec_GeV"], ref["active3nu_events"], where="mid", color=color, lw=1.3, label=label)
            ax.step(iss["Erec_GeV"], iss["iss24_events"], where="mid", color=FLUX_COLORS[color], lw=1.2, ls="--")
            g = ref["active3nu_events"].to_numpy(dtype=float)
            y = iss["iss24_events"].to_numpy(dtype=float)
            rel = np.divide(y - g, g, out=np.zeros_like(y), where=np.abs(g) > 1.0e-18)
            diffs.extend(rel[np.isfinite(rel)].tolist())
            rax.step(ref["Erec_GeV"], rel, where="mid", color=FLUX_COLORS[color], lw=1.0)
            ymax = max(ymax, float(np.nanmax(g)), float(np.nanmax(y)))
            if color == "black":
                ax.errorbar(ref["Erec_GeV"], g, yerr=np.sqrt(np.maximum(g, 0.0)), fmt="none", ecolor="black", elinewidth=0.7)
        ax.set_xlim(0.5, 8.0)
        ax.set_ylim(0.0, 1.15 * ymax if ymax > 0.0 else 1.0)
        ax.set_ylabel("Events per 0.25 GeV", fontsize=10, fontweight="bold")
        ax.tick_params(labelbottom=False, direction="in", top=True, right=True, labelsize=9)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        ax.text(
            0.47,
            0.88,
            title
            + f"\nND dk2nu avg, {EXPOSURE_YEARS:g} yr/mode"
            + f"\nISS(2,4) pt {POINT_ID}"
            + "\nSolid: active 3nu bloc"
            + "\nDashed: ISS(2,4)",
            transform=ax.transAxes,
            fontsize=6.2,
            fontweight="bold",
            va="top",
        )
        ax.legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.98, 0.63))
        finite = np.asarray(diffs, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            pad = 0.2 * max(hi - lo, 1.0e-6)
            rax.set_ylim(lo - pad, hi + pad)
        rax.axhline(0.0, color="black", lw=0.7)
        rax.set_xlim(0.5, 8.0)
        rax.set_xlabel("Reconstructed Energy (GeV)", fontsize=10, fontweight="bold")
        rax.set_ylabel(r"$\Delta N/N$", fontsize=8)
        rax.tick_params(direction="in", top=True, right=True, labelsize=8)
        rax.minorticks_on()
        rax.tick_params(which="minor", direction="in", top=True, right=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.97)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=240)
    plt.close(fig)


def plot_raw_probabilities(df: pd.DataFrame) -> None:
    channels = [
        ("right_mu_to_e", r"$P_{\mu e}$"),
        ("right_mu_to_mu", r"$P_{\mu\mu}$"),
        ("right_mu_active", r"$P_{\mu\to active}$"),
    ]
    panels = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
    titles = [r"$\nu_e$ app.", r"$\bar{\nu}_e$ app.", r"$\nu_\mu$ dis.", r"$\bar{\nu}_\mu$ dis."]
    fig, axes = plt.subplots(len(channels), len(panels), figsize=(13.2, 7.8), sharex=True, squeeze=False)
    for row, (channel, ylabel) in enumerate(channels):
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            sub = df[(df["panel"] == panel) & (df["channel"] == channel)].sort_values("E_GeV")
            x = sub["E_GeV"].to_numpy(dtype=float)
            p3 = sub["benchmark3nu_probability"].to_numpy(dtype=float)
            p5 = sub["iss24_probability"].to_numpy(dtype=float)
            ax.plot(x, p5, color="tab:blue", lw=1.1, ls="--", label="ISS(2,4)", zorder=2)
            ax.plot(x, p3, color="black", lw=1.8, label="3nu active", zorder=5)
            ax.set_xlim(0.5, 8.0)
            vals = np.concatenate([p3[np.isfinite(p3)], p5[np.isfinite(p5)]])
            if vals.size:
                lo = float(np.min(vals))
                hi = float(np.max(vals))
                pad = 0.08 * max(hi - lo, 1.0e-6)
                ax.set_ylim(lo - pad, hi + pad)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
            ax.minorticks_on()
            ax.tick_params(which="minor", direction="in", top=True, right=True)
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == len(channels) - 1:
                ax.set_xlabel(r"$E_\nu$ [GeV]", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle(f"dk2nu source averaged probability - ISS(2,4) point {POINT_ID}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    OUT_PROB_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PROB_FIG, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ND dk2nu Fig4-like spectra and raw probabilities for an ISS(2,4) point.")
    parser.add_argument("--point-id", type=int, default=POINT_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_point(args.point_id)
    spectra, probabilities = build_tables()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(OUT_CSV, index=False)
    probabilities.to_csv(OUT_PROB_CSV, index=False)
    plot_fig4(spectra)
    plot_raw_probabilities(probabilities)
    print(f"Point ISS(2,4): {POINT_ID} (pmns_pass=1, eta_pass=1)")
    print(f"CSV spectres: {OUT_CSV}")
    print(f"CSV probas: {OUT_PROB_CSV}")
    print(f"Figure Fig4: {OUT_FIG}")
    print(f"Figure probas brutes: {OUT_PROB_FIG}")


if __name__ == "__main__":
    main()
