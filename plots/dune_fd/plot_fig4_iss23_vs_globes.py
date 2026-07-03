import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fig4_fd_iss23_vs_globes.csv")
DEFAULT_OUT = Path("figures/dune_fd/iss23/construct23_point70/fig4/fig4_fd_iss23_vs_globes.png")
ISS_COLORS = {
    "black": "0.35",
    "blue": "deepskyblue",
    "limegreen": "forestgreen",
    "red": "tomato",
}


def step(ax, data, y_col, **kwargs):
    ax.step(data["Erec_GeV"], data[y_col], where="mid", **kwargs)


def panel_component(df, panel, component):
    return df[(df["panel"] == panel) & (df["component"] == component)].sort_values("Erec_GeV")


def combined_component(df, panel, components, y_col):
    base = None
    for component in components:
        data = panel_component(df, panel, component)
        if base is None:
            base = data[["Erec_GeV", y_col]].copy()
        else:
            base[y_col] = base[y_col].to_numpy() + data[y_col].to_numpy()
    return base


def reference_column(df):
    if "benchmark3nu_events" not in df.columns:
        raise ValueError("Colonne benchmark3nu_events manquante: relance le preset C FD point 70.")
    return "benchmark3nu_events"


def reference_label(df):
    _ = df
    return r"$3\nu$ active $U_{\rm solver}^{3\times3}$"


def draw_pair(ax, rax, df, panel, title, ylim, app=True, anti=False):
    if app:
        stacks = [
            ("nc", ["nc"], "red", "NC"),
            ("numu", ["nc", "numu"], "limegreen", r"$(\bar{\nu}_\mu+\nu_\mu)$ CC" if anti else r"$(\nu_\mu+\bar{\nu}_\mu)$ CC"),
            ("beam", ["nc", "numu", "beam"], "blue", r"Beam $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Beam $(\nu_e+\bar{\nu}_e)$ CC"),
            ("total", ["nc", "numu", "beam", "signal"], "black", r"Signal $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Signal $(\nu_e+\bar{\nu}_e)$ CC"),
        ]
    else:
        stacks = [
            ("nc", ["nc"], "red", "NC"),
            ("wrong_mu", ["nc", "wrong_mu"], "limegreen", r"$\nu_\mu$ CC" if anti else r"$\bar{\nu}_\mu$ CC"),
            ("total", ["nc", "wrong_mu", "tau", "signal"], "black", r"Signal $\bar{\nu}_\mu$ CC" if anti else r"Signal $\nu_\mu$ CC"),
        ]

    relative_differences = []
    y_max = 0.0
    ref_col = reference_column(df)
    for _, components, color, label in stacks:
        reference = combined_component(df, panel, components, ref_col)
        iss = combined_component(df, panel, components, "iss23_events")
        y_max = max(y_max, float(reference[ref_col].max()), float(iss["iss23_events"].max()))
        step(ax, reference, ref_col, color=color, linewidth=1.3, label=label)
        step(ax, iss, "iss23_events", color=ISS_COLORS[color], linewidth=1.2, linestyle="--")

        g = reference[ref_col].to_numpy(dtype=float)
        y = iss["iss23_events"].to_numpy(dtype=float)
        rel = np.full_like(g, np.nan, dtype=float)
        mask = g > 1.0e-12
        rel[mask] = (y[mask] - g[mask]) / g[mask]
        relative_differences.extend(rel[np.isfinite(rel)].tolist())
        rdata = reference[["Erec_GeV"]].copy()
        rdata["relative_delta"] = rel
        step(rax, rdata, "relative_delta", color=ISS_COLORS[color], linewidth=1.0)

        if color == "black":
            yerr = np.sqrt(np.maximum(reference[ref_col].to_numpy(), 0.0))
            ax.errorbar(reference["Erec_GeV"], reference[ref_col], yerr=yerr, fmt="none", ecolor="black", elinewidth=0.7)

    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(0.0, max(float(ylim), 1.12 * y_max))
    ax.set_ylabel("Events per 0.25 GeV", fontsize=10, fontweight="bold")
    ax.tick_params(labelbottom=False)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.text(
        0.52,
        0.88,
        title + "\nSolid: active 3nu bloc\nDashed: ISS(2,3) pt 70",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
    )
    ax.legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.98, 0.70))

    rax.axhline(0.0, color="black", linewidth=0.7)
    rax.set_xlim(0.5, 8.0)
    finite = np.asarray(relative_differences, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        span = max(hi - lo, 1.0e-6)
        rax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
    else:
        rax.set_ylim(-1.0, 1.0)
    rax.set_xlabel("Reconstructed Energy (GeV)", fontsize=10, fontweight="bold")
    rax.set_ylabel(r"$\Delta N/N$", fontsize=8)
    rax.tick_params(direction="in", top=True, right=True, labelsize=8)
    rax.minorticks_on()
    rax.tick_params(which="minor", direction="in", top=True, right=True)


def print_checks(df):
    ref_col = reference_column(df)
    print(f"Coherence checks, stacked total ISS vs {reference_label(df)}:")
    specs = [
        ("FHC_app", ["nc", "numu", "beam", "signal"]),
        ("RHC_app", ["nc", "numu", "beam", "signal"]),
        ("FHC_dis", ["nc", "wrong_mu", "tau", "signal"]),
        ("RHC_dis", ["nc", "wrong_mu", "tau", "signal"]),
    ]
    for panel, components in specs:
        reference = combined_component(df, panel, components, ref_col)
        iss = combined_component(df, panel, components, "iss23_events")
        g = reference[ref_col].to_numpy()
        y = iss["iss23_events"].to_numpy()
        mask = g > 1e-9
        if not np.any(mask):
            print(f"  {panel}: aucune reference non nulle; check ignore")
            continue
        rel = np.zeros_like(g)
        rel[mask] = (y[mask] - g[mask]) / g[mask]
        sum_g = np.sum(g)
        sum_rel = (np.sum(y) - sum_g) / sum_g if sum_g > 0.0 else np.nan
        print(
            f"  {panel}: max |rel| = {np.max(np.abs(rel[mask])):.3e}, "
            f"sum rel = {sum_rel:.3e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DUNE FD Fig. 4 GLoBES vs ISS(2,3) point 70.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    fig = plt.figure(figsize=(8.4, 10.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[3.2, 0.95, 3.2, 0.95], hspace=0.06, wspace=0.34)
    axes = {
        "FHC_app": (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])),
        "RHC_app": (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])),
        "FHC_dis": (fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[3, 0])),
        "RHC_dis": (fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[3, 1])),
    }
    draw_pair(*axes["FHC_app"], df, "FHC_app", r"$\nu_e$ Appearance", 280, app=True, anti=False)
    draw_pair(*axes["RHC_app"], df, "RHC_app", r"$\bar{\nu}_e$ Appearance", 95, app=True, anti=True)
    draw_pair(*axes["FHC_dis"], df, "FHC_dis", r"$\nu_\mu$ Disappearance", 1450, app=False, anti=False)
    draw_pair(*axes["RHC_dis"], df, "RHC_dis", r"$\bar{\nu}_\mu$ Disappearance", 620, app=False, anti=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.97)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    plt.close(fig)
    print_checks(df)
    print(f"Figure sauvegardee: {args.out.resolve()}")


if __name__ == "__main__":
    main()
