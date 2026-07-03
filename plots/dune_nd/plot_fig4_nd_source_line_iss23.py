import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fig4_nd_source_line_iss23_vs_active3nu.csv")
DEFAULT_OUT = Path("figures/dune_nd/iss23/construct23_point70/fig4/fig4_nd_source_line_iss23_vs_active3nu.png")
EXPOSURE_YEARS = 6.5
ISS_COLORS = {
    "black": "0.35",
    "blue": "deepskyblue",
    "limegreen": "forestgreen",
    "red": "tomato",
}
PROB_CHANNELS = [
    ("right_mu_to_e", r"$P_{\mu e}$"),
    ("right_mu_to_mu", r"$P_{\mu\mu}$"),
    ("right_mu_active", r"$P_{\mu\to active}$"),
]


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


def panel_stacks(app=True, anti=False):
    if app:
        return [
            ("nc", ["nc"], "red", "NC"),
            ("numu", ["nc", "numu"], "limegreen", r"$(\bar{\nu}_\mu+\nu_\mu)$ CC" if anti else r"$(\nu_\mu+\bar{\nu}_\mu)$ CC"),
            ("beam", ["nc", "numu", "beam"], "blue", r"Beam $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Beam $(\nu_e+\bar{\nu}_e)$ CC"),
            ("signal", ["nc", "numu", "beam", "signal"], "black", r"Signal $(\bar{\nu}_e+\nu_e)$ CC" if anti else r"Signal $(\nu_e+\bar{\nu}_e)$ CC"),
        ]
    return [
        ("nc", ["nc"], "red", "NC"),
        ("wrong_mu", ["nc", "wrong_mu"], "limegreen", r"$\nu_\mu$ CC" if anti else r"$\bar{\nu}_\mu$ CC"),
        ("signal", ["nc", "wrong_mu", "tau", "signal"], "black", r"Signal $\bar{\nu}_\mu$ CC" if anti else r"Signal $\nu_\mu$ CC"),
    ]


def panel_ylim(df, panel, app=True, anti=False):
    ymax = 0.0
    for _, components, _, _ in panel_stacks(app=app, anti=anti):
        globes = combined_component(df, panel, components, "globes_events")
        iss = combined_component(df, panel, components, "iss23_events")
        ymax = max(ymax, float(globes["globes_events"].max()), float(iss["iss23_events"].max()))
    return 1.15 * ymax if ymax > 0.0 else 1.0


def difference_limits(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = max(hi - lo, 1.0e-6)
    pad = 0.2 * span
    return lo - pad, hi + pad


def residual_limits(values, x_values=None, x_min=None):
    finite = np.asarray(values, dtype=float)
    if x_values is not None and x_min is not None:
        x = np.asarray(x_values, dtype=float)
        finite = finite[x >= x_min]
    return difference_limits(finite)
    

def draw_pair(ax, rax, df, panel, title, model_label, point_label, app=True, anti=False, bottom_mode="signal-difference"):
    residual_values = []
    black_curve = None
    for _, components, color, label in panel_stacks(app=app, anti=anti):
        globes = combined_component(df, panel, components, "globes_events")
        iss = combined_component(df, panel, components, "iss23_events")
        step(ax, globes, "globes_events", color=color, linewidth=1.3, label=label)
        step(ax, iss, "iss23_events", color=ISS_COLORS[color], linewidth=1.2, linestyle="--")

        if bottom_mode == "summed-residuals":
            g = globes["globes_events"].to_numpy(dtype=float)
            y = iss["iss23_events"].to_numpy(dtype=float)
            residual = np.divide(y - g, g, out=np.zeros_like(y), where=np.abs(g) > 1.0e-18)
            residual_values.extend(residual[np.isfinite(residual)].tolist())
            rdata = globes[["Erec_GeV"]].copy()
            rdata["summed_residual"] = residual
            step(rax, rdata, "summed_residual", color=ISS_COLORS[color], linewidth=1.0)

        if color == "black":
            black_curve = (globes, iss)
            yerr = np.sqrt(np.maximum(globes["globes_events"].to_numpy(), 0.0))
            ax.errorbar(globes["Erec_GeV"], globes["globes_events"], yerr=yerr, fmt="none", ecolor="black", elinewidth=0.7)

    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(0.0, panel_ylim(df, panel, app=app, anti=anti))
    ax.set_ylabel("Events per 0.25 GeV", fontsize=10, fontweight="bold")
    ax.tick_params(labelbottom=False)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.text(
        0.47,
        0.88,
        title
        + f"\n{model_label}, {EXPOSURE_YEARS:g} yr/mode"
        + f"\nLND = 574 m, ldec = 194 m"
        + "\nSolid: active 3nu bloc"
        + f"\nDashed: ISS(2,3) {point_label}",
        transform=ax.transAxes,
        fontsize=6.2,
        fontweight="bold",
        va="top",
    )
    ax.legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.98, 0.63))

    rax.axhline(0.0, color="black", linewidth=0.7)
    if bottom_mode in ("signal-difference", "signal-residual"):
        signal = panel_component(df, panel, "signal")
        signal_3nu = signal["globes_events"].to_numpy(dtype=float)
        signal_iss = signal["iss23_events"].to_numpy(dtype=float)
        rdata = signal[["Erec_GeV"]].copy()
        if bottom_mode == "signal-difference":
            signal_residual = signal_iss - signal_3nu
            rdata["signal_residual"] = signal_residual
            step(rax, rdata, "signal_residual", color=ISS_COLORS["black"], linewidth=1.1)
            rax.set_ylim(*difference_limits(signal_residual))
            rax.set_ylabel(r"$\Delta N_{\rm sig}$", fontsize=8)
        else:
            signal_residual = np.divide(
                signal_iss - signal_3nu,
                signal_3nu,
                out=np.full_like(signal_iss, np.nan),
                where=np.abs(signal_3nu) > 1.0e-18,
            )
            rdata["signal_residual"] = signal_residual
            if app:
                rdata = rdata[rdata["Erec_GeV"] >= 1.0]
                signal_residual = rdata["signal_residual"].to_numpy(dtype=float)
            step(rax, rdata, "signal_residual", color=ISS_COLORS["black"], linewidth=1.1)
            rax.set_ylim(
                *residual_limits(
                    signal_residual,
                )
            )
            rax.set_ylabel(r"$\Delta N_{\rm sig}/N_{\rm sig}$", fontsize=8)
    elif bottom_mode == "black-curve-residual":
        globes, iss = black_curve
        g = globes["globes_events"].to_numpy(dtype=float)
        y = iss["iss23_events"].to_numpy(dtype=float)
        residual = np.divide(y - g, g, out=np.full_like(y, np.nan), where=np.abs(g) > 1.0e-18)
        rdata = globes[["Erec_GeV"]].copy()
        rdata["black_curve_residual"] = residual
        step(rax, rdata, "black_curve_residual", color=ISS_COLORS["black"], linewidth=1.1)
        rax.set_ylim(*difference_limits(residual))
        rax.set_ylabel(r"$\Delta N/N$", fontsize=8)
    else:
        rax.set_ylim(*difference_limits(residual_values))
        rax.set_ylabel(r"$\Delta N/N$", fontsize=8)
    rax.set_xlim(0.5, 8.0)
    rax.set_xlabel("Reconstructed Energy (GeV)", fontsize=10, fontweight="bold")
    rax.tick_params(direction="in", top=True, right=True, labelsize=8)
    rax.minorticks_on()
    rax.tick_params(which="minor", direction="in", top=True, right=True)


def print_checks(df):
    point_id = int(df["point_id"].iloc[0]) if "point_id" in df.columns and not df.empty else 70
    print(f"ND source-line checks, displayed pure components ISS vs 3nu active point {point_id}:")
    specs = [
        ("FHC_app", [("nc", ["nc"]), ("numu", ["numu"]), ("beam", ["beam"]), ("signal", ["signal"])]),
        ("RHC_app", [("nc", ["nc"]), ("numu", ["numu"]), ("beam", ["beam"]), ("signal", ["signal"])]),
        ("FHC_dis", [("nc", ["nc"]), ("wrong_mu", ["wrong_mu"]), ("signal", ["signal"])]),
        ("RHC_dis", [("nc", ["nc"]), ("wrong_mu", ["wrong_mu"]), ("signal", ["signal"])]),
    ]
    for panel, component_specs in specs:
        print(f"  {panel}:")
        for label, components in component_specs:
            globes = combined_component(df, panel, components, "globes_events")
            iss = combined_component(df, panel, components, "iss23_events")
            g = globes["globes_events"].to_numpy()
            y = iss["iss23_events"].to_numpy()
            mask = g > 1e-9
            if not np.any(mask):
                print(f"    {label}: reference is zero in all bins")
                continue
            rel = np.zeros_like(g)
            rel[mask] = (y[mask] - g[mask]) / g[mask]
            print(
                f"    {label}: max |rel| = {np.max(np.abs(rel[mask])):.3e}, "
                f"sum rel = {(np.sum(y) - np.sum(g)) / np.sum(g):.3e}"
            )


def raw_probability_path(spectrum_path):
    return spectrum_path.with_name(spectrum_path.stem + "_raw_probabilities.csv")


def raw_probability_figure_path(out_path):
    return out_path.with_name(out_path.stem + "_raw_probabilities.png")


def plot_raw_probabilities(prob_csv, out_path):
    if not prob_csv.exists():
        print(f"Probabilites brutes non tracees: fichier absent {prob_csv}")
        return
    pdf = pd.read_csv(prob_csv)
    if pdf.empty:
        print(f"Probabilites brutes non tracees: fichier vide {prob_csv}")
        return

    panels = ["FHC_app", "RHC_app", "FHC_dis", "RHC_dis"]
    titles = [r"$\nu_e$ app.", r"$\bar{\nu}_e$ app.", r"$\nu_\mu$ dis.", r"$\bar{\nu}_\mu$ dis."]
    point_id = int(pdf["point_id"].iloc[0]) if "point_id" in pdf.columns else 0
    source_model = str(pdf["source_model"].iloc[0]) if "source_model" in pdf.columns else "unknown"
    fig, axes = plt.subplots(
        len(PROB_CHANNELS),
        len(panels),
        figsize=(13.2, 7.8),
        sharex=True,
        squeeze=False,
    )

    for row, (channel, ylabel) in enumerate(PROB_CHANNELS):
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            data = pdf[(pdf["panel"] == panel) & (pdf["channel"] == channel)].sort_values("E_GeV")
            if data.empty:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                continue
            x = data["E_GeV"].to_numpy(dtype=float)
            p3 = data["benchmark3nu_probability"].to_numpy(dtype=float)
            p4 = data["iss23_probability"].to_numpy(dtype=float)
            ax.plot(x, p4, color="tab:blue", linewidth=1.1, linestyle="--", label="ISS(2,3)", zorder=2)
            ax.plot(x, p3, color="black", linewidth=1.8, label="3nu active", zorder=5)
            ax.set_xlim(0.5, 8.0)
            finite_probs = np.concatenate([p3[np.isfinite(p3)], p4[np.isfinite(p4)]])
            if finite_probs.size:
                lo = float(np.min(finite_probs))
                hi = float(np.max(finite_probs))
                span = max(hi - lo, 1.0e-6)
                ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
            ax.minorticks_on()
            ax.tick_params(which="minor", direction="in", top=True, right=True)
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == len(PROB_CHANNELS) - 1:
                ax.set_xlabel(r"$E_\nu$ [GeV]", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=7, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Probabilites brutes sauvegardees: {out_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DUNE ND source-line Fig.4-like spectra vs ISS(2,3).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--bottom-mode",
        choices=("signal-difference", "signal-residual", "black-curve-residual", "summed-residuals"),
        default="signal-difference",
        help="Content of the lower panels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    source_model = str(df["source_model"].iloc[0]) if "source_model" in df.columns and not df.empty else "uniform"
    point_id = int(df["point_id"].iloc[0]) if "point_id" in df.columns and not df.empty else 70
    point_label = f"point {point_id}"
    model_label = "ND dk2nu avg" if source_model == "dk2nu" else ("ND point-source" if source_model == "point" else "ND source-line avg")
    fig = plt.figure(figsize=(8.4, 10.2))
    grid = fig.add_gridspec(4, 2, height_ratios=[3.2, 0.95, 3.2, 0.95], hspace=0.06, wspace=0.34)
    axes = {
        "FHC_app": (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])),
        "RHC_app": (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])),
        "FHC_dis": (fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[3, 0])),
        "RHC_dis": (fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[3, 1])),
    }
    draw_pair(*axes["FHC_app"], df, "FHC_app", r"$\nu_e$ Appearance", model_label, point_label, app=True, anti=False, bottom_mode=args.bottom_mode)
    draw_pair(*axes["RHC_app"], df, "RHC_app", r"$\bar{\nu}_e$ Appearance", model_label, point_label, app=True, anti=True, bottom_mode=args.bottom_mode)
    draw_pair(*axes["FHC_dis"], df, "FHC_dis", r"$\nu_\mu$ Disappearance", model_label, point_label, app=False, anti=False, bottom_mode=args.bottom_mode)
    draw_pair(*axes["RHC_dis"], df, "RHC_dis", r"$\bar{\nu}_\mu$ Disappearance", model_label, point_label, app=False, anti=True, bottom_mode=args.bottom_mode)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.97)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    plt.close(fig)
    print_checks(df)
    plot_raw_probabilities(raw_probability_path(args.input), raw_probability_figure_path(args.out))
    print(f"Figure sauvegardee: {args.out.resolve()}")


if __name__ == "__main__":
    main()
