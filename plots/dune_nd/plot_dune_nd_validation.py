import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns found: {candidates}\nAvailable: {list(df.columns)}")


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.abs(b) > 0.0
    out[mask] = a[mask] / b[mask]
    return out


def safe_rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.abs(b) > 0.0
    out[mask] = (a[mask] - b[mask]) / b[mask]
    return out


def default_figure_dir(input_dir):
    parts = input_dir.parts
    point_name = None
    for part in reversed(parts):
        if part.startswith("point_"):
            point_name = part
            break
    if point_name is None:
        point_name = input_dir.name
    return Path("figures") / "dune_nd" / point_name / "validation"


def add_energy_center(df):
    if "Erec_GeV" in df.columns:
        return df

    low = find_col(df, ["E_low_GeV", "Erec_low_GeV", "E_rec_low_GeV"])
    high = find_col(df, ["E_high_GeV", "Erec_high_GeV", "E_rec_high_GeV"])
    out = df.copy()
    out["Erec_GeV"] = 0.5 * (out[low].astype(float) + out[high].astype(float))
    return out


def wide_prediction_to_long(pred, null):
    pred = add_energy_center(pred)
    null = add_energy_center(null)

    frames = []
    sample_specs = [
        ("mu_like", "mu_like", "null_mu_like"),
        ("e_like", "e_like", "null_e_like"),
    ]
    for sample, pred_col, null_col in sample_specs:
        if pred_col not in pred.columns or null_col not in null.columns:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Erec_GeV": pred["Erec_GeV"],
                    "sample": sample,
                    "N_pred": pred[pred_col],
                    "N_null": null[null_col],
                }
            )
        )

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def generic_prediction_to_long(pred, null):
    pred = add_energy_center(pred)
    null = add_energy_center(null)

    n_pred = find_col(pred, ["N_pred", "events_pred", "mu", "rate", "counts"])
    n_null = find_col(null, ["N_null", "events_null", "mu_null", "rate_null", "counts"])

    sample_pred = next((c for c in ["sample", "sample_name", "channel"] if c in pred.columns), None)
    sample_null = next((c for c in ["sample", "sample_name", "channel"] if c in null.columns), None)

    if sample_pred and sample_null:
        pred_ren = pred.rename(columns={sample_pred: "sample", n_pred: "N_pred"})
        null_ren = null.rename(columns={sample_null: "sample", n_null: "N_null"})
        return pd.merge(
            pred_ren[["Erec_GeV", "sample", "N_pred"]],
            null_ren[["Erec_GeV", "sample", "N_null"]],
            on=["Erec_GeV", "sample"],
            how="inner",
        )

    merged = pd.merge(
        pred[["Erec_GeV", n_pred]].rename(columns={n_pred: "N_pred"}),
        null[["Erec_GeV", n_null]].rename(columns={n_null: "N_null"}),
        on="Erec_GeV",
        how="inner",
    )
    merged["sample"] = "all"
    return merged


def build_merged_table(pred, null):
    merged = wide_prediction_to_long(pred, null)
    if merged is None:
        merged = generic_prediction_to_long(pred, null)

    merged["ratio_pred_null"] = safe_ratio(merged["N_pred"], merged["N_null"])
    merged["rel_residual"] = safe_rel_diff(merged["N_pred"], merged["N_null"])
    merged["abs_residual"] = merged["N_pred"] - merged["N_null"]
    return merged


def safe_nan_minmax(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.min(finite)), float(np.max(finite))


def plot_sample_figures(merged, outdir):
    for sample, data in merged.groupby("sample"):
        data = data.sort_values("Erec_GeV")

        plt.figure(figsize=(8, 5))
        plt.step(data["Erec_GeV"], data["N_null"], where="mid", label="Null / no sterile")
        plt.step(data["Erec_GeV"], data["N_pred"], where="mid", label="ISS / sterile point")
        plt.xlabel(r"$E_{\rm rec}$ [GeV]")
        plt.ylabel("Expected events / bin")
        plt.title(f"Reconstructed spectrum: {sample}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"spectrum_{sample}.png", dpi=200)
        plt.close()

        finite_ratio = np.isfinite(data["ratio_pred_null"])
        plt.figure(figsize=(8, 4))
        if finite_ratio.any():
            plt.axhline(1.0, color="black", linewidth=1)
            plt.plot(data["Erec_GeV"], data["ratio_pred_null"], marker="o", markersize=3, linewidth=1)
            plt.ylabel(r"$N_{\rm pred}/N_{\rm null}$")
            plt.title(f"Sterile distortion ratio: {sample}")
        else:
            plt.axhline(0.0, color="black", linewidth=1)
            plt.plot(data["Erec_GeV"], data["abs_residual"], marker="o", markersize=3, linewidth=1)
            plt.ylabel(r"$N_{\rm pred}-N_{\rm null}$")
            plt.title(f"Absolute excess, zero null spectrum: {sample}")
        plt.xlabel(r"$E_{\rm rec}$ [GeV]")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"ratio_{sample}.png", dpi=200)
        plt.close()

        finite_residual = np.isfinite(data["rel_residual"])
        plt.figure(figsize=(8, 4))
        plt.axhline(0.0, color="black", linewidth=1)
        if finite_residual.any():
            plt.plot(data["Erec_GeV"], 100.0 * data["rel_residual"], marker="o", markersize=3, linewidth=1)
            plt.ylabel(r"$(N_{\rm pred}-N_{\rm null})/N_{\rm null}$ [%]")
            plt.title(f"Relative residuals: {sample}")
        else:
            plt.plot(data["Erec_GeV"], data["abs_residual"], marker="o", markersize=3, linewidth=1)
            plt.ylabel(r"$N_{\rm pred}-N_{\rm null}$")
            plt.title(f"Absolute residuals, zero null spectrum: {sample}")
        plt.xlabel(r"$E_{\rm rec}$ [GeV]")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"residuals_{sample}.png", dpi=200)
        plt.close()


def plot_summary(merged, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    zero_null_samples = []

    for sample, data in merged.groupby("sample"):
        data = data.sort_values("Erec_GeV")
        axes[0].step(data["Erec_GeV"], data["N_pred"], where="mid", label=f"{sample} pred")
        axes[0].step(data["Erec_GeV"], data["N_null"], where="mid", linestyle="--", label=f"{sample} null")
        if np.isfinite(data["ratio_pred_null"]).any():
            axes[1].plot(data["Erec_GeV"], data["ratio_pred_null"], marker="o", markersize=2, linewidth=1, label=sample)
            axes[2].plot(data["Erec_GeV"], 100.0 * data["rel_residual"], marker="o", markersize=2, linewidth=1, label=sample)
        else:
            zero_null_samples.append(sample)

    axes[0].set_ylabel("Events / bin")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_ylabel(r"$N_{\rm pred}/N_{\rm null}$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Residual [%]")
    axes[2].set_xlabel(r"$E_{\rm rec}$ [GeV]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    if zero_null_samples:
        note = "Relative plots undefined for zero-null samples: " + ", ".join(zero_null_samples)
        axes[1].text(0.01, 0.95, note, transform=axes[1].transAxes, fontsize=8, va="top")
        axes[2].text(0.01, 0.95, note, transform=axes[2].transAxes, fontsize=8, va="top")

    fig.suptitle("DUNE ND first-step validation")
    plt.tight_layout()
    plt.savefig(outdir / "validation_summary.png", dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot first-step DUNE ND validation diagnostics.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dune_nd/minimal_onaxis/point_25"),
        help="Directory containing spectrum_pred.csv, spectrum_null.csv, residuals.csv and point_observables.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for plots. Defaults to figures/dune_nd/<point>/validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    outdir = args.outdir or default_figure_dir(input_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    spectrum_pred_path = input_dir / "spectrum_pred.csv"
    spectrum_null_path = input_dir / "spectrum_null.csv"
    residuals_path = input_dir / "residuals.csv"
    point_observables_path = input_dir / "point_observables.csv"

    pred = pd.read_csv(spectrum_pred_path)
    null = pd.read_csv(spectrum_null_path)
    res = pd.read_csv(residuals_path)
    obs = pd.read_csv(point_observables_path)

    print("spectrum_pred columns:", pred.columns.tolist())
    print("spectrum_null columns:", null.columns.tolist())
    print("residuals columns:", res.columns.tolist())
    print("point_observables columns:", obs.columns.tolist())

    merged = build_merged_table(pred, null)

    ratio_min, ratio_max = safe_nan_minmax(merged["ratio_pred_null"])
    rel_min, rel_max = safe_nan_minmax(merged["rel_residual"])
    zero_null_samples = sorted(
        sample for sample, data in merged.groupby("sample")
        if not np.isfinite(data["ratio_pred_null"]).any()
    )
    print("\nBasic checks:")
    print("N_pred min/max:", merged["N_pred"].min(), merged["N_pred"].max())
    print("N_null min/max:", merged["N_null"].min(), merged["N_null"].max())
    print("ratio min/max:", ratio_min, ratio_max)
    print("relative residual min/max:", rel_min, rel_max)
    if zero_null_samples:
        print("zero-null samples with undefined relative plots:", ", ".join(zero_null_samples))

    plot_sample_figures(merged, outdir)
    plot_summary(merged, outdir)

    print("\nPoint observables:")
    print(obs.head().to_string(index=False))

    if not res.empty:
        print("\nResidual table columns validated:", res.columns.tolist())

    merged.to_csv(outdir / "merged_validation_table.csv", index=False)
    print(f"\nPlots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
