#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_mu3.png"

# Non-singularity criterion for mu3:
# keep points with sufficiently large sigma_min.
MU3_SIGMA_MIN_MIN_EV = 1e-14
MU3_SINGULAR_VALUES_HIST_BINS = 140


def require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes mu3 manquantes: {missing}")


def build_mu3_observables(df):
    required = [
        "muH11_eV", "muH12_eV", "muH21_eV", "muH22_eV",
        "muH01_eV", "muH02_eV", "mu00_eV",
    ]
    require_columns(df, required)

    # Build the full (n, 3, 3) array in one vectorized shot — no Python loop.
    h11 = df["muH11_eV"].to_numpy(dtype=float)
    h12 = df["muH12_eV"].to_numpy(dtype=float)
    h21 = df["muH21_eV"].to_numpy(dtype=float)
    h22 = df["muH22_eV"].to_numpy(dtype=float)
    h01 = df["muH01_eV"].to_numpy(dtype=float)
    h02 = df["muH02_eV"].to_numpy(dtype=float)
    mu00 = df["mu00_eV"].to_numpy(dtype=float)

    n = len(df)
    # Stack into shape (n, 3, 3) directly via column arrays
    mu3 = np.stack([
        np.column_stack([h11, h12, h01]),
        np.column_stack([h21, h22, h02]),
        np.column_stack([h01, h02, mu00]),
    ], axis=1)  # (n, 3, 3)

    mu_abs = np.abs(mu3)

    # np.linalg.svd supports stacked matrices since NumPy 1.14 — one call for all n.
    singular_values = np.linalg.svd(mu3, compute_uv=False)  # (n, 3), descending

    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, -1]
    condition_numbers = sigma_max / np.where(sigma_min > 1e-15, sigma_min, 1e-15)

    mu_h0_norm = np.sqrt(h01 ** 2 + h02 ** 2)
    mu_h_trace = h11 + h22
    mu_h_det = h11 * h22 - h12 * h21

    return {
        "mu_abs": mu_abs,
        "singular_values": singular_values,
        "condition_numbers": condition_numbers,
        "mu_h0_norm": mu_h0_norm,
        "mu_h_trace": mu_h_trace,
        "mu_h_det": mu_h_det,
        "mu00": mu00,
    }


def log_bins(values, nbins=70):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return nbins
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmin, vmax):
        vmin *= 0.9
        vmax *= 1.1
    return np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)


def build_nonsingular_mask(observables):
    sigma_min = observables["singular_values"][:, -1]
    return (
        np.isfinite(sigma_min)
        & (sigma_min >= MU3_SIGMA_MIN_MIN_EV)
    )


def filter_observables(observables, mask):
    return {key: np.asarray(values)[mask] for key, values in observables.items()}


def robust_limits(values, lower_q=0.5, upper_q=99.5, positive_only=False):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if positive_only:
        vals = vals[vals > 0.0]
    if vals.size == 0:
        return None

    lo = np.percentile(vals, lower_q)
    hi = np.percentile(vals, upper_q)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    if positive_only:
        lo = max(lo, np.finfo(float).tiny)

    if hi <= lo:
        span = max(abs(lo), 1.0) * 0.1
        lo -= span
        hi += span
        if positive_only:
            lo = max(lo, np.finfo(float).tiny)

    return lo, hi


def clip_1d(values, limits, positive_only=False):
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    if positive_only:
        valid &= vals > 0.0

    lo, hi = limits
    in_frame = valid & (vals >= lo) & (vals <= hi)
    shown = vals[in_frame]
    hidden_count = int(valid.sum() - in_frame.sum())
    shown_count = int(in_frame.sum())
    return shown, shown_count, hidden_count


def clip_2d(x_values, y_values, x_limits, y_limits):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x_lo, x_hi = x_limits
    y_lo, y_hi = y_limits
    in_frame = valid & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    
    shown_x = x[in_frame]
    shown_y = y[in_frame]
    hidden_count = int(valid.sum() - in_frame.sum())
    shown_count = int(in_frame.sum())
    return shown_x, shown_y, shown_count, hidden_count


def hidden_ratio_percent(shown_count, hidden_count):
    if shown_count <= 0:
        return np.inf if hidden_count > 0 else 0.0
    return 100.0 * hidden_count / shown_count


def ratio_suffix(shown_count, hidden_count):
    ratio = hidden_ratio_percent(shown_count, hidden_count)
    if np.isinf(ratio):
        return "(inf%)"
    return f"({ratio:.1f}%)"


def should_use_log_scale(values, positive_only=True, min_decades=3.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if positive_only:
        vals = vals[vals > 0.0]
    if vals.size < 20:
        return False

    lo = np.percentile(vals, 1.0)
    hi = np.percentile(vals, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0.0:
        return False
    lo = max(lo, np.finfo(float).tiny)
    if hi <= lo:
        return False

    decades = np.log10(hi / lo)
    return decades >= min_decades


def choose_adaptive_limits(
    values,
    preferred_limits,
    *,
    positive_only=False,
    lower_q=0.5,
    upper_q=99.5,
    min_fill_fraction=0.25,
):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if positive_only:
        vals = vals[vals > 0.0]

    if vals.size == 0:
        return preferred_limits

    data_min = float(vals.min())
    data_max = float(vals.max())
    pref_min, pref_max = preferred_limits

    if not np.isfinite(pref_min) or not np.isfinite(pref_max) or pref_max <= pref_min:
        limits = robust_limits(vals, lower_q=lower_q, upper_q=upper_q, positive_only=positive_only)
        return limits if limits is not None else preferred_limits

    outside = (data_min < pref_min) or (data_max > pref_max)

    if positive_only and pref_min > 0.0 and pref_max > 0.0 and data_min > 0.0 and data_max > 0.0:
        pref_span = np.log10(pref_max / pref_min) if pref_max > pref_min else np.inf
        data_span = np.log10(data_max / data_min) if data_max > data_min else 0.0
    else:
        pref_span = pref_max - pref_min
        data_span = data_max - data_min

    too_loose = (pref_span > 0.0) and ((data_span / pref_span) < min_fill_fraction)

    if outside or too_loose:
        limits = robust_limits(vals, lower_q=lower_q, upper_q=upper_q, positive_only=positive_only)
        return limits if limits is not None else preferred_limits

    return preferred_limits


def set_hist_ylim_clipped(ax, hist_arrays, clip_percentile=97.0):
    heights = []
    for h in hist_arrays:
        arr = np.asarray(h, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size:
            heights.append(arr)

    if not heights:
        return

    all_heights = np.concatenate(heights)
    y_top = np.percentile(all_heights, clip_percentile)
    if np.isfinite(y_top) and y_top > 0.0:
        ax.set_ylim(0.0, 1.15 * y_top)


def plot_mu3_panels(axes, observables, subtitle, shared_limits=None):
    if shared_limits is None:
        shared_limits = {}

    mu_abs = observables["mu_abs"]
    singular_values = observables["singular_values"]
    condition_numbers = observables["condition_numbers"]

    mu_h11 = mu_abs[:, 0, 0]
    mu_h12 = mu_abs[:, 0, 1]
    mu_h22 = mu_abs[:, 1, 1]
    mu_h_all = np.concatenate([mu_h11, mu_h12, mu_h22])
    mu_h_log = should_use_log_scale(mu_h_all, positive_only=True)
    if mu_h_log:
        mu_h_bins = log_bins(mu_h_all, nbins=80)
        axes[0].set_xscale("log")
    else:
        finite = mu_h_all[np.isfinite(mu_h_all)]
        if finite.size > 0:
            mu_h_bins = np.linspace(finite.min(), finite.max(), 80)
        else:
            mu_h_bins = 80

    n1, _, _ = axes[0].hist(mu_h11, bins=mu_h_bins, density=True, alpha=0.75, label=r"$|\mu_{H,11}|$", color="#1f77b4")
    n2, _, _ = axes[0].hist(mu_h12, bins=mu_h_bins, density=True, alpha=0.75, label=r"$|\mu_{H,12}|$", color="#ff7f0e")
    n3, _, _ = axes[0].hist(mu_h22, bins=mu_h_bins, density=True, alpha=0.75, label=r"$|\mu_{H,22}|$", color="#2ca02c")
    axes[0].set_title(f"Bloc $\\mu_H$ - {subtitle}")
    axes[0].set_xlabel(r"$|\mu_H|$ [eV]")
    axes[0].set_ylabel("Densité")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    mu_h0_values = observables["mu_h0_norm"]
    mu_h0_xlim = shared_limits.get("mu_h0_xlim")
    if mu_h0_xlim is None:
        mu_h0_xlim = choose_adaptive_limits(mu_h0_values, (1e-1, 1e2), positive_only=True)
    mu_h0_bins = log_bins(mu_h0_values, nbins=90)
    axes[1].set_xscale("log")
    n_h0, _, _ = axes[1].hist(mu_h0_values, bins=mu_h0_bins, density=True, color="#df1212", alpha=0.8)
    axes[1].set_title(rf"Norme de $\mu_{{H0}}$ - {subtitle}")
    axes[1].set_xlabel(r"$\|\mu_{H0}\|$ [eV]")
    axes[1].set_ylabel("Densité")
    axes[1].grid(alpha=0.25)

    mu00_abs = np.abs(observables["mu00"])
    mu00_bins = log_bins(mu00_abs, nbins=80)
    axes[2].set_xscale("log")
    n_mu00, _, _ = axes[2].hist(mu00_abs, bins=mu00_bins, density=True, color="#df1212", alpha=0.8)
    mu00_xlim = shared_limits.get("mu00_xlim")
    if mu00_xlim is None:
        mu00_xlim = choose_adaptive_limits(mu00_abs, (1e-2, 1e-1), positive_only=True)
    axes[2].set_title(rf"Distribution de $|\mu_{{00}}|$ - {subtitle}")
    axes[2].set_xlabel(r"$|\mu_{00}|$ [eV]")
    axes[2].set_ylabel("Densité")
    axes[2].set_xscale("log")
    axes[2].grid(alpha=0.25)

    sv_labels = [r"$\sigma_1(\mu_3)$", r"$\sigma_2(\mu_3)$", r"$\sigma_3(\mu_3)$"]
    sv_colors = ["red", "blue", "green"]
    sv_limits = robust_limits(singular_values.flatten(), lower_q=0.5, upper_q=99.5, positive_only=True)
    if sv_limits is not None:
        sv_bins = np.linspace(sv_limits[0], sv_limits[1], MU3_SINGULAR_VALUES_HIST_BINS)
    else:
        finite_sv = singular_values[np.isfinite(singular_values)]
        if finite_sv.size > 0:
            sv_bins = np.linspace(finite_sv.min(), finite_sv.max(), MU3_SINGULAR_VALUES_HIST_BINS)
        else:
            sv_bins = MU3_SINGULAR_VALUES_HIST_BINS
    sv_hist_heights = []
    for idx in range(3):
        n_sv, _, _ = axes[3].hist(
            singular_values[:, idx],
            bins=sv_bins,
            density=True,
            alpha=0.45,
            label=sv_labels[idx],
            color=sv_colors[idx],
        )
        sv_hist_heights.append(n_sv)
    axes[3].set_xscale("linear")
    axes[3].set_title(rf"Valeurs singulières de $\mu_3$ - {subtitle}")
    axes[3].set_xlabel(r"$\sigma_i(\mu_3)$ [eV]")
    axes[3].set_ylabel("Densité")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=8)

    kappa_bins = log_bins(condition_numbers, nbins=80)
    n_kappa, _, _ = axes[4].hist(
        condition_numbers,
        bins=kappa_bins,
        density=True,
        color="#df1212",
        alpha=0.8,
    )
    axes[4].set_xscale("log")
    finite_kappa = condition_numbers[np.isfinite(condition_numbers) & (condition_numbers > 0.0)]
    if finite_kappa.size > 0:
        kappa_lo = max(1e12, float(np.percentile(finite_kappa, 0.5)))
        kappa_hi = float(finite_kappa.max()) * 1.05
        if kappa_hi <= kappa_lo:
            kappa_hi = kappa_lo * 10.0
        kappa_xlim = (kappa_lo, kappa_hi)
    else:
        kappa_xlim = (1e12, 1e16)
    axes[4].set_title(rf"$\kappa(\mu_3)=\sigma_{{max}}/\sigma_{{min}}$ - {subtitle}")
    axes[4].set_xlabel(r"$\kappa(\mu_3)$")
    axes[4].set_ylabel("Densité")
    axes[4].grid(alpha=0.25)

    trace_values = observables["mu_h_trace"]
    det_values = observables["mu_h_det"]
    valid_trace_det = np.isfinite(trace_values) & np.isfinite(det_values)
    trace_all = trace_values[valid_trace_det]
    det_all = det_values[valid_trace_det]

    axes[5].scatter(trace_all, det_all, s=6, alpha=0.25, color="#df1212")
    if should_use_log_scale(trace_all, positive_only=True):
        axes[5].set_xscale("log")
    if should_use_log_scale(det_all, positive_only=True):
        axes[5].set_yscale("log")
    axes[5].set_title(rf"Trace vs déterminant de $\mu_H$ - {subtitle}")
    axes[5].set_xlabel(r"Tr$(\mu_H)$ [eV]")
    axes[5].set_ylabel(r"det$(\mu_H)$ [eV$^2$]")
    axes[5].grid(alpha=0.25)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier de points conservés introuvable: {DATA_DIR}")

    df = load_kept_points_dataframe(DATA_DIR)
    if df.empty:
        raise RuntimeError("Aucun point conservé trouvé.")

    if "pmns_pass" not in df.columns:
        df["pmns_pass"] = 1
    if "eta_pass" not in df.columns:
        df["eta_pass"] = 0

    df_pmns = df[df["pmns_pass"] == 1].copy()
    if df_pmns.empty:
        raise RuntimeError("Aucun point avec pmns_pass == 1.")

    df_pmns_eta = df[(df["pmns_pass"] == 1) & (df["eta_pass"] == 1)].copy()
    if df_pmns_eta.empty:
        raise RuntimeError("Aucun point avec pmns_pass == 1 et eta_pass == 1.")

    observables_pmns_raw = build_mu3_observables(df_pmns)
    observables_pmns_eta_raw = build_mu3_observables(df_pmns_eta)

    pmns_mask = build_nonsingular_mask(observables_pmns_raw)
    pmns_eta_mask = build_nonsingular_mask(observables_pmns_eta_raw)

    observables_pmns = filter_observables(observables_pmns_raw, pmns_mask)
    observables_pmns_eta = filter_observables(observables_pmns_eta_raw, pmns_eta_mask)

    shared_limits = {
        "mu00_xlim": choose_adaptive_limits(
            np.abs(observables_pmns["mu00"]),
            (1e-2, 1e-1),
            positive_only=True,
        ),
        "mu_h0_xlim": choose_adaptive_limits(
            observables_pmns["mu_h0_norm"],
            (1e-1, 1e2),
            positive_only=True,
        ),
    }

    n_pmns_before = len(df_pmns)
    n_pmns_after = int(pmns_mask.sum())
    n_pmns_eta_before = len(df_pmns_eta)
    n_pmns_eta_after = int(pmns_eta_mask.sum())

    if n_pmns_after == 0:
        raise RuntimeError("Le filtre de non-singularité retire tous les points PMNS OK.")
    if n_pmns_eta_after == 0:
        raise RuntimeError("Le filtre de non-singularité retire tous les points PMNS+ETA OK.")

    print(
        "[mu3-filter] criterion: "
        f"sigma_min >= {MU3_SIGMA_MIN_MIN_EV:.1e} eV"
    )
    print(f"PMNS OK: n_pmns_before kept")
    print(f"PMNS+ETA OK: n_pmns_eta_before kept")

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.reshape(4, 3)

    plot_mu3_panels(axes[:2, :].flatten(), observables_pmns, "PMNS OK", shared_limits=shared_limits)
    plot_mu3_panels(axes[2:, :].flatten(), observables_pmns_eta, "PMNS+ETA OK", shared_limits=shared_limits)

    fig.suptitle(
        f"PMNS OK: {n_pmns_after}/{n_pmns_before} | "
        f"PMNS+ETA OK: {n_pmns_eta_after}/{n_pmns_eta_before} | "
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
