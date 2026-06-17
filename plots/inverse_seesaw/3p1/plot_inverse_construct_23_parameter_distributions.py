#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_construct_23_config import get_inverse_kept_points_dir
from inverse_construct_23_kept_points import load_kept_points_dataframe

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "figures" / "inverse_seesaw" / "3p1" / "inverse_construct_23_parameter_distributions.png"
DATA_DIR = get_inverse_kept_points_dir()
F_PARAMETER_REPRESENTATION = "abs_phase"  # "abs_phase" or "re_im"


def to_degrees_if_radians(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    # Backward compatibility: older CSVs saved phi_zeta in radians
    # while using the "*_deg" field name.
    max_abs = np.max(np.abs(finite))
    if max_abs <= (2.0 * np.pi + 1e-6):
        return np.rad2deg(arr)
    return arr


def optional_column(df, key: str):
    if key in df.columns:
        return df[key].values
    return np.zeros(len(df), dtype=float)


def f_components(df, fij: str):
    mag = df[fij].values
    phase_rad = np.deg2rad(df[f"{fij}_phase_deg"].values)
    real = mag * np.cos(phase_rad)
    imag = mag * np.sin(phase_rad)
    return real, imag


def build_f_parameters(df_eta_pass, df_eta_fail):
    if F_PARAMETER_REPRESENTATION == "abs_phase":
        return {
            'f11_abs': (df_eta_pass['f11'].values, df_eta_fail['f11'].values, r'$|f_{11}|$'),
            'f12_abs': (df_eta_pass['f12'].values, df_eta_fail['f12'].values, r'$|f_{12}|$'),
            'f21_abs': (df_eta_pass['f21'].values, df_eta_fail['f21'].values, r'$|f_{21}|$'),
            'f22_abs': (df_eta_pass['f22'].values, df_eta_fail['f22'].values, r'$|f_{22}|$'),
            'phi_f11': (df_eta_pass['f11_phase_deg'].values, df_eta_fail['f11_phase_deg'].values, r'$\phi_{f_{11}}$ [deg]'),
            'phi_f12': (df_eta_pass['f12_phase_deg'].values, df_eta_fail['f12_phase_deg'].values, r'$\phi_{f_{12}}$ [deg]'),
            'phi_f21': (df_eta_pass['f21_phase_deg'].values, df_eta_fail['f21_phase_deg'].values, r'$\phi_{f_{21}}$ [deg]'),
            'phi_f22': (df_eta_pass['f22_phase_deg'].values, df_eta_fail['f22_phase_deg'].values, r'$\phi_{f_{22}}$ [deg]'),
        }

    if F_PARAMETER_REPRESENTATION == "re_im":
        f11_re_pass, f11_im_pass = f_components(df_eta_pass, 'f11')
        f11_re_fail, f11_im_fail = f_components(df_eta_fail, 'f11')
        f12_re_pass, f12_im_pass = f_components(df_eta_pass, 'f12')
        f12_re_fail, f12_im_fail = f_components(df_eta_fail, 'f12')
        f21_re_pass, f21_im_pass = f_components(df_eta_pass, 'f21')
        f21_re_fail, f21_im_fail = f_components(df_eta_fail, 'f21')
        f22_re_pass, f22_im_pass = f_components(df_eta_pass, 'f22')
        f22_re_fail, f22_im_fail = f_components(df_eta_fail, 'f22')

        return {
            'f11_re': (f11_re_pass, f11_re_fail, r'$\Re(f_{11})$'),
            'f11_im': (f11_im_pass, f11_im_fail, r'$\Im(f_{11})$'),
            'f12_re': (f12_re_pass, f12_re_fail, r'$\Re(f_{12})$'),
            'f12_im': (f12_im_pass, f12_im_fail, r'$\Im(f_{12})$'),
            'f21_re': (f21_re_pass, f21_re_fail, r'$\Re(f_{21})$'),
            'f21_im': (f21_im_pass, f21_im_fail, r'$\Im(f_{21})$'),
            'f22_re': (f22_re_pass, f22_re_fail, r'$\Re(f_{22})$'),
            'f22_im': (f22_im_pass, f22_im_fail, r'$\Im(f_{22})$'),
        }

    raise ValueError("F_PARAMETER_REPRESENTATION must be 'abs_phase' or 're_im'.")


def load_distributions():
    """Load kept points and extract parameter distributions (PMNS-kept points only)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Kept points directory not found: {DATA_DIR}")

    df_all = load_kept_points_dataframe(DATA_DIR)
    print(f"Total points found in CSV: {len(df_all)}")

    required_columns = ["pmns_pass", "eta_pass"]
    missing = [col for col in required_columns if col not in df_all.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns in construct_23 dataset: "
            + ", ".join(missing)
            + ". Regenerate the CSV with pmns_pass/eta_pass metadata."
        )

    df_all["pmns_pass"] = df_all["pmns_pass"].fillna(0).astype(int)
    df_all["eta_pass"] = df_all["eta_pass"].fillna(0).astype(int)

    df_pmns = df_all[df_all["pmns_pass"] == 1].copy()
    n_total = len(df_all)
    n_pmns = len(df_pmns)
    print(f"Points PMNS OK: {n_pmns}")

    if n_total > 0 and n_pmns == n_total:
        print(
            "[info] All loaded points have pmns_pass=1. "
            "This usually means the input dataset is already PMNS-filtered."
        )

    if n_pmns == 0:
        raise RuntimeError("No pmns_pass=1 points found!")
    
    # Separate pmns_pass=1 points by eta_pass status
    df_eta_pass = df_pmns[df_pmns['eta_pass'] == 1].copy()
    df_eta_fail = df_pmns[df_pmns['eta_pass'] == 0].copy()
    
    n_eta_pass = len(df_eta_pass)
    n_eta_fail = len(df_eta_fail)

    zeta_direction_eta_pass_deg = to_degrees_if_radians(df_eta_pass['zeta_direction_deg'].values)
    zeta_direction_eta_fail_deg = to_degrees_if_radians(df_eta_fail['zeta_direction_deg'].values)
    
    # Extract parameter distributions: red=pmns+eta, blue=pmns only
    parameters = {
        'dm41': (df_eta_pass['dm41_target_eV2'].values, df_eta_fail['dm41_target_eV2'].values, r'$\Delta m_{41}$ [eV$^2$]'),
        'zeta_norm': (df_eta_pass['zeta_norm'].values, df_eta_fail['zeta_norm'].values, r'$\|\zeta\|$'),
        'zeta_direction': (zeta_direction_eta_pass_deg, zeta_direction_eta_fail_deg, r'$\phi_\zeta$ [deg]'),
        'majorana_alpha21': (optional_column(df_eta_pass, 'majorana_alpha21_deg'), optional_column(df_eta_fail, 'majorana_alpha21_deg'), r'$\alpha_{21}$ [deg]'),
        'majorana_alpha31': (optional_column(df_eta_pass, 'majorana_alpha31_deg'), optional_column(df_eta_fail, 'majorana_alpha31_deg'), r'$\alpha_{31}$ [deg]'),
        'M1': (df_eta_pass['M1_GeV'].values, df_eta_fail['M1_GeV'].values, r'$M_1$ [GeV]'),
        'M2': (df_eta_pass['M2_GeV'].values, df_eta_fail['M2_GeV'].values, r'$M_2$ [GeV]'),
    }

    parameters.update(build_f_parameters(df_eta_pass, df_eta_fail))
    
    return parameters, n_total, n_pmns, n_eta_pass, n_eta_fail


def plot_parameter(ax, label, eta_pass_values, eta_fail_values, xlabel):
    """Plot stacked histogram for a single parameter, colored by eta_pass status"""
    values = list(eta_pass_values) + list(eta_fail_values)
    if len(values) > 0:
        # Use sqrt of sample size to determine bin count, capped between 35 and 120
        bins = max(35, min(120, int(len(values) ** 0.5)))
        
        ax.hist(
            [eta_pass_values, eta_fail_values],
            bins=bins,
            density=True,
            stacked=True,
            color=["red", "#1f77b4"],
            alpha=0.8,
            linewidth=0.6,
            edgecolor="black",
            label=["PMNS+ETA OK", "PMNS OK mais ETA KO"],
        )
    
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité de probabilité")
    ax.grid(alpha=0.25)


def main():
    parameters, n_total, n_pmns, n_eta_pass, n_eta_fail = load_distributions()
    
    # Create figure with subplots (3 columns, auto rows)
    ncols = 3
    param_list = list(parameters.items())
    nrows = (len(param_list) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.0 * nrows))
    axes_flat = axes.flatten()
    
    # Plot each parameter
    for ax, (param_name, (eta_pass_vals, eta_fail_vals, xlabel)) in zip(axes_flat, param_list):
        plot_parameter(ax, param_name, eta_pass_vals, eta_fail_vals, xlabel)

    if len(param_list) > 0:
        axes_flat[0].legend(loc="upper right", fontsize=8)
    
    # Turn off unused subplots
    for ax in axes_flat[len(param_list):]:
        ax.axis("off")
    
    # Set main title
    if n_pmns == n_total:
        title = (
            f"Distributions des paramètres (échantillon déjà PMNS OK: {n_pmns}/{n_total}; "
            f"+eta OK {n_eta_pass}/{n_pmns})"
        )
    else:
        title = (
            f"Distributions des paramètres ({n_pmns}/{n_total} PMNS OK; "
            f"+eta OK {n_eta_pass}/{n_pmns})"
        )
    fig.suptitle(title)
    
    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight')
    print(f"Figure sauvegardée: {OUTPUT_PATH}")

    plt.close(fig)


if __name__ == "__main__":
    main()
