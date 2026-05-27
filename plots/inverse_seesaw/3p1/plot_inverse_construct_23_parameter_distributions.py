#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_construct_23_sterile_parameter_distributions.png")
CSV_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_3p1.csv")


def load_distributions():
    """Load CSV and extract parameter distributions (solve_ok=1 only)"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows in CSV: {len(df)}")
    
    # Filter to pmns_pass=1 points only (primary filter)
    df_pmns = df[(df['solve_ok'] == 1) & (df['pmns_pass'] == 1)].copy()
    
    n_pmns = len(df_pmns)
    print(f"Points PMNS OK (pmns_pass=1): {n_pmns}")
    
    if n_pmns == 0:
        raise RuntimeError("No pmns_pass=1 points found!")
    
    # Separate pmns_pass=1 points by eta_pass status
    df_eta_pass = df_pmns[df_pmns['eta_pass'] == 1].copy()
    df_eta_fail = df_pmns[df_pmns['eta_pass'] == 0].copy()
    
    n_eta_pass = len(df_eta_pass)
    n_eta_fail = len(df_eta_fail)
    
    # Extract parameter distributions: red=pmns+eta, blue=pmns only
    parameters = {
        'dm41': (df_eta_pass['dm41_target_eV2'].values, df_eta_fail['dm41_target_eV2'].values, r'$\Delta m_{41}$ [eV$^2$]'),
        'zeta_norm': (df_eta_pass['zeta_norm'].values, df_eta_fail['zeta_norm'].values, r'$\|\zeta\|$'),
        'zeta_direction': (df_eta_pass['zeta_direction_deg'].values, df_eta_fail['zeta_direction_deg'].values, r'$\phi_\zeta$ [deg]'),
        'f11': (df_eta_pass['f11'].values, df_eta_fail['f11'].values, r'$f_{11}$'),
        'f12': (df_eta_pass['f12'].values, df_eta_fail['f12'].values, r'$f_{12}$'),
        'f21': (df_eta_pass['f21'].values, df_eta_fail['f21'].values, r'$f_{21}$'),
        'f22': (df_eta_pass['f22'].values, df_eta_fail['f22'].values, r'$f_{22}$'),
        'M1': (df_eta_pass['M1_GeV'].values, df_eta_fail['M1_GeV'].values, r'$M_1$ [GeV]'),
        'M2': (df_eta_pass['M2_GeV'].values, df_eta_fail['M2_GeV'].values, r'$M_2$ [GeV]'),
    }
    
    return parameters, n_pmns, n_eta_pass, n_eta_fail


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
        )
    
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité de probabilité")
    ax.grid(alpha=0.25)


def main():
    parameters, n_pmns, n_eta_pass, n_eta_fail = load_distributions()
    
    # Create figure with subplots (3 columns, auto rows)
    ncols = 3
    param_list = list(parameters.items())
    nrows = (len(param_list) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.0 * nrows))
    axes_flat = axes.flatten()
    
    # Plot each parameter
    for ax, (param_name, (eta_pass_vals, eta_fail_vals, xlabel)) in zip(axes_flat, param_list):
        plot_parameter(ax, param_name, eta_pass_vals, eta_fail_vals, xlabel)
    
    # Turn off unused subplots
    for ax in axes_flat[len(param_list):]:
        ax.axis("off")
    
    # Set main title
    fig.suptitle(
        f"Distributions des paramètres construct_23 ({n_pmns} pts PMNS OK — Rouge: +η OK ({n_eta_pass}), Bleu: η non OK ({n_eta_fail}))"
    )
    
    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight')
    print(f"Figure sauvegardée: {OUTPUT_PATH}")

    plt.close(fig)


if __name__ == "__main__":
    main()
