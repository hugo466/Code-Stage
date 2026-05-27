#!/usr/bin/env python3
"""
Plot parameter distributions for construct_23 filtered points by eta regime.
Similar to inverse_pmns_filter parameter distributions but for construct_23 with sterile angles.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
FILTERED_DIR = Path("data/inverse_seesaw/3p1/inverse_construct_23_filtered")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_construct_23_regime_distributions.png")


def load_regime_data():
    """Load filtered CSV files by regime"""
    regimes = {}
    for regime in ['light_low', 'intermediate']:
        csv_file = FILTERED_DIR / f"inverse_construct_23_{regime}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            regimes[regime] = df
            print(f"Loaded {regime:15s}: {len(df)} points")
    
    if not regimes:
        raise RuntimeError("No regime CSV files found!")
    
    return regimes


def plot_parameter_by_regime(ax, label, regimes, column, xlabel, color_map=None):
    """Plot histogram overlaid for multiple regimes"""
    if color_map is None:
        color_map = {'light_low': 'red', 'intermediate': '#1f77b4'}
    
    data_list = []
    labels_list = []
    colors = []
    
    for regime in ['light_low', 'intermediate']:
        if regime in regimes:
            data = regimes[regime][column].dropna().values
            if len(data) > 0:
                data_list.append(data)
                labels_list.append(regime)
                colors.append(color_map[regime])
    
    if data_list:
        # Determine bins from combined data
        all_data = np.concatenate(data_list)
        bins = max(35, min(120, int(len(all_data) ** 0.5)))
        
        ax.hist(
            data_list,
            bins=bins,
            density=True,
            stacked=True,
            color=colors,
            alpha=0.8,
            linewidth=0.6,
            edgecolor="black",
            label=labels_list
        )
    
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité de probabilité")
    ax.grid(alpha=0.25)


def main():
    regimes = load_regime_data()
    
    # Parameter specifications
    params = [
        ('dm41_target_eV2', r'$\Delta m_{41}$ [eV$^2$]'),
        ('zeta_norm', r'$\|\zeta\|$'),
        ('zeta_direction_deg', r'$\phi_\zeta$ [deg]'),
        ('f11', r'$f_{11}$'),
        ('f12', r'$f_{12}$'),
        ('f21', r'$f_{21}$'),
        ('f22', r'$f_{22}$'),
        ('M1_GeV', r'$M_1$ [GeV]'),
        ('M2_GeV', r'$M_2$ [GeV]'),
    ]
    
    ncols = 3
    nrows = (len(params) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.0 * nrows))
    axes_flat = axes.flatten()
    
    # Plot each parameter
    for ax, (column, xlabel) in zip(axes_flat, params):
        plot_parameter_by_regime(ax, column, regimes, column, xlabel)
    
    # Turn off unused subplots
    for ax in axes_flat[len(params):]:
        ax.axis("off")
    
    # Add legend
    axes_flat[0].legend(
        ["Stérile léger bas (dm41 ∈ [0.1, 1.0])", "Intermédiaire (dm41 ∈ [1.0, 100.0])"],
        loc="best",
        framealpha=0.9,
    )
    
    # Title
    total_points = sum(len(df) for df in regimes.values())
    light_low = len(regimes.get('light_low', []))
    intermediate = len(regimes.get('intermediate', []))
    
    fig.suptitle(
        f"construct_23 filtrés par critères eta (Total: {total_points} points, Rouge: {light_low}, Bleu: {intermediate})",
        fontsize=14,
        fontweight='bold'
    )
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight')
    print(f"\nFigure sauvegardée: {OUTPUT_PATH}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
