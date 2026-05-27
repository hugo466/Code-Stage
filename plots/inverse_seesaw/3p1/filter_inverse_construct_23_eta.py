#!/usr/bin/env python3
"""
Filter construct_23 3+1 inverse seesaw scanning points based on:
1. Non-unitarity constraints (eta_pass from C code)
2. DM41 regime classification (light low, light high, heavy)

Similar to inverse_pmns_filter but for construct_23 with sterile angles.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
CSV_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_3p1.csv")
REGIME_CONFIG = {
    'dm41_light_low_min': 0.1,      # eV^2
    'dm41_light_low_max': 1.0,      # eV^2
    'dm41_light_high_min': 100.0,   # eV^2
}

# Output directories
OUTPUT_DIR_BASE = Path("data/inverse_seesaw/3p1/inverse_construct_23_filtered")


class RegimeClassifier:
    """Classify points into dm41 regimes"""
    
    def __init__(self, config):
        self.dm41_low_min = config['dm41_light_low_min']
        self.dm41_low_max = config['dm41_light_low_max']
        self.dm41_high_min = config['dm41_light_high_min']
    
    def classify(self, dm41):
        """Return regime name"""
        if dm41 < self.dm41_low_min:
            return None  # Below valid range
        elif dm41 <= self.dm41_low_max:
            return 'light_low'
        elif dm41 < self.dm41_high_min:
            return 'intermediate'  # Between light regimes
        else:
            return 'light_high'
    
    def all_regimes(self):
        return ['light_low', 'intermediate', 'light_high']


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoading construct_23 CSV: {CSV_PATH}")
    print(f"Total points: {len(df)}")
    
    # Ensure columns exist
    if 'solve_ok' not in df.columns or 'eta_pass' not in df.columns or 'dm41_target_eV2' not in df.columns:
        raise ValueError("CSV missing required columns: solve_ok, eta_pass, dm41_target_eV2")
    
    # Filter solved points with eta constraints satisfied
    df_solved = df[df['solve_ok'] == 1].copy()
    print(f"Solved points (solve_ok=1): {len(df_solved)}")
    
    df_eta_pass = df_solved[df_solved['eta_pass'] == 1].copy()
    print(f"Points with eta constraints satisfied: {len(df_eta_pass)}")
    
    if len(df_eta_pass) == 0:
        print("WARNING: No points pass eta constraints!")
        return
    
    # Classify by dm41 regime
    classifier = RegimeClassifier(REGIME_CONFIG)
    df_eta_pass['regime'] = df_eta_pass['dm41_target_eV2'].apply(classifier.classify)
    
    # Create output directory
    OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    
    # Save points by regime
    print(f"\nSaving filtered points by regime:")
    regime_counts = {}
    for regime in classifier.all_regimes():
        df_regime = df_eta_pass[df_eta_pass['regime'] == regime]
        if len(df_regime) > 0:
            regime_counts[regime] = len(df_regime)
            # Save as individual text files (similar to PMNS_filter)
            regime_dir = OUTPUT_DIR_BASE / regime
            regime_dir.mkdir(parents=True, exist_ok=True)
            
            for _, row in df_regime.iterrows():
                point_id = int(row['point_id'])
                point_file = regime_dir / f"{point_id}.txt"
                
                # Write point details
                with open(point_file, 'w') as f:
                    f.write("# construct_23 point with sterile angle parametrization\n")
                    f.write(f"point_id = {point_id}\n")
                    f.write(f"regime = {regime}\n")
                    f.write(f"eta_pass = {row['eta_pass']}\n")
                    f.write(f"dm41_target_eV2 = {row['dm41_target_eV2']}\n")
                    f.write(f"dm21_target_eV2 = {row['dm21_target_eV2']}\n")
                    f.write(f"dm31_target_eV2 = {row['dm31_target_eV2']}\n")
                    f.write(f"dm21_calc_eV2 = {row['dm21_calc_eV2']}\n")
                    f.write(f"dm31_calc_eV2 = {row['dm31_calc_eV2']}\n")
                    f.write(f"dm41_calc_eV2 = {row['dm41_calc_eV2']}\n")
                    f.write(f"theta14_deg = {row['theta14_deg']}\n")
                    f.write(f"theta24_deg = {row['theta24_deg']}\n")
                    f.write(f"theta34_deg = {row['theta34_deg']}\n")
                    f.write(f"delta_cp_sterile_deg = {row['delta_cp_sterile_deg']}\n")
                    f.write(f"f11 = {row['f11']}\n")
                    f.write(f"f12 = {row['f12']}\n")
                    f.write(f"f21 = {row['f21']}\n")
                    f.write(f"f22 = {row['f22']}\n")
                    f.write(f"M1_GeV = {row['M1_GeV']}\n")
                    f.write(f"M2_GeV = {row['M2_GeV']}\n")
            
            print(f"  {regime:15s}: {len(df_regime):6d} points → {regime_dir}/")
    
    # Summary statistics
    print(f"\nRegime summary:")
    print(f"  light_low      (dm41 ∈ [{REGIME_CONFIG['dm41_light_low_min']}, {REGIME_CONFIG['dm41_light_low_max']}]): {regime_counts.get('light_low', 0)} points")
    print(f"  intermediate   (dm41 ∈ [{REGIME_CONFIG['dm41_light_low_max']}, {REGIME_CONFIG['dm41_light_high_min']}]): {regime_counts.get('intermediate', 0)} points")
    print(f"  light_high     (dm41 ≥ {REGIME_CONFIG['dm41_light_high_min']}): {regime_counts.get('light_high', 0)} points")
    print(f"\n  Total with eta constraints: {len(df_eta_pass)} points")
    
    # Also save combined CSV by regime
    for regime in classifier.all_regimes():
        df_regime = df_eta_pass[df_eta_pass['regime'] == regime]
        if len(df_regime) > 0:
            csv_file = OUTPUT_DIR_BASE / f"inverse_construct_23_{regime}.csv"
            df_regime.to_csv(csv_file, index=False)
            print(f"  CSV saved: {csv_file}")


if __name__ == "__main__":
    main()
