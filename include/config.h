#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

#define MAX_DM41_VALUES 5000
#define MAX_DM54_VALUES 5000
#define MAX_STERILE_NEUTRINOS 8
#define MAX_INVERSE_MU_VALUES 1024
#define MAX_SCAN_VALUES_PER_PARAM 1024

typedef enum {
    OPERATION_UNSET = 0,
    OPERATION_ENERGY_3P1,
    OPERATION_ENERGY_3P2,
    OPERATION_DISTANCE_3P1,
    OPERATION_HEATMAP_DELTA_PMUE,
    OPERATION_HEATMAP_DELTA_PMUE_3P2,
    OPERATION_HEATMAP_DELTA_PMUMU_3P2,
    OPERATION_CP_HEATMAP_3P1,
    OPERATION_INVERSE_SEESAW_3P1,
    OPERATION_INVERSE_PMNS_FILTER_3P1,
    OPERATION_INVERSE_PMNS_FILTER_3P2,
    OPERATION_INVERSE_CONSTRUCT_23_3P1
} SimulationOperation;

typedef struct {
    double baseline_km;
    double energy_min_GeV;
    double energy_max_GeV;
    double energy_step_GeV;
    int    energy_logspace;   /* 1 = points log-uniformes, 0 = lineaire */
    int    energy_points;     /* nb de points si logspace=1 */

    double theta12_deg;
    double theta13_deg;
    double theta23_deg;
    double delta_cp_deg;

    int n_sterile;
    double theta_active_sterile_deg[3][MAX_STERILE_NEUTRINOS];
    double delta_active_sterile_deg[3][MAX_STERILE_NEUTRINOS];
    double theta_sterile_sterile_deg[MAX_STERILE_NEUTRINOS][MAX_STERILE_NEUTRINOS];
    double delta_sterile_sterile_deg[MAX_STERILE_NEUTRINOS][MAX_STERILE_NEUTRINOS];

    double dm21_eV2;
    double dm31_eV2;

    int gaussian_filter_enabled;
    double sigmaE_over_E;

    int matter_effects_enabled;
    double matter_density_g_cm3;
    double matter_electron_fraction;
    double matter_neutron_fraction;
    int matter_include_neutral_current_sterile;
    int matter_evolution_steps;
    double matter_a_cc_coeff_eV2_per_GeV_per_gcm3;

    SimulationOperation operation;

    double dm41_values_eV2[MAX_DM41_VALUES];
    int dm41_count;
    int dm41_range_enabled;
    double dm41_range_min_eV2;
    double dm41_range_max_eV2;
    double dm41_range_step_eV2;
    int dm41_range_logspace;
    int dm41_range_points;

    double dm54_values_eV2[MAX_DM54_VALUES];
    int dm54_count;
    int dm54_range_enabled;
    double dm54_range_min_eV2;
    double dm54_range_max_eV2;
    double dm54_range_step_eV2;
    int dm54_range_logspace;
    int dm54_range_points;
    double dm41_3p2_eV2;
    int dm41_3p2_is_set;
    double dm41_heatmap_3p2_values_eV2[16];
    int dm41_heatmap_3p2_count;

    char output_csv_path[256];
    char output_csv_3p2_path[256];
    char output_heatmap_csv_path[256];
    char output_heatmap_3p2_csv_path[256];
    char output_heatmap_pmumu_3p2_csv_path[256];
    char output_cp_heatmap_csv_path[256];
    char output_inverse_csv_path[256];
    char output_inverse_md_csv_path[256];
    char output_inverse_pmns_filter_csv_path[256];
    char output_inverse_construct_23_csv_path[256];
    char inverse_kept_points_dir[256];
    int inverse_clear_kept_points_dir;

    /* Inverse seesaw (3+1 léger) */
    double inverse_mD_3x2_GeV[3][2];
    double inverse_M_2x2_GeV[2][2];
    double inverse_mu_H_2x2_eV[2][2];
    double inverse_mu_H0_2x1_eV[2];
    double inverse_mu00_eV;
    double inverse_mu00_values_eV[MAX_INVERSE_MU_VALUES];
    int inverse_mu00_count;

    /* Parametrisation de Casas-Ibarra (mD 3x2 automatique) */
    double inverse_ci_m_light_eV[3];
    double inverse_ci_M_heavy_GeV[2];
    double inverse_ci_alpha21_deg;
    double inverse_ci_alpha31_deg;
    double inverse_ci_z_real;
    double inverse_ci_z_imag;
    int inverse_ci_normal_ordering;

    /* Cibles NuFIT 6.0 2024 pour les angles de melange actifs et delta_CP */
    double inverse_nufit_theta12_deg;
    double inverse_nufit_theta23_deg;
    double inverse_nufit_theta13_deg;
    double inverse_nufit_deltacp_deg;

    /* Contraintes experimentales sur |U_PMNS| (3x3 actif) */
    double inverse_pmns_abs_min_3x3[3][3];
    double inverse_pmns_abs_max_3x3[3][3];
    double inverse_eta_abs_max_3x3[3][3];
    double inverse_eta_abs_max_nonunitarity_3x3[3][3];
    double inverse_eta_abs_max_light_highdm_3x3[3][3];
    double inverse_eta_abs_max_light_lowdm_3x3[3][3];
    double inverse_eta_dm41_low_min_eV2;
    double inverse_eta_dm41_low_max_eV2;
    double inverse_eta_dm41_high_min_eV2;
    double inverse_br_muegamma_max;

    /* Tirage aleatoire pour le filtre PMNS inverse seesaw */
    int inverse_random_samples;
    int inverse_random_seed;
    double inverse_random_mu_min_eV;
    double inverse_random_mu_max_eV;
    double inverse_random_MR_min_eV;
    double inverse_random_MR_max_eV;
    double inverse_random_mD_min_eV;
    double inverse_random_mD_max_eV;

    /* Construction adaptee (nR,nSL)=(2,3): scan aleatoire des parametres libres */
    int inverse_construct_23_samples;
    int inverse_construct_23_seed;
    double inverse_construct_23_dm41_min_eV2;
    double inverse_construct_23_dm41_max_eV2;
    double inverse_construct_23_zeta_norm_min;
    double inverse_construct_23_zeta_norm_max;
    double inverse_construct_23_zeta_direction_min_deg;
    double inverse_construct_23_zeta_direction_max_deg;
    double inverse_construct_23_f11_min;
    double inverse_construct_23_f11_max;
    double inverse_construct_23_f12_min;
    double inverse_construct_23_f12_max;
    double inverse_construct_23_f21_min;
    double inverse_construct_23_f21_max;
    double inverse_construct_23_f22_min;
    double inverse_construct_23_f22_max;
    double inverse_construct_23_f_det_min_abs;
    double inverse_construct_23_M1_min_GeV;
    double inverse_construct_23_M1_max_GeV;
    double inverse_construct_23_M2_min_GeV;
    double inverse_construct_23_M2_max_GeV;

    /* Ranges de scan inverse seesaw (min/max/pas) */
    double inverse_scan_mu00_min_eV;
    double inverse_scan_mu00_max_eV;
    double inverse_scan_mu00_step_eV;

    double inverse_scan_M1_min_GeV;
    double inverse_scan_M1_max_GeV;
    double inverse_scan_M1_step_GeV;
    double inverse_scan_M2_min_GeV;
    double inverse_scan_M2_max_GeV;
    double inverse_scan_M2_step_GeV;

    double inverse_scan_z_real_min;
    double inverse_scan_z_real_max;
    double inverse_scan_z_real_step;
    double inverse_scan_z_imag_min;
    double inverse_scan_z_imag_max;
    double inverse_scan_z_imag_step;

    double inverse_scan_muH11_min_eV;
    double inverse_scan_muH11_max_eV;
    double inverse_scan_muH11_step_eV;
    double inverse_scan_muH22_min_eV;
    double inverse_scan_muH22_max_eV;
    double inverse_scan_muH22_step_eV;

    double inverse_scan_muH01_min_eV;
    double inverse_scan_muH01_max_eV;
    double inverse_scan_muH01_step_eV;
    double inverse_scan_muH02_min_eV;
    double inverse_scan_muH02_max_eV;
    double inverse_scan_muH02_step_eV;

    /* Scan delta_41 pour heatmap ACP */
    double delta41_values_deg[3600];
    int delta41_count;
    int delta41_range_enabled;
    double delta41_range_min_deg;
    double delta41_range_max_deg;
    int delta41_range_points;

    /* Baselines multiples pour heatmap ACP (et futurs scans multi-detecteurs) */
    double baseline_values_km[16];
    int baseline_count;

    double dist_min_km;
    double dist_max_km;
    double dist_step_km;
    double dist_fixed_energy_GeV;
    char output_dist_csv_path[256];
} SimulationConfig;

int load_config(const char *file_path, SimulationConfig *cfg);
void print_config_summary(const SimulationConfig *cfg);
const char *operation_to_string(SimulationOperation operation);

#endif
