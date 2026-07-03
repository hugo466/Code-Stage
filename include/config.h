#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

#define MAX_DM41_VALUES 5000
#define MAX_DM54_VALUES 5000
#define MAX_STERILE_NEUTRINOS 8
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
    OPERATION_INVERSE_PMNS_FILTER_3P1,
    OPERATION_INVERSE_PMNS_FILTER_3P2,
    OPERATION_INVERSE_CONSTRUCT_23_3P1,
    OPERATION_INVERSE_CONSTRUCT_24_3P2,
    OPERATION_DUNE_ND_PREDICT_SPECTRUM,
    OPERATION_DUNE_FD_FIG4_VALIDATION,
    OPERATION_DUNE_ND_FIG4_SOURCE_LINE,
    OPERATION_DUNE_BASELINE_EFFECTS_SENSITIVITY
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
    char output_inverse_pmns_filter_csv_path[256];
    char output_inverse_construct_23_csv_path[256];
    char output_inverse_construct_24_csv_path[256];
    char inverse_kept_points_dir[256];
    int inverse_clear_kept_points_dir;

    char dune_theory_index_csv[256];
    char dune_theory_model[32];
    char dune_point_source[32];
    int dune_point_id;
    char dune_beam_mode[16];
    char dune_flux_model[32];
    char dune_flux_format[32];
    char dune_flux_fhc_file[256];
    char dune_flux_rhc_file[256];
    char dune_baseline_model[32];
    char dune_source_model[32];
    char dune_dk2nu_flux_z_fhc_file[256];
    char dune_dk2nu_flux_z_rhc_file[256];
    int dune_source_debug;
    double dune_detector_distance_m;
    double dune_source_z_start_m;
    double dune_decay_pipe_length_m;
    int dune_source_z_bins;
    char dune_osc_engine[32];
    int dune_matter_enabled;
    char dune_xsec_model[32];
    char dune_xsec_format[32];
    char dune_xsec_cc_file[256];
    char dune_xsec_nc_file[256];
    char dune_detectors[64];
    char dune_ndlar_response_model[32];
    char dune_ndlar_migration_model[32];
    char dune_ndlar_category_model[32];
    char dune_samples_enabled[64];
    char dune_samples_axis[32];
    double dune_Erec_min_GeV;
    double dune_Erec_max_GeV;
    int dune_Erec_bins;
    char dune_spectrum_pred_csv[256];
    char dune_spectrum_null_csv[256];
    char dune_residuals_csv[256];
    char dune_point_observables_csv[256];

    char sensitivity_detector_mode[32];
    char sensitivity_asimov_mode[32];
    char sensitivity_test_backend[32];
    char sensitivity_points_index_csv[256];
    char sensitivity_output_csv[256];
    char sensitivity_source_model[32];
    char sensitivity_scan_plane[32];
    char sensitivity_nd_dk2nu_flux_z_fhc_file[256];
    char sensitivity_nd_dk2nu_flux_z_rhc_file[256];
    char sensitivity_fd_dk2nu_flux_z_fhc_file[256];
    char sensitivity_fd_dk2nu_flux_z_rhc_file[256];
    int sensitivity_max_points;
    int sensitivity_point_offset;
    int sensitivity_shape_systematics_enabled;
    int sensitivity_priors_enabled;
    int sensitivity_minimizer_max_iter;
    double sensitivity_minimizer_tolerance;
    double sensitivity_poisson_epsilon;
    double sensitivity_dm41_min_eV2;
    double sensitivity_dm41_max_eV2;
    int sensitivity_dm41_points;
    int sensitivity_dm41_logspace;
    double sensitivity_theta14_min_deg;
    double sensitivity_theta14_max_deg;
    int sensitivity_theta14_points;
    int sensitivity_theta14_logspace;
    double sensitivity_sin2_theta14_min;
    double sensitivity_sin2_theta14_max;
    double sensitivity_theta24_min_deg;
    double sensitivity_theta24_max_deg;
    int sensitivity_theta24_points;
    int sensitivity_theta24_logspace;
    double sensitivity_sin2_theta24_min;
    double sensitivity_sin2_theta24_max;
    double sensitivity_theta34_deg;
    double sensitivity_delta24_deg;
    double sensitivity_delta34_deg;

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
    int inverse_construct_23_dm41_logspace;
    double inverse_construct_23_zeta_norm_min;
    double inverse_construct_23_zeta_norm_max;
    double inverse_construct_23_zeta_direction_min_deg;
    double inverse_construct_23_zeta_direction_max_deg;
    double inverse_construct_23_zeta_phase_min_deg;
    double inverse_construct_23_zeta_phase_max_deg;
    double inverse_construct_23_alpha21_min_deg;
    double inverse_construct_23_alpha21_max_deg;
    double inverse_construct_23_alpha31_min_deg;
    double inverse_construct_23_alpha31_max_deg;
    double inverse_construct_23_f11_min;
    double inverse_construct_23_f11_max;
    double inverse_construct_23_f11_phase_min_deg;
    double inverse_construct_23_f11_phase_max_deg;
    double inverse_construct_23_f12_min;
    double inverse_construct_23_f12_max;
    double inverse_construct_23_f12_phase_min_deg;
    double inverse_construct_23_f12_phase_max_deg;
    double inverse_construct_23_f21_min;
    double inverse_construct_23_f21_max;
    double inverse_construct_23_f21_phase_min_deg;
    double inverse_construct_23_f21_phase_max_deg;
    double inverse_construct_23_f22_min;
    double inverse_construct_23_f22_max;
    double inverse_construct_23_f22_phase_min_deg;
    double inverse_construct_23_f22_phase_max_deg;
    double inverse_construct_23_f_det_min_abs;
    double inverse_construct_23_f_det_max_abs;
    double inverse_construct_23_f_sigma_min_min;
    double inverse_construct_23_kappa_f_max;
    double inverse_construct_23_M1_min_GeV;
    double inverse_construct_23_M1_max_GeV;
    double inverse_construct_23_M2_min_GeV;
    double inverse_construct_23_M2_max_GeV;

    /* Construction adaptee (nR,nSL)=(2,4): secteur leger 3+2 avec C=V diag(s1,s2) W */
    int inverse_construct_24_samples;
    int inverse_construct_24_seed;
    double inverse_construct_24_dm41_min_eV2;
    double inverse_construct_24_dm41_max_eV2;
    int inverse_construct_24_dm41_logspace;
    double inverse_construct_24_dm51_min_eV2;
    double inverse_construct_24_dm51_max_eV2;
    int inverse_construct_24_dm51_logspace;
    double inverse_construct_24_s1_min;
    double inverse_construct_24_s1_max;
    double inverse_construct_24_s2_min;
    double inverse_construct_24_s2_max;
    double inverse_construct_24_v_angle_min_deg;
    double inverse_construct_24_v_angle_max_deg;
    double inverse_construct_24_w_angle_min_deg;
    double inverse_construct_24_w_angle_max_deg;
    double inverse_construct_24_phase_min_deg;
    double inverse_construct_24_phase_max_deg;
    double inverse_construct_24_alpha21_min_deg;
    double inverse_construct_24_alpha21_max_deg;
    double inverse_construct_24_alpha31_min_deg;
    double inverse_construct_24_alpha31_max_deg;

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
