#include "scan.h"

#include "constants.h"
#include "io_csv.h"
#include "oscillation.h"
#include "pmns.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>

int run_scan_heatmap_delta_pmue(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 1 || cfg->dm41_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_csv_path[0] == '\0') {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u4[N_FLAVORS_3P1][N_FLAVORS_3P1];
    if (pmns_build_3plus1(cfg, u4) != 0) {
        return 3;
    }

    double complex u3[3][3];
    pmns_build_3x3(
        deg_to_rad(cfg->theta12_deg),
        deg_to_rad(cfg->theta13_deg),
        deg_to_rad(cfg->theta23_deg),
        deg_to_rad(cfg->delta_cp_deg),
        u3);

    FILE *out = open_delta_pmue_heatmap_csv(cfg->output_heatmap_csv_path);
    if (!out) {
        return 2;
    }

    const double mass_sq_3nu[3] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2};

    for (int idx = 0; idx < cfg->dm41_count; ++idx) {
        const double dm41 = cfg->dm41_values_eV2[idx];
        const double mass_sq_3p1[N_FLAVORS_3P1] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41};

        for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
            const double p_mue_3pns = probability_with_config_n(
                N_FLAVORS_3P1,
                FLAVOR_MU,
                FLAVOR_E,
                energy,
                cfg->baseline_km,
                mass_sq_3p1,
                u4,
                cfg,
                0);

            const double p_mue_3nu = probability_with_config_n(
                3,
                FLAVOR_MU,
                FLAVOR_E,
                energy,
                cfg->baseline_km,
                mass_sq_3nu,
                u3,
                cfg,
                0);

            const double delta_p_mue = p_mue_3pns - p_mue_3nu;
            write_delta_pmue_heatmap_row(out, energy, dm41, p_mue_3pns, p_mue_3nu, delta_p_mue);
        }
    }

    fclose(out);
    printf("CSV genere (heatmap Delta Pmue): %s\n", cfg->output_heatmap_csv_path);
    return 0;
}

int run_scan_heatmap_delta_pmue_3p2(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 2 || cfg->dm41_heatmap_3p2_count != 4 || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_3p2_csv_path[0] == '\0') {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u5[N_FLAVORS_3P2][N_FLAVORS_3P2];
    if (pmns_build_3plus2(cfg, u5) != 0) {
        return 3;
    }

    double complex u3[3][3];
    pmns_build_3x3(
        deg_to_rad(cfg->theta12_deg),
        deg_to_rad(cfg->theta13_deg),
        deg_to_rad(cfg->theta23_deg),
        deg_to_rad(cfg->delta_cp_deg),
        u3);

    FILE *out = open_delta_pmue_heatmap_3p2_csv(cfg->output_heatmap_3p2_csv_path);
    if (!out) {
        return 2;
    }

    const double mass_sq_3nu[3] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2};

    for (int idx41 = 0; idx41 < cfg->dm41_heatmap_3p2_count; ++idx41) {
        const double dm41 = cfg->dm41_heatmap_3p2_values_eV2[idx41];

        for (int idx54 = 0; idx54 < cfg->dm54_count; ++idx54) {
            const double dm54 = cfg->dm54_values_eV2[idx54];
            const double dm51 = dm41 + dm54;
            const double mass_sq_3p2[N_FLAVORS_3P2] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41, dm51};

            for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
                const double p_mue_3p2 = probability_with_config_n(
                    N_FLAVORS_3P2,
                    FLAVOR_MU,
                    FLAVOR_E,
                    energy,
                    cfg->baseline_km,
                    mass_sq_3p2,
                    u5,
                    cfg,
                    0);

                const double p_mue_3nu = probability_with_config_n(
                    3,
                    FLAVOR_MU,
                    FLAVOR_E,
                    energy,
                    cfg->baseline_km,
                    mass_sq_3nu,
                    u3,
                    cfg,
                    0);

                const double delta_p_mue = p_mue_3p2 - p_mue_3nu;
                write_delta_pmue_heatmap_3p2_row(out, energy, dm41, dm54, p_mue_3p2, p_mue_3nu, delta_p_mue);
            }
        }
    }

    fclose(out);
    printf("CSV genere (heatmap Delta Pmue 3+2): %s\n", cfg->output_heatmap_3p2_csv_path);
    return 0;
}

int run_scan_heatmap_delta_pmumu_3p2(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 2 || cfg->dm41_heatmap_3p2_count != 4 || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_pmumu_3p2_csv_path[0] == '\0') {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u5[N_FLAVORS_3P2][N_FLAVORS_3P2];
    if (pmns_build_3plus2(cfg, u5) != 0) {
        return 3;
    }

    double complex u3[3][3];
    pmns_build_3x3(
        deg_to_rad(cfg->theta12_deg),
        deg_to_rad(cfg->theta13_deg),
        deg_to_rad(cfg->theta23_deg),
        deg_to_rad(cfg->delta_cp_deg),
        u3);

    FILE *out = open_delta_pmumu_heatmap_3p2_csv(cfg->output_heatmap_pmumu_3p2_csv_path);
    if (!out) {
        return 2;
    }

    const double mass_sq_3nu[3] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2};

    for (int idx41 = 0; idx41 < cfg->dm41_heatmap_3p2_count; ++idx41) {
        const double dm41 = cfg->dm41_heatmap_3p2_values_eV2[idx41];

        for (int idx54 = 0; idx54 < cfg->dm54_count; ++idx54) {
            const double dm54 = cfg->dm54_values_eV2[idx54];
            const double dm51 = dm41 + dm54;
            const double mass_sq_3p2[N_FLAVORS_3P2] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41, dm51};

            for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
                const double p_mumu_3p2 = probability_with_config_n(
                    N_FLAVORS_3P2,
                    FLAVOR_MU,
                    FLAVOR_MU,
                    energy,
                    cfg->baseline_km,
                    mass_sq_3p2,
                    u5,
                    cfg,
                    0);

                const double p_mumu_3nu = probability_with_config_n(
                    3,
                    FLAVOR_MU,
                    FLAVOR_MU,
                    energy,
                    cfg->baseline_km,
                    mass_sq_3nu,
                    u3,
                    cfg,
                    0);

                const double delta_p_mumu = p_mumu_3p2 - p_mumu_3nu;
                write_delta_pmumu_heatmap_3p2_row(out, energy, dm41, dm54, p_mumu_3p2, p_mumu_3nu, delta_p_mumu);
            }
        }
    }

    fclose(out);
    printf("CSV genere (heatmap Delta Pmumu 3+2): %s\n", cfg->output_heatmap_pmumu_3p2_csv_path);
    return 0;
}

int run_scan_cp_heatmap_3p1(const SimulationConfig *cfg) {
    const int use_log = cfg->energy_logspace && cfg->energy_points >= 2;
    if (!cfg || cfg->n_sterile < 1 || cfg->dm41_count <= 0 || cfg->delta41_count <= 0 ||
        cfg->output_cp_heatmap_csv_path[0] == '\0') {
        return 1;
    }
    if (!use_log && cfg->energy_step_GeV <= 0.0) {
        return 1;
    }

    int n_energy = 0;
    double energy_arr[16384];
    if (use_log) {
        const int np = (cfg->energy_points < 16384) ? cfg->energy_points : 16384;
        const double log_min = log10(cfg->energy_min_GeV);
        const double log_max = log10(cfg->energy_max_GeV);
        for (int k = 0; k < np; ++k) {
            energy_arr[k] = pow(10.0, log_min + k * (log_max - log_min) / (np - 1));
        }
        n_energy = np;
    } else {
        for (double e = cfg->energy_min_GeV; e <= cfg->energy_max_GeV + 1e-12; e += cfg->energy_step_GeV) {
            if (n_energy >= 16384) {
                break;
            }
            energy_arr[n_energy++] = e;
        }
    }

    int n_baselines = cfg->baseline_count;
    double baselines[16];
    if (n_baselines > 0) {
        for (int i = 0; i < n_baselines && i < 16; ++i) {
            baselines[i] = cfg->baseline_values_km[i];
        }
    } else {
        baselines[0] = cfg->baseline_km;
        n_baselines = 1;
    }

    ensure_directory_exists("data");

    FILE *out = open_cp_heatmap_3p1_csv(cfg->output_cp_heatmap_csv_path);
    if (!out) {
        return 2;
    }

    for (int idx_base = 0; idx_base < n_baselines; ++idx_base) {
        const double baseline_km = baselines[idx_base];

        for (int idx_dm41 = 0; idx_dm41 < cfg->dm41_count; ++idx_dm41) {
            const double dm41 = cfg->dm41_values_eV2[idx_dm41];
            const double mass_sq[N_FLAVORS_3P1] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41};

            for (int idx_d41 = 0; idx_d41 < cfg->delta41_count; ++idx_d41) {
                const double delta41_deg = cfg->delta41_values_deg[idx_d41];

                SimulationConfig local_cfg = *cfg;
                local_cfg.delta_active_sterile_deg[0][0] = delta41_deg;

                double complex u_nu[N_FLAVORS_3P1][N_FLAVORS_3P1];
                if (pmns_build_3plus1(&local_cfg, u_nu) != 0) {
                    fclose(out);
                    return 3;
                }

                for (int ke = 0; ke < n_energy; ++ke) {
                    const double energy = energy_arr[ke];
                    const double p_nu = probability_with_config_n(
                        N_FLAVORS_3P1,
                        FLAVOR_MU,
                        FLAVOR_E,
                        energy,
                        baseline_km,
                        mass_sq,
                        u_nu,
                        cfg,
                        0);

                    const double p_anti = probability_with_config_n(
                        N_FLAVORS_3P1,
                        FLAVOR_MU,
                        FLAVOR_E,
                        energy,
                        baseline_km,
                        mass_sq,
                        u_nu,
                        cfg,
                        1);

                    const double acp = p_nu - p_anti;
                    write_cp_heatmap_3p1_row(out, baseline_km, energy, delta41_deg, dm41, p_nu, p_anti, acp);
                }
            }
        }
    }

    fclose(out);
    printf("CSV genere (heatmap ACP 3+1): %s\n", cfg->output_cp_heatmap_csv_path);
    return 0;
}
