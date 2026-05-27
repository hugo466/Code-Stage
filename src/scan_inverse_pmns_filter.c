#include "scan.h"

#include "inverse_seesaw.h"
#include "utils.h"

#include <math.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static double asin_deg_from_abs(double value) {
    if (value < 0.0) {
        value = 0.0;
    }
    if (value > 1.0) {
        value = 1.0;
    }
    return asin(value) * 180.0 / M_PI;
}

static double uniform_random(double min_value, double max_value) {
    const double t = (double)rand() / (double)RAND_MAX;
    return min_value + (max_value - min_value) * t;
}

static double loop_function_g_gamma(double x) {
    if (x <= 0.0) {
        return 0.0;
    }

    if (fabs(x - 1.0) < 1e-8) {
        return 0.125;
    }

    return -x * (2.0 * x * x + 5.0 * x - 1.0) / (4.0 * pow(1.0 - x, 3.0))
           - 3.0 * x * x * x * log(x) / (2.0 * pow(1.0 - x, 4.0));
}

static double branching_ratio_mu_to_e_gamma(const InverseSeesaw3p1Result *result) {
    const double alpha_em = 1.0 / 137.035999084;
    const double m_w_eV = 80.379e9;
    double amplitude = 0.0;

    for (int i = 0; i < 8; ++i) {
        const double x = (result->masses_full_eV[i] * result->masses_full_eV[i]) / (m_w_eV * m_w_eV);
        const double g = loop_function_g_gamma(x);
        amplitude += result->mixing_8x8[1][i] * result->mixing_8x8[0][i] * g;
    }

    return (3.0 * alpha_em / (32.0 * M_PI)) * amplitude * amplitude;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

static void clear_txt_files_in_dir(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        return;
    }

    struct dirent *entry;
    char path[512];

    while ((entry = readdir(dir)) != NULL) {
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len < 4) {
            continue;
        }
        if (strcmp(name + len - 4, ".txt") != 0) {
            continue;
        }

        snprintf(path, sizeof(path), "%s/%s", dir_path, name);
        remove(path);
    }

    closedir(dir);
}

static int find_next_kept_point_index_in_dir(const char *dir_path) {
    int idx = 1;
    char path[512];

    while (1) {
        snprintf(path, sizeof(path), "%s/%d.txt", dir_path, idx);
        if (!file_exists(path)) {
            return idx;
        }
        ++idx;
    }
}

static void write_kept_point_details(int point_id,
                                     const char *kept_points_dir,
                                     const InverseSeesaw3p1Input *input,
                                     const InverseSeesaw3p1Result *result,
                                     double dm32_eV2,
                                     const double abs_u3[3][3],
                                     const double eta_abs_3x3[3][3],
                                     double br_muegamma,
                                     int eta_pass) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%d.txt", kept_points_dir, point_id);

    FILE *f = fopen(path, "w");
    if (!f) {
        return;
    }
        fprintf(f, "=== METADATA ===\n");
        fprintf(f, "point_id = %d\n", point_id);
        fprintf(f, "eta_pass = %d\n\n", eta_pass ? 1 : 0);

        fprintf(f, "=== INPUT PARAMETERS ===\n");
        fprintf(f, "mu00_eV = %.10e\n", input->mu00_eV);
        fprintf(f, "M_2x2_GeV = [%.10e, %.10e; %.10e, %.10e]\n",
            input->M_2x2_GeV[0][0], input->M_2x2_GeV[0][1],
            input->M_2x2_GeV[1][0], input->M_2x2_GeV[1][1]);
        fprintf(f, "mD_3x2_GeV = [%.10e, %.10e; %.10e, %.10e; %.10e, %.10e]\n",
            input->mD_3x2_GeV[0][0], input->mD_3x2_GeV[0][1],
            input->mD_3x2_GeV[1][0], input->mD_3x2_GeV[1][1],
            input->mD_3x2_GeV[2][0], input->mD_3x2_GeV[2][1]);
        fprintf(f, "mu_H_2x2_eV = [%.10e, %.10e; %.10e, %.10e]\n",
            input->mu_H_2x2_eV[0][0], input->mu_H_2x2_eV[0][1],
            input->mu_H_2x2_eV[1][0], input->mu_H_2x2_eV[1][1]);
        fprintf(f, "mu_H0_2x1_eV = [%.10e, %.10e]\n\n",
            input->mu_H0_2x1_eV[0], input->mu_H0_2x1_eV[1]);

        fprintf(f, "=== FILTER OBSERVABLES (EFFECTIVE 4x4) ===\n");
        fprintf(f, "masses_eV = [%.10e, %.10e, %.10e, %.10e]\n",
            result->masses_eV[0], result->masses_eV[1], result->masses_eV[2], result->masses_eV[3]);
        fprintf(f, "dm21_eV2 = %.10e\n", result->dm21_eV2);
        fprintf(f, "dm31_eV2 = %.10e\n", result->dm31_eV2);
        fprintf(f, "dm32_eV2 = %.10e\n", dm32_eV2);
        fprintf(f, "dm41_eV2 = %.10e\n", result->dm41_eV2);
        fprintf(f, "theta14_deg = %.10e\n", result->theta14_deg);
        fprintf(f, "theta24_deg = %.10e\n", result->theta24_deg);
        fprintf(f, "theta34_deg = %.10e\n", result->theta34_deg);
        fprintf(f, "sterile_state_index = %d\n", result->sterile_state_index);
        fprintf(f, "PMNS_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n",
            abs_u3[0][0], abs_u3[0][1], abs_u3[0][2],
            abs_u3[1][0], abs_u3[1][1], abs_u3[1][2],
            abs_u3[2][0], abs_u3[2][1], abs_u3[2][2]);
        fprintf(f, "eta_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n\n",
            eta_abs_3x3[0][0], eta_abs_3x3[0][1], eta_abs_3x3[0][2],
            eta_abs_3x3[1][0], eta_abs_3x3[1][1], eta_abs_3x3[1][2],
            eta_abs_3x3[2][0], eta_abs_3x3[2][1], eta_abs_3x3[2][2]);
        fprintf(f, "br_muegamma = %.10e\n\n", br_muegamma);

        fprintf(f, "=== FULL 8x8 SUMMARY ===\n");
        fprintf(f, "light_state_indices = [%d, %d, %d, %d]\n",
            result->light_state_indices[0], result->light_state_indices[1],
            result->light_state_indices[2], result->light_state_indices[3]);
        fprintf(f, "heavy_state_indices = [%d, %d, %d, %d]\n",
            result->heavy_state_indices[0], result->heavy_state_indices[1],
            result->heavy_state_indices[2], result->heavy_state_indices[3]);
        fprintf(f, "light_masses_eV = [%.10e, %.10e, %.10e, %.10e]\n",
            result->light_masses_eV[0], result->light_masses_eV[1], result->light_masses_eV[2], result->light_masses_eV[3]);
        fprintf(f, "heavy_masses_eV = [%.10e, %.10e, %.10e, %.10e]\n",
            result->heavy_masses_eV[0], result->heavy_masses_eV[1], result->heavy_masses_eV[2], result->heavy_masses_eV[3]);
        fprintf(f, "masses_full_eV = [%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]\n",
            result->masses_full_eV[0], result->masses_full_eV[1], result->masses_full_eV[2], result->masses_full_eV[3],
            result->masses_full_eV[4], result->masses_full_eV[5], result->masses_full_eV[6], result->masses_full_eV[7]);

        {
        const double m1 = result->masses_full_eV[0];
        fprintf(f,
            "dm_full_ref_m1_eV2 = [%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]\n\n",
            result->masses_full_eV[1] * result->masses_full_eV[1] - m1 * m1,
            result->masses_full_eV[2] * result->masses_full_eV[2] - m1 * m1,
            result->masses_full_eV[3] * result->masses_full_eV[3] - m1 * m1,
            result->masses_full_eV[4] * result->masses_full_eV[4] - m1 * m1,
            result->masses_full_eV[5] * result->masses_full_eV[5] - m1 * m1,
            result->masses_full_eV[6] * result->masses_full_eV[6] - m1 * m1,
            result->masses_full_eV[7] * result->masses_full_eV[7] - m1 * m1);
        }

        fprintf(f, "=== ACTIVE-HEAVY MIXING ===\n");
        fprintf(f,
            "theta_active_heavy_deg = [%.10e, %.10e, %.10e, %.10e; %.10e, %.10e, %.10e, %.10e; %.10e, %.10e, %.10e, %.10e]\n",
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[0][0]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[0][1]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[0][2]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[0][3]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[1][0]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[1][1]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[1][2]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[1][3]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[2][0]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[2][1]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[2][2]),
            asin_deg_from_abs(result->active_heavy_mixing_abs_3x4[2][3]));
        fprintf(f,
            "active_heavy_mixing_abs_3x4 = [%.10e, %.10e, %.10e, %.10e; %.10e, %.10e, %.10e, %.10e; %.10e, %.10e, %.10e, %.10e]\n\n",
            result->active_heavy_mixing_abs_3x4[0][0], result->active_heavy_mixing_abs_3x4[0][1],
            result->active_heavy_mixing_abs_3x4[0][2], result->active_heavy_mixing_abs_3x4[0][3],
            result->active_heavy_mixing_abs_3x4[1][0], result->active_heavy_mixing_abs_3x4[1][1],
            result->active_heavy_mixing_abs_3x4[1][2], result->active_heavy_mixing_abs_3x4[1][3],
            result->active_heavy_mixing_abs_3x4[2][0], result->active_heavy_mixing_abs_3x4[2][1],
            result->active_heavy_mixing_abs_3x4[2][2], result->active_heavy_mixing_abs_3x4[2][3]);

        fprintf(f, "=== MATRICES ===\n");
        fprintf(f, "mixing_8x8 =\n");
    for (int row = 0; row < 8; ++row) {
        fprintf(f,
                "  [%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]\n",
                result->mixing_8x8[row][0], result->mixing_8x8[row][1], result->mixing_8x8[row][2], result->mixing_8x8[row][3],
                result->mixing_8x8[row][4], result->mixing_8x8[row][5], result->mixing_8x8[row][6], result->mixing_8x8[row][7]);
    }
        fprintf(f, "\n");
    fprintf(f, "m_full_8x8_eV =\n");
    for (int row = 0; row < 8; ++row) {
        fprintf(f,
                "  [%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]\n",
                result->m_full_8x8_eV[row][0], result->m_full_8x8_eV[row][1], result->m_full_8x8_eV[row][2], result->m_full_8x8_eV[row][3],
                result->m_full_8x8_eV[row][4], result->m_full_8x8_eV[row][5], result->m_full_8x8_eV[row][6], result->m_full_8x8_eV[row][7]);
    }
        fprintf(f, "\n");
    fprintf(f, "m_light_4x4_eV =\n");
    for (int row = 0; row < 4; ++row) {
        fprintf(f,
                "  [%.10e, %.10e, %.10e, %.10e]\n",
                result->m_light_4x4_eV[row][0], result->m_light_4x4_eV[row][1], result->m_light_4x4_eV[row][2], result->m_light_4x4_eV[row][3]);
    }

    fclose(f);
}

static int eta_constraints_satisfied(const InverseSeesaw3p1Result *result,
                                     const SimulationConfig *cfg,
                                     double eta_abs_3x3[3][3]) {
    const double dm41_eV2 = result->dm41_eV2;
    const double (*eta4_max_3x3)[3] = cfg->inverse_eta_abs_max_nonunitarity_3x3;
    const double (*etaH_max_3x3)[3] = cfg->inverse_eta_abs_max_nonunitarity_3x3;
    const double ew_scale_eV = 174.0e9;
    int heavy_above_ew = 1;

    if (dm41_eV2 >= cfg->inverse_eta_dm41_low_min_eV2 && dm41_eV2 <= cfg->inverse_eta_dm41_low_max_eV2) {
        eta4_max_3x3 = cfg->inverse_eta_abs_max_light_lowdm_3x3;
    } else if (dm41_eV2 >= cfg->inverse_eta_dm41_high_min_eV2) {
        eta4_max_3x3 = cfg->inverse_eta_abs_max_light_highdm_3x3;
    }

    for (int i = 0; i < 4; ++i) {
        if (result->heavy_masses_eV[i] < ew_scale_eV) {
            heavy_above_ew = 0;
            break;
        }
    }

    const int sterile_idx = result->sterile_state_index;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double eta4_sum = 0.0;
            double etaH_sum = 0.0;

            eta4_sum = result->mixing_8x8[a][sterile_idx] * result->mixing_8x8[b][sterile_idx];

            for (int heavy_col = 4; heavy_col < 8; ++heavy_col) {
                etaH_sum += result->mixing_8x8[a][heavy_col] * result->mixing_8x8[b][heavy_col];
            }

            const double eta4 = 0.5 * eta4_sum;
            const double etaH = 0.5 * etaH_sum;
            const double eta_total = eta4 + etaH;

            const double eta4_abs = fabs(eta4);
            const double etaH_abs = fabs(etaH);
            eta_abs_3x3[a][b] = fabs(eta_total);

            if (eta4_abs > eta4_max_3x3[a][b]) {
                return 0;
            }

            if (heavy_above_ew && etaH_abs > etaH_max_3x3[a][b]) {
                return 0;
            }
        }
    }

    return 1;
}

static void print_progress_bar(long long done, long long total, long long kept, long long eta_kept) {
    const int bar_width = 40;
    const int filled = (total > 0) ? (int)((done * bar_width) / total) : 0;
    const int percent = (total > 0) ? (int)((done * 100) / total) : 100;

    printf("\r\033[K[");
    for (int i = 0; i < bar_width; ++i) {
        printf(i < filled ? "=" : " ");
    }
    printf("] %3d%%  tested=%lld/%lld  kept=%lld  eta_pass=%lld",
           percent, done, total, kept, eta_kept);
    fflush(stdout);
}

static void build_ordered_mass_indices_3p1(const InverseSeesaw3p1Result *result,
                                           int ordered_mass_index[4]) {
    const int sterile_idx = result->sterile_state_index;
    int active_indices[3];
    int active_pos = 0;

    for (int i = 0; i < 4; ++i) {
        if (i != sterile_idx) {
            active_indices[active_pos++] = i;
        }
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (result->masses_eV[active_indices[j]] < result->masses_eV[active_indices[i]]) {
                const int tmp = active_indices[i];
                active_indices[i] = active_indices[j];
                active_indices[j] = tmp;
            }
        }
    }

    ordered_mass_index[0] = active_indices[0];
    ordered_mass_index[1] = active_indices[1];
    ordered_mass_index[2] = active_indices[2];
    ordered_mass_index[3] = sterile_idx;
}

static int pmns_constraints_satisfied(const InverseSeesaw3p1Result *result,
                                      const SimulationConfig *cfg,
                                      double abs_u3[3][3]) {
    int ordered_mass_index[4] = {-1, -1, -1, -1};
    build_ordered_mass_indices_3p1(result, ordered_mass_index);

    for (int flavor = 0; flavor < 3; ++flavor) {
        for (int mass = 0; mass < 3; ++mass) {
            const int source_mass = ordered_mass_index[mass];
            const double value = fabs(result->mixing_4x4[flavor][source_mass]);
            abs_u3[flavor][mass] = value;
            if (value < cfg->inverse_pmns_abs_min_3x3[flavor][mass] ||
                value > cfg->inverse_pmns_abs_max_3x3[flavor][mass]) {
                return 0;
            }
        }
    }

    return 1;
}

int run_scan_inverse_pmns_filter_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->output_inverse_pmns_filter_csv_path[0] == '\0') {
        return 1;
    }

    const clock_t t0 = clock();

    if (cfg->inverse_random_seed > 0) {
        srand((unsigned int)cfg->inverse_random_seed);
    } else {
        srand((unsigned int)time(NULL));
    }

    ensure_directory_exists("data");
    ensure_directory_exists(cfg->inverse_kept_points_dir);

    if (cfg->inverse_clear_kept_points_dir) {
        clear_txt_files_in_dir(cfg->inverse_kept_points_dir);
    }

    FILE *out = fopen(cfg->output_inverse_pmns_filter_csv_path, "w");
    if (!out) {
        return 2;
    }

    fprintf(out,
            "point_id,sample_id,mu00_eV,"
            "M11_GeV,M12_GeV,M21_GeV,M22_GeV,"
            "mD11_GeV,mD12_GeV,mD21_GeV,mD22_GeV,mD31_GeV,mD32_GeV,"
            "muH11_eV,muH12_eV,muH21_eV,muH22_eV,muH01_eV,muH02_eV,"
            "m1_eV,m2_eV,m3_eV,m4_eV,dm21_eV2,dm31_eV2,dm32_eV2,dm41_eV2,"
            "Ue1_abs,Ue2_abs,Ue3_abs,Umu1_abs,Umu2_abs,Umu3_abs,Utau1_abs,Utau2_abs,Utau3_abs,"
            "m5_eV,m6_eV,m7_eV,m8_eV,dm51_eV2,dm61_eV2,dm71_eV2,dm81_eV2,"
            "Ue5_abs,Ue6_abs,Ue7_abs,Ue8_abs,br_muegamma,eta_pass\n");

    long long tested_count = 0;
    long long kept_count = 0;
    long long eta_kept_count = 0;
    int next_point_id = cfg->inverse_clear_kept_points_dir
                            ? 1
                            : find_next_kept_point_index_in_dir(cfg->inverse_kept_points_dir);
    int last_percent_printed = -1;

    const double mr_min_gev = cfg->inverse_random_MR_min_eV * 1e-9;
    const double mr_max_gev = cfg->inverse_random_MR_max_eV * 1e-9;
    const double md_min_gev = cfg->inverse_random_mD_min_eV * 1e-9;
    const double md_max_gev = cfg->inverse_random_mD_max_eV * 1e-9;

    for (int sample = 0; sample < cfg->inverse_random_samples; ++sample) {
        InverseSeesaw3p1Input input;
        InverseSeesaw3p1Result result;
        ++tested_count;

        input.mu00_eV = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                input.M_2x2_GeV[i][j] = uniform_random(mr_min_gev, mr_max_gev);
            }
            input.mu_H0_2x1_eV[i] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        }
        /* mu_H doit etre symetrique: tirer seulement j >= i */
        input.mu_H_2x2_eV[0][0] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        input.mu_H_2x2_eV[1][1] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        input.mu_H_2x2_eV[0][1] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        input.mu_H_2x2_eV[1][0] = input.mu_H_2x2_eV[0][1];

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 2; ++col) {
                input.mD_3x2_GeV[row][col] = uniform_random(md_min_gev, md_max_gev);
            }
        }

        const int solve_ret = inverse_seesaw_solve_3p1(&input, &result);
        if (solve_ret != 0) {
            const int done = sample + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (done * 100) / cfg->inverse_random_samples
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar(done, cfg->inverse_random_samples, kept_count, eta_kept_count);
                last_percent_printed = percent;
            }
            continue;
        }

        double abs_u3[3][3];
        if (!pmns_constraints_satisfied(&result, cfg, abs_u3)) {
            const int done = sample + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (done * 100) / cfg->inverse_random_samples
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar(done, cfg->inverse_random_samples, kept_count, eta_kept_count);
                last_percent_printed = percent;
            }
            continue;
        }

        int ordered_mass_index[4] = {-1, -1, -1, -1};
        double active_mass_sq[3] = {0.0, 0.0, 0.0};
        build_ordered_mass_indices_3p1(&result, ordered_mass_index);

        for (int i = 0; i < 3; ++i) {
            const int idx = ordered_mass_index[i];
            active_mass_sq[i] = result.masses_eV[idx] * result.masses_eV[idx];
        }

        const double dm32_eV2 = active_mass_sq[2] - active_mass_sq[1];
        double eta_abs_3x3[3][3] = {{0.0}};
        const int eta_pass = eta_constraints_satisfied(&result, cfg, eta_abs_3x3);
        if (eta_pass) {
            ++eta_kept_count;
        }

        const double br_muegamma = branching_ratio_mu_to_e_gamma(&result);

        const int point_id = next_point_id++;
        write_kept_point_details(point_id, cfg->inverse_kept_points_dir, &input, &result, dm32_eV2, abs_u3, eta_abs_3x3, br_muegamma, eta_pass);

        ++kept_count;
        {
            const double m1 = result.masses_full_eV[0];
            const double dm51 = result.masses_full_eV[4] * result.masses_full_eV[4] - m1 * m1;
            const double dm61 = result.masses_full_eV[5] * result.masses_full_eV[5] - m1 * m1;
            const double dm71 = result.masses_full_eV[6] * result.masses_full_eV[6] - m1 * m1;
            const double dm81 = result.masses_full_eV[7] * result.masses_full_eV[7] - m1 * m1;

        fprintf(out,
            "%d,%d,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e\n",
            point_id,
                sample,
                input.mu00_eV,
                input.M_2x2_GeV[0][0], input.M_2x2_GeV[0][1], input.M_2x2_GeV[1][0], input.M_2x2_GeV[1][1],
                input.mD_3x2_GeV[0][0], input.mD_3x2_GeV[0][1], input.mD_3x2_GeV[1][0], input.mD_3x2_GeV[1][1], input.mD_3x2_GeV[2][0], input.mD_3x2_GeV[2][1],
                input.mu_H_2x2_eV[0][0], input.mu_H_2x2_eV[0][1], input.mu_H_2x2_eV[1][0], input.mu_H_2x2_eV[1][1], input.mu_H0_2x1_eV[0], input.mu_H0_2x1_eV[1],
                result.masses_eV[ordered_mass_index[0]],
                result.masses_eV[ordered_mass_index[1]],
                result.masses_eV[ordered_mass_index[2]],
                result.masses_eV[ordered_mass_index[3]],
                result.dm21_eV2,
                result.dm31_eV2,
                dm32_eV2,
                result.dm41_eV2,
                abs_u3[0][0],
                abs_u3[0][1],
                abs_u3[0][2],
                abs_u3[1][0],
                abs_u3[1][1],
                abs_u3[1][2],
                abs_u3[2][0],
                abs_u3[2][1],
                abs_u3[2][2],
                result.masses_full_eV[4],
                result.masses_full_eV[5],
                result.masses_full_eV[6],
                result.masses_full_eV[7],
                dm51,
                dm61,
                dm71,
                dm81,
                fabs(result.mixing_8x8[0][4]),
                fabs(result.mixing_8x8[0][5]),
                fabs(result.mixing_8x8[0][6]),
                fabs(result.mixing_8x8[0][7]),
                br_muegamma,
                eta_pass ? 1 : 0);
            }

        {
            const int done = sample + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (done * 100) / cfg->inverse_random_samples
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar(done, cfg->inverse_random_samples, kept_count, eta_kept_count);
                last_percent_printed = percent;
            }
        }
    }

    if (cfg->inverse_random_samples > 0) {
        print_progress_bar(cfg->inverse_random_samples,
                   cfg->inverse_random_samples,
                   kept_count,
                   eta_kept_count);
        printf("\n");
    }

    fclose(out);
    printf("Scan inverse PMNS filtre: %lld tested, %lld kept (PMNS), %lld kept (PMNS+eta)\n", tested_count, kept_count, eta_kept_count);
    {
        const clock_t t1 = clock();
        const double elapsed_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
        printf("Temps d'execution: %.3f s\n", elapsed_s);
    }
    printf("CSV genere (inverse PMNS filtre 3+1): %s\n", cfg->output_inverse_pmns_filter_csv_path);

    return 0;
}
