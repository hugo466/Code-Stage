#include "scan.h"

#include "constants.h"
#include "io_csv.h"
#include "oscillation.h"
#include "pmns.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>

static long long estimate_linear_point_count(double min_value, double max_value, double step) {
    if (step <= 0.0 || max_value < min_value) {
        return 0;
    }
    const double span = (max_value - min_value) / step;
    return (long long)floor(span + 1.0 + 1e-12);
}

static void print_progress_bar_scan(const char *label, long long done, long long total) {
    const int bar_width = 40;
    const double ratio = (total > 0) ? ((double)done / (double)total) : 1.0;
    const int filled = (int)(ratio * bar_width + 0.5);
    const int percent = (int)(ratio * 100.0 + 0.5);

    printf("\r%s [", label);
    for (int i = 0; i < bar_width; ++i) {
        putchar(i < filled ? '#' : '-');
    }
    printf("] %3d%% (%lld/%lld)", percent, done, total);
    fflush(stdout);

    if (done >= total) {
        printf("\n");
        fflush(stdout);
    }
}

int run_scan_energy_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 1 || cfg->dm41_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_csv_path[0] == '\0') {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u[N_FLAVORS_3P1][N_FLAVORS_3P1];
    if (pmns_build_3plus1(cfg, u) != 0) {
        return 3;
    }

    FILE *out = open_probability_csv(cfg->output_csv_path);
    if (!out) {
        return 2;
    }

    const long long n_energy = estimate_linear_point_count(cfg->energy_min_GeV, cfg->energy_max_GeV, cfg->energy_step_GeV);
    const long long total_points = (long long)cfg->dm41_count * n_energy;
    long long done_points = 0;
    int last_percent = -1;

    for (int idx = 0; idx < cfg->dm41_count; ++idx) {
        const double dm41 = cfg->dm41_values_eV2[idx];
        const double mass_sq[N_FLAVORS_3P1] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41};
        const double complex u_mu4 = u[FLAVOR_MU][3];
        const double complex u_e4 = u[FLAVOR_E][3];

        for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
            const double complex amp_mumu = transition_amplitude_n(
                N_FLAVORS_3P1, FLAVOR_MU, FLAVOR_MU, energy, cfg->baseline_km, mass_sq, u);
            const double complex amp_mue = transition_amplitude_n(
                N_FLAVORS_3P1, FLAVOR_MU, FLAVOR_E, energy, cfg->baseline_km, mass_sq, u);

            const double p_mumu = probability_with_config_n(
                N_FLAVORS_3P1,
                FLAVOR_MU,
                FLAVOR_MU,
                energy,
                cfg->baseline_km,
                mass_sq,
                u,
                cfg,
                0);

            const double p_mue = probability_with_config_n(
                N_FLAVORS_3P1,
                FLAVOR_MU,
                FLAVOR_E,
                energy,
                cfg->baseline_km,
                mass_sq,
                u,
                cfg,
                0);

            write_probability_row(out, energy, dm41, u_mu4, u_e4, amp_mumu, p_mumu, amp_mue, p_mue);

            ++done_points;
            if (total_points > 0) {
                const int percent = (int)((done_points * 100) / total_points);
                if (percent != last_percent || done_points == total_points) {
                    print_progress_bar_scan("Scan energy 3+1", done_points, total_points);
                    last_percent = percent;
                }
            }
        }
    }

    fclose(out);
    printf("CSV genere: %s\n", cfg->output_csv_path);
    return 0;
}

int run_scan_energy_3p2(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 2 || !cfg->dm41_3p2_is_set || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_csv_3p2_path[0] == '\0') {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u[N_FLAVORS_3P2][N_FLAVORS_3P2];
    if (pmns_build_3plus2(cfg, u) != 0) {
        return 3;
    }

    FILE *out = open_probability_csv_3p2(cfg->output_csv_3p2_path);
    if (!out) {
        return 2;
    }

    const double dm41 = cfg->dm41_3p2_eV2;
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

    const long long n_energy = estimate_linear_point_count(cfg->energy_min_GeV, cfg->energy_max_GeV, cfg->energy_step_GeV);
    const long long total_points = (long long)n_baselines * (long long)cfg->dm54_count * n_energy;
    long long done_points = 0;
    int last_percent = -1;

    for (int idx_base = 0; idx_base < n_baselines; ++idx_base) {
        const double baseline_km = baselines[idx_base];

        for (int idx54 = 0; idx54 < cfg->dm54_count; ++idx54) {
            const double dm54 = cfg->dm54_values_eV2[idx54];
            const double dm51 = dm41 + dm54;
            const double mass_sq[N_FLAVORS_3P2] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41, dm51};

            for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
                const double complex amp_mumu = transition_amplitude_n(
                    N_FLAVORS_3P2, FLAVOR_MU, FLAVOR_MU, energy, baseline_km, mass_sq, u);
                const double complex amp_mue = transition_amplitude_n(
                    N_FLAVORS_3P2, FLAVOR_MU, FLAVOR_E, energy, baseline_km, mass_sq, u);

                const double p_mumu = probability_with_config_n(
                    N_FLAVORS_3P2,
                    FLAVOR_MU,
                    FLAVOR_MU,
                    energy,
                    baseline_km,
                    mass_sq,
                    u,
                    cfg,
                    0);

                const double p_mue = probability_with_config_n(
                    N_FLAVORS_3P2,
                    FLAVOR_MU,
                    FLAVOR_E,
                    energy,
                    baseline_km,
                    mass_sq,
                    u,
                    cfg,
                    0);

                write_probability_row_3p2(out, baseline_km, energy, dm41, dm54, dm51, amp_mumu, p_mumu, amp_mue, p_mue);

                ++done_points;
                if (total_points > 0) {
                    const int percent = (int)((done_points * 100) / total_points);
                    if (percent != last_percent || done_points == total_points) {
                        print_progress_bar_scan("Scan energy 3+2", done_points, total_points);
                        last_percent = percent;
                    }
                }
            }
        }
    }

    fclose(out);
    printf("CSV genere (3+2): %s\n", cfg->output_csv_3p2_path);
    return 0;
}

int run_scan_distance_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->n_sterile < 1 || cfg->dist_step_km <= 0.0 || cfg->output_dist_csv_path[0] == '\0' || cfg->dm41_count <= 0) {
        return 1;
    }

    ensure_directory_exists("data");

    double complex u[N_FLAVORS_3P1][N_FLAVORS_3P1];
    if (pmns_build_3plus1(cfg, u) != 0) {
        return 3;
    }

    // Build standard 3x3 PMNS matrix for comparison
    double complex u3[3][3];
    pmns_build_3x3(
        deg_to_rad(cfg->theta12_deg),
        deg_to_rad(cfg->theta13_deg),
        deg_to_rad(cfg->theta23_deg),
        deg_to_rad(cfg->delta_cp_deg),
        u3);

    FILE *out = open_distance_csv(cfg->output_dist_csv_path);
    if (!out) {
        return 2;
    }

    // mass_sq_3nu ne dépend pas de dm41 : défini une fois pour toutes
    const double mass_sq_3nu[3] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2};

    // Boucle baseline en extérieur : calcul 3ν une seule fois par distance
    for (double baseline_km = cfg->dist_min_km; baseline_km <= cfg->dist_max_km + 1e-9; baseline_km += cfg->dist_step_km) {

        // Probabilités 3ν standard, indépendantes de dm41
        const double p_mumu_3nu = probability_with_config_n(
            3,
            FLAVOR_MU,
            FLAVOR_MU,
            cfg->dist_fixed_energy_GeV,
            baseline_km,
            mass_sq_3nu,
            u3,
            cfg,
            0);

        const double p_mue_3nu = probability_with_config_n(
            3,
            FLAVOR_MU,
            FLAVOR_E,
            cfg->dist_fixed_energy_GeV,
            baseline_km,
            mass_sq_3nu,
            u3,
            cfg,
            0);

        // Boucle dm41 en intérieur : seul le terme stérile varie
        for (int idx = 0; idx < cfg->dm41_count; ++idx) {
            const double dm41 = cfg->dm41_values_eV2[idx];
            const double mass_sq[N_FLAVORS_3P1] = {0.0, cfg->dm21_eV2, cfg->dm31_eV2, dm41};

            const double p_mumu = probability_with_config_n(
                N_FLAVORS_3P1,
                FLAVOR_MU,
                FLAVOR_MU,
                cfg->dist_fixed_energy_GeV,
                baseline_km,
                mass_sq,
                u,
                cfg,
                0);

            const double p_mue = probability_with_config_n(
                N_FLAVORS_3P1,
                FLAVOR_MU,
                FLAVOR_E,
                cfg->dist_fixed_energy_GeV,
                baseline_km,
                mass_sq,
                u,
                cfg,
                0);

            write_distance_row(out, baseline_km, dm41, p_mumu, p_mue, p_mumu_3nu, p_mue_3nu);
        }
    }

    fclose(out);
    printf("CSV genere: %s\n", cfg->output_dist_csv_path);
    return 0;
}
