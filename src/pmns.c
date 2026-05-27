#include "pmns.h"

#include "oscillation.h"

#include <math.h>
#include <stdio.h>

static void set_identity(int size, double complex u[size][size]) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            u[row][col] = (row == col) ? 1.0 : 0.0;
        }
    }
}

static void rotate_mass_columns(
    int size,
    double complex u[size][size],
    int i,
    int j,
    double theta,
    double delta_phase) {

    const double c = cos(theta);
    const double s = sin(theta);

    for (int row = 0; row < size; ++row) {
        const double complex col_i_old = u[row][i];
        const double complex col_j_old = u[row][j];

        u[row][i] = c * col_i_old + s * cexp(-I * delta_phase) * col_j_old;
        u[row][j] = -s * cexp(I * delta_phase) * col_i_old + c * col_j_old;
    }
}

static void extend_pmns_3x3(int size, const double complex u3[3][3], double complex u[size][size]) {
    set_identity(size, u);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            u[row][col] = u3[row][col];
        }
    }
}

void pmns_build_3x3(
    double theta12,
    double theta13,
    double theta23,
    double delta_cp,
    double complex u3[3][3]) {

    const double c12 = cos(theta12);
    const double s12 = sin(theta12);
    const double c13 = cos(theta13);
    const double s13 = sin(theta13);
    const double c23 = cos(theta23);
    const double s23 = sin(theta23);

    u3[0][0] = c12 * c13;
    u3[0][1] = s12 * c13;
    u3[0][2] = s13 * cexp(-I * delta_cp);

    u3[1][0] = -s12 * c23 - c12 * s23 * s13 * cexp(I * delta_cp);
    u3[1][1] = c12 * c23 - s12 * s23 * s13 * cexp(I * delta_cp);
    u3[1][2] = s23 * c13;

    u3[2][0] = s12 * s23 - c12 * c23 * s13 * cexp(I * delta_cp);
    u3[2][1] = -c12 * s23 - s12 * c23 * s13 * cexp(I * delta_cp);
    u3[2][2] = c23 * c13;
}

static double pmns_3p2_target_cost(
    const SimulationConfig *cfg,
    double theta14_deg,
    double theta24_deg,
    double theta15_deg,
    double theta25_deg,
    double target_abs) {

    SimulationConfig local_cfg = *cfg;
    local_cfg.theta_active_sterile_deg[0][0] = theta14_deg;
    local_cfg.theta_active_sterile_deg[1][0] = theta24_deg;
    local_cfg.theta_active_sterile_deg[0][1] = theta15_deg;
    local_cfg.theta_active_sterile_deg[1][1] = theta25_deg;

    double complex u5[5][5];
    if (pmns_build_general(&local_cfg, 2, u5) != 0) {
        return 1e12;
    }

    const double ue4 = cabs(u5[0][3]);
    const double ue5 = cabs(u5[0][4]);
    const double umu4 = cabs(u5[1][3]);
    const double umu5 = cabs(u5[1][4]);

    const double d1 = ue4 - target_abs;
    const double d2 = ue5 - target_abs;
    const double d3 = umu4 - target_abs;
    const double d4 = umu5 - target_abs;
    return d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4;
}

static double clamp_angle_deg(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 89.0) {
        return 89.0;
    }
    return value;
}

static void optimize_pmns_3p2_angles_for_targets(
    const SimulationConfig *cfg,
    double target_abs,
    double *theta14_deg,
    double *theta24_deg,
    double *theta15_deg,
    double *theta25_deg) {

    double angles[4] = {
        cfg->theta_active_sterile_deg[0][0],
        cfg->theta_active_sterile_deg[1][0],
        cfg->theta_active_sterile_deg[0][1],
        cfg->theta_active_sterile_deg[1][1],
    };

    const double steps[] = {10.0, 2.0, 0.5, 0.1, 0.02};
    double best_cost = pmns_3p2_target_cost(cfg, angles[0], angles[1], angles[2], angles[3], target_abs);

    for (int s = 0; s < (int)(sizeof(steps) / sizeof(steps[0])); ++s) {
        const double step = steps[s];
        int improved_at_this_step = 1;

        while (improved_at_this_step) {
            improved_at_this_step = 0;

            for (int k = 0; k < 4; ++k) {
                const double original = angles[k];
                double local_best_angle = original;
                double local_best_cost = best_cost;

                const double candidates[2] = {
                    clamp_angle_deg(original - step),
                    clamp_angle_deg(original + step)
                };

                for (int c = 0; c < 2; ++c) {
                    angles[k] = candidates[c];
                    const double cost = pmns_3p2_target_cost(cfg, angles[0], angles[1], angles[2], angles[3], target_abs);

                    if (cost < local_best_cost) {
                        local_best_cost = cost;
                        local_best_angle = angles[k];
                    }
                }

                angles[k] = local_best_angle;
                if (local_best_cost + 1e-16 < best_cost) {
                    best_cost = local_best_cost;
                    improved_at_this_step = 1;
                }
            }
        }
    }

    *theta14_deg = angles[0];
    *theta24_deg = angles[1];
    *theta15_deg = angles[2];
    *theta25_deg = angles[3];
}

int pmns_build_general(const SimulationConfig *cfg, int n_sterile, double complex u[3 + n_sterile][3 + n_sterile]) {
    if (!cfg || n_sterile < 1 || n_sterile > MAX_STERILE_NEUTRINOS) {
        return 1;
    }

    const int n_flavors = 3 + n_sterile;
    double complex u3[3][3];

    pmns_build_3x3(
        deg_to_rad(cfg->theta12_deg),
        deg_to_rad(cfg->theta13_deg),
        deg_to_rad(cfg->theta23_deg),
        deg_to_rad(cfg->delta_cp_deg),
        u3);

    extend_pmns_3x3(n_flavors, u3, u);

    for (int sterile_idx = 0; sterile_idx < n_sterile; ++sterile_idx) {
        const int mass_idx = 3 + sterile_idx;
        for (int active_idx = 0; active_idx < 3; ++active_idx) {
            rotate_mass_columns(
                n_flavors,
                u,
                active_idx,
                mass_idx,
                deg_to_rad(cfg->theta_active_sterile_deg[active_idx][sterile_idx]),
                deg_to_rad(cfg->delta_active_sterile_deg[active_idx][sterile_idx]));
        }
    }

    for (int i = 0; i < n_sterile; ++i) {
        for (int j = i + 1; j < n_sterile; ++j) {
            rotate_mass_columns(
                n_flavors,
                u,
                3 + i,
                3 + j,
                deg_to_rad(cfg->theta_sterile_sterile_deg[i][j]),
                deg_to_rad(cfg->delta_sterile_sterile_deg[i][j]));
        }
    }

    return 0;
}

int pmns_build_3plus1(const SimulationConfig *cfg, double complex u[4][4]) {
    return pmns_build_general(cfg, 1, u);
}

int pmns_build_3plus2(const SimulationConfig *cfg, double complex u[5][5]) {
    if (!cfg) {
        return 1;
    }

    const double target_abs = 0.316;
    double theta14_opt = cfg->theta_active_sterile_deg[0][0];
    double theta24_opt = cfg->theta_active_sterile_deg[1][0];
    double theta15_opt = cfg->theta_active_sterile_deg[0][1];
    double theta25_opt = cfg->theta_active_sterile_deg[1][1];

    optimize_pmns_3p2_angles_for_targets(
        cfg,
        target_abs,
        &theta14_opt,
        &theta24_opt,
        &theta15_opt,
        &theta25_opt);

    printf("Angles optimises (deg) pour |Ue4|=|Ue5|=|Umu4|=|Umu5|=%.3f: theta14=%.6f, theta24=%.6f, theta15=%.6f, theta25=%.6f\n",
           target_abs,
           theta14_opt,
           theta24_opt,
           theta15_opt,
           theta25_opt);

    SimulationConfig local_cfg = *cfg;
    local_cfg.theta_active_sterile_deg[0][0] = theta14_opt;
    local_cfg.theta_active_sterile_deg[1][0] = theta24_opt;
    local_cfg.theta_active_sterile_deg[0][1] = theta15_opt;
    local_cfg.theta_active_sterile_deg[1][1] = theta25_opt;

    return pmns_build_general(&local_cfg, 2, u);
}