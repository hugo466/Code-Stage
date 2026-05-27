#include "scan.h"

#include "casas_ibarra.h"
#include "constants.h"
#include "inverse_seesaw.h"
#include "oscillation.h"
#include "pmns.h"
#include "utils.h"

#include <complex.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void print_mixing_matrix_4x4(const InverseSeesaw3p1Result *result, double mu00_eV) {
    printf("Matrice de melange 4x4 (mu00 = %.6g eV):\n", mu00_eV);
    for (int row = 0; row < 4; ++row) {
        printf("  [");
        for (int col = 0; col < 4; ++col) {
            printf("% .8e%s", result->mixing_4x4[row][col], (col < 3 ? ", " : ""));
        }
        printf("]\n");
    }
}

static double clamp_unit_iss24(double x) {
    if (x > 1.0) return 1.0;
    if (x < -1.0) return -1.0;
    return x;
}

static double loop_function_g_gamma_iss24(double x) {
    if (x <= 0.0) {
        return 10.0 / 3.0;
    }
    const double one_minus_x = 1.0 - x;
    if (fabs(one_minus_x) < 1e-8) {
        return 17.0 / 6.0;
    }
    const double num = 10.0 - 43.0 * x + 78.0 * x * x - 49.0 * x * x * x + 4.0 * x * x * x * x + 18.0 * x * x * x * log(x);
    const double den = 3.0 * one_minus_x * one_minus_x * one_minus_x * one_minus_x;
    return num / den;
}

static double uniform_random(double min_value, double max_value) {
    const double t = (double)rand() / (double)RAND_MAX;
    return min_value + (max_value - min_value) * t;
}

static double uniform_random_log10(double min_value, double max_value) {
    const double log_min = log10(min_value);
    const double log_max = log10(max_value);
    const double t = (double)rand() / (double)RAND_MAX;
    return pow(10.0, log_min + (log_max - log_min) * t);
}

static double vector_norm3(const double v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static int normalize3(double v[3]) {
    const double n = vector_norm3(v);
    if (n < 1e-15) {
        return 1;
    }
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
    return 0;
}

static double dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void remove_projection3(double v[3], const double basis[3]) {
    const double c = dot3(v, basis);
    v[0] -= c * basis[0];
    v[1] -= c * basis[1];
    v[2] -= c * basis[2];
}

static void mat3_vec_mul(const double a[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i) {
        out[i] = a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2];
    }
}

static double clamp_unit_real(double x) {
    if (x > 1.0) return 1.0;
    if (x < -1.0) return -1.0;
    return x;
}

static void build_u3p1_from_zeta(const double p1[3],
                                 const double p2[3],
                                 const double y[3],
                                 const double zeta[3],
                                 double out_u[4][4]) {
    const double s2 = zeta[0] * zeta[0] + zeta[1] * zeta[1] + zeta[2] * zeta[2];
    const double s = sqrt(fmax(0.0, s2));
    const double c0 = sqrt(fmax(0.0, 1.0 - s2));

    double A[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    if (s > 1e-14) {
        const double w[3] = {zeta[0] / s, zeta[1] / s, zeta[2] / s};
        const double alpha = (1.0 - c0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A[i][j] -= alpha * w[i] * w[j];
            }
        }
    }

    double col_m2_active[3];
    double col_m3_active[3];
    double col_m4_active[3] = {zeta[0], zeta[1], zeta[2]};
    double col_m1_active[3] = {y[0], y[1], y[2]};
    mat3_vec_mul(A, p1, col_m2_active);
    mat3_vec_mul(A, p2, col_m3_active);

    const double bottom_m2 = -(zeta[0] * p1[0] + zeta[1] * p1[1] + zeta[2] * p1[2]);
    const double bottom_m3 = -(zeta[0] * p2[0] + zeta[1] * p2[1] + zeta[2] * p2[2]);
    const double bottom_m4 = c0;
    const double bottom_m1 = 0.0;

    out_u[0][0] = col_m2_active[0];
    out_u[1][0] = col_m2_active[1];
    out_u[2][0] = col_m2_active[2];
    out_u[3][0] = bottom_m2;

    out_u[0][1] = col_m3_active[0];
    out_u[1][1] = col_m3_active[1];
    out_u[2][1] = col_m3_active[2];
    out_u[3][1] = bottom_m3;

    out_u[0][2] = col_m4_active[0];
    out_u[1][2] = col_m4_active[1];
    out_u[2][2] = col_m4_active[2];
    out_u[3][2] = bottom_m4;

    out_u[0][3] = col_m1_active[0];
    out_u[1][3] = col_m1_active[1];
    out_u[2][3] = col_m1_active[2];
    out_u[3][3] = bottom_m1;
}

static int check_orthonormal_columns_4x4_real(const double u[4][4], double tol) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double dot = 0.0;
            for (int r = 0; r < 4; ++r) {
                dot += u[r][i] * u[r][j];
            }
            if (i == j) {
                if (fabs(dot - 1.0) > tol) {
                    return 0;
                }
            } else {
                if (fabs(dot) > tol) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int inverse_2x2_real(const double m[2][2], double inv[2][2]) {
    const double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (fabs(det) < 1e-18) {
        return 1;
    }

    inv[0][0] = m[1][1] / det;
    inv[0][1] = -m[0][1] / det;
    inv[1][0] = -m[1][0] / det;
    inv[1][1] = m[0][0] / det;
    return 0;
}

static void mat3_mul(const double a[3][3], const double b[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
}

static void mat3_transpose(const double in[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = in[j][i];
        }
    }
}

static double frob_norm_4x4(const double m[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            s += m[i][j] * m[i][j];
        }
    }
    return sqrt(s);
}

static void build_basis_from_pmns_exact(const SimulationConfig *cfg, double y[3], double p1[3], double p2[3]) {

    double c0[3], c1[3], c2[3];

    const int have_nufit = (cfg->inverse_nufit_theta12_deg != 0.0 ||
                            cfg->inverse_nufit_theta13_deg != 0.0 ||
                            cfg->inverse_nufit_theta23_deg != 0.0 ||
                            cfg->inverse_nufit_deltacp_deg != 0.0);

    if (have_nufit) {
        const double t12 = deg_to_rad(cfg->inverse_nufit_theta12_deg);
        const double t13 = deg_to_rad(cfg->inverse_nufit_theta13_deg);
        const double t23 = deg_to_rad(cfg->inverse_nufit_theta23_deg);
        const double c12 = cos(t12);
        const double s12 = sin(t12);
        const double c13 = cos(t13);
        const double s13 = sin(t13);
        const double c23 = cos(t23);
        const double s23 = sin(t23);

        y[0] = c12 * c13;
        y[1] = -s12 * c23 - c12 * s23 * s13;
        y[2] = s12 * s23 - c12 * c23 * s13;

        p1[0] = s12 * c13;
        p1[1] = c12 * c23 - s12 * s23 * s13;
        p1[2] = -c12 * s23 - s12 * c23 * s13;

        p2[0] = s13;
        p2[1] = s23 * c13;
        p2[2] = c23 * c13;

        normalize3(y);
        normalize3(p1);
        normalize3(p2);
        return;

    } else {
        printf("INFO: Base active construite depuis les milieux des intervalles experimentaux (angles NuFIT absents)\n");
        for (int i = 0; i < 3; ++i) {
            c0[i] = 0.5 * (cfg->inverse_pmns_abs_min_3x3[i][0] + cfg->inverse_pmns_abs_max_3x3[i][0]);
            c1[i] = 0.5 * (cfg->inverse_pmns_abs_min_3x3[i][1] + cfg->inverse_pmns_abs_max_3x3[i][1]);
            c2[i] = 0.5 * (cfg->inverse_pmns_abs_min_3x3[i][2] + cfg->inverse_pmns_abs_max_3x3[i][2]);
        }
    }

    /* Gram-Schmidt orthonormalization */
    memcpy(y, c0, sizeof(double) * 3);
    if (normalize3(y) != 0) {
        y[0] = 1.0; y[1] = 0.0; y[2] = 0.0;
    }

    memcpy(p1, c1, sizeof(double) * 3);
    remove_projection3(p1, y);
    if (normalize3(p1) != 0) {
        p1[0] = 0.0; p1[1] = 1.0; p1[2] = 0.0;
        remove_projection3(p1, y);
        normalize3(p1);
    }

    memcpy(p2, c2, sizeof(double) * 3);
    remove_projection3(p2, y);
    remove_projection3(p2, p1);
    if (normalize3(p2) != 0) {
        p2[0] = y[1] * p1[2] - y[2] * p1[1];
        p2[1] = y[2] * p1[0] - y[0] * p1[2];
        p2[2] = y[0] * p1[1] - y[1] * p1[0];
        normalize3(p2);
    }
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

static int eta_constraints_satisfied_3p1(const InverseSeesaw3p1Result *result,
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

            {
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
    }

    return 1;
}

int run_scan_inverse_construct_23_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->output_inverse_construct_23_csv_path[0] == '\0' || cfg->inverse_construct_23_samples <= 0) {
        return 1;
    }

    const double dm21_target = cfg->dm21_eV2;
    const double dm31_target = cfg->dm31_eV2;
    if (dm21_target <= 0.0 || dm31_target <= 0.0) {
        return 2;
    }

    if (cfg->inverse_construct_23_seed > 0) {
        srand((unsigned int)cfg->inverse_construct_23_seed);
    } else {
        srand((unsigned int)time(NULL));
    }

    ensure_directory_exists("data");

    FILE *out = fopen(cfg->output_inverse_construct_23_csv_path, "w");
    if (!out) {
        return 3;
    }

        fprintf(out,
            "point_id,solve_ok,eta_pass,pmns_pass,dm41_target_eV2,dm21_target_eV2,dm31_target_eV2,"
            "dm21_calc_eV2,dm31_calc_eV2,dm41_calc_eV2,"
            "zeta_norm,zeta_direction_deg,"
            "theta14_deg,theta24_deg,theta34_deg,delta_cp_sterile_deg,"
            "f11,f12,f21,f22,det_f,M1_GeV,M2_GeV,"
            "pmns_rms_abs_error,mL_rel_frob_error\n");

    double y[3], p1[3], p2[3];
    build_basis_from_pmns_exact(cfg, y, p1, p2);
    double pmns_target_abs[3][3];
    
    // Check if NuFIT angles are provided (non-zero values indicate they were loaded)
    if (cfg->inverse_nufit_theta12_deg != 0.0 || cfg->inverse_nufit_theta23_deg != 0.0 || 
        cfg->inverse_nufit_theta13_deg != 0.0 || cfg->inverse_nufit_deltacp_deg != 0.0) {
        // Build PMNS matrix from NuFIT 6.0 2024 angles
        double complex u_nufit[3][3];
        pmns_build_3x3(
            deg_to_rad(cfg->inverse_nufit_theta12_deg),
            deg_to_rad(cfg->inverse_nufit_theta13_deg),
            deg_to_rad(cfg->inverse_nufit_theta23_deg),
            deg_to_rad(cfg->inverse_nufit_deltacp_deg),
            u_nufit);
        
        // Extract absolute values into target matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                pmns_target_abs[i][j] = cabs(u_nufit[i][j]);
            }
        }

    } else {
        // Fallback: use midpoint of intervals if angles are not provided
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                pmns_target_abs[i][j] = 0.5 * (cfg->inverse_pmns_abs_min_3x3[i][j] + cfg->inverse_pmns_abs_max_3x3[i][j]);
            }
        }
        printf("INFO: Using midpoint PMNS target (NuFIT angles not provided)\n");
    }

    int solved_ok = 0;
    for (int sample = 0; sample < cfg->inverse_construct_23_samples; ++sample) {
        const double m4_min = sqrt(cfg->inverse_construct_23_dm41_min_eV2);
        const double m4_max = sqrt(cfg->inverse_construct_23_dm41_max_eV2);
        const double m4 = uniform_random_log10(m4_min, m4_max);
        const double dm41 = m4 * m4;
        const double m2 = sqrt(dm21_target);
        const double m3 = sqrt(dm31_target);

        /* Sample active-sterile mixing vector zeta = P c with y^T zeta = 0 */
        const double zeta_norm = uniform_random(cfg->inverse_construct_23_zeta_norm_min,
                                                cfg->inverse_construct_23_zeta_norm_max);
        const double zeta_direction_deg = uniform_random(cfg->inverse_construct_23_zeta_direction_min_deg,
                                                         cfg->inverse_construct_23_zeta_direction_max_deg);
        const double phi = deg_to_rad(zeta_direction_deg);
        const double c1 = zeta_norm * cos(phi);
        const double c2 = zeta_norm * sin(phi);
        double zeta[3] = {
            p1[0] * c1 + p2[0] * c2,
            p1[1] * c1 + p2[1] * c2,
            p1[2] * c1 + p2[2] * c2
        };

        const double ue4 = fabs(zeta[0]);
        const double umu4 = fabs(zeta[1]);
        const double utau4 = fabs(zeta[2]);
        const double c14 = sqrt(fmax(0.0, 1.0 - ue4 * ue4));
        const double sin24 = (c14 > 1e-14) ? umu4 / c14 : 0.0;
        const double c24 = sqrt(fmax(0.0, 1.0 - sin24 * sin24));
        const double denom34 = c14 * c24;
        const double sin34 = (denom34 > 1e-14) ? utau4 / denom34 : 0.0;

        const double theta14 = asin(clamp_unit_real(ue4)) * 180.0 / M_PI;
        const double theta24 = asin(clamp_unit_real(sin24)) * 180.0 / M_PI;
        const double theta34 = asin(clamp_unit_real(sin34)) * 180.0 / M_PI;
        const double delta_cp_sterile = zeta_direction_deg;

        /* Construct M3 as diagonal mass matrix in eV (no rotation applied) */
        /* Build E = R_zeta * blockdiag(U_nu, 1) early — used for PMNS pass check */
        double E[4][4] = {{0.0}};
        build_u3p1_from_zeta(p1, p2, y, zeta, E);

        /* Check active 3x3 block of E against NuFIT bounds.
         * Column mapping: mass=0(nu1)->col3, mass=1(nu2)->col0, mass=2(nu3)->col1 */
        int pmns_pass = 1;
        double pmns_rms = 0.0;
        {
            const int mass_to_col[3] = {3, 0, 1};
            for (int flavor = 0; flavor < 3; ++flavor) {
                for (int mass = 0; mass < 3; ++mass) {
                    const double v = fabs(E[flavor][mass_to_col[mass]]);
                    const double diff = v - pmns_target_abs[flavor][mass];
                    pmns_rms += diff * diff;
                    if (v < cfg->inverse_pmns_abs_min_3x3[flavor][mass] ||
                        v > cfg->inverse_pmns_abs_max_3x3[flavor][mass]) {
                        pmns_pass = 0;
                    }
                }
            }
            pmns_rms = sqrt(pmns_rms / 9.0);
        }

        double M3[3][3] = {{m2, 0.0, 0.0}, {0.0, m3, 0.0}, {0.0, 0.0, m4}};

        double f[2][2];
        double det_f = 0.0;
        int have_f = 0;
        for (int trial = 0; trial < 128; ++trial) {
            f[0][0] = uniform_random(cfg->inverse_construct_23_f11_min, cfg->inverse_construct_23_f11_max);
            f[0][1] = uniform_random(cfg->inverse_construct_23_f12_min, cfg->inverse_construct_23_f12_max);
            f[1][0] = uniform_random(cfg->inverse_construct_23_f21_min, cfg->inverse_construct_23_f21_max);
            f[1][1] = uniform_random(cfg->inverse_construct_23_f22_min, cfg->inverse_construct_23_f22_max);
            det_f = f[0][0] * f[1][1] - f[0][1] * f[1][0];
            if (fabs(det_f) >= cfg->inverse_construct_23_f_det_min_abs) {
                have_f = 1;
                break;
            }
        }
        if (!have_f) {
                fprintf(out,
                    "%d,0,-1,-1,%.10e,%.10e,%.10e,"
                    "nan,nan,nan,"
                    "%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%.10e,%.10e,nan,nan,nan,nan\n",
                    sample + 1,
                    dm41,
                    dm21_target,
                    dm31_target,
                    zeta_norm,
                    zeta_direction_deg,
                    theta14,
                    theta24,
                    theta34,
                    delta_cp_sterile,
                    f[0][0], f[0][1], f[1][0], f[1][1], det_f);
            continue;
        }

        const double M1 = uniform_random(cfg->inverse_construct_23_M1_min_GeV, cfg->inverse_construct_23_M1_max_GeV);
        const double M2 = uniform_random(cfg->inverse_construct_23_M2_min_GeV, cfg->inverse_construct_23_M2_max_GeV);

        double finv[2][2];
        if (inverse_2x2_real(f, finv) != 0) {
                fprintf(out,
                    "%d,0,-1,-1,%.10e,%.10e,%.10e,nan,nan,nan,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,nan,nan\n",
                    sample + 1,
                    dm41,
                    dm21_target,
                    dm31_target,
                    zeta_norm,
                    zeta_direction_deg,
                    theta14,
                    theta24,
                    theta34,
                    delta_cp_sterile,
                    f[0][0], f[0][1], f[1][0], f[1][1], det_f,
                    M1, M2);
            continue;
        }

        double Qinv[3][3] = {
            {finv[0][0], finv[0][1], 0.0},
            {finv[1][0], finv[1][1], 0.0},
            {0.0, 0.0, -1.0}
        };

        double QinvT[3][3], tmp3[3][3], mu3[3][3];
        mat3_transpose(Qinv, QinvT);
        mat3_mul(Qinv, M3, tmp3);
        mat3_mul(tmp3, QinvT, mu3);

        double P[3][2] = {
            {p1[0], p2[0]},
            {p1[1], p2[1]},
            {p1[2], p2[2]}
        };
        double F[3][2] = {{0.0}};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                F[i][j] = P[i][0] * f[0][j] + P[i][1] * f[1][j];
            }
        }

        InverseSeesaw3p1Input input;
        memset(&input, 0, sizeof(input));

        input.M_2x2_GeV[0][0] = M1;
        input.M_2x2_GeV[1][1] = M2;
        input.M_2x2_GeV[0][1] = 0.0;
        input.M_2x2_GeV[1][0] = 0.0;

        for (int a = 0; a < 3; ++a) {
            input.mD_3x2_GeV[a][0] = F[a][0] * M1;
            input.mD_3x2_GeV[a][1] = F[a][1] * M2;
        }

        input.mu_H_2x2_eV[0][0] = mu3[0][0];
        input.mu_H_2x2_eV[0][1] = mu3[0][1];
        input.mu_H_2x2_eV[1][0] = mu3[1][0];
        input.mu_H_2x2_eV[1][1] = mu3[1][1];
        input.mu_H0_2x1_eV[0] = mu3[0][2];
        input.mu_H0_2x1_eV[1] = mu3[1][2];
        input.mu00_eV = mu3[2][2];

        InverseSeesaw3p1Result result;
        const int solve_ret = inverse_seesaw_solve_3p1(&input, &result);
        if (solve_ret != 0) {
                fprintf(out,
                    "%d,0,-1,-1,%.10e,%.10e,%.10e,nan,nan,nan,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,nan,nan\n",
                    sample + 1,
                    dm41,
                    dm21_target,
                    dm31_target,
                    zeta_norm,
                    zeta_direction_deg,
                    theta14,
                    theta24,
                    theta34,
                    delta_cp_sterile,
                    f[0][0], f[0][1], f[1][0], f[1][1], det_f,
                    M1, M2);
            continue;
        }

        ++solved_ok;

        int ordered_mass_index[4] = {-1, -1, -1, -1};
        build_ordered_mass_indices_3p1(&result, ordered_mass_index);

        /* Build U_{3+1} = R_zeta * blockdiag(U_nu, 1), with massless column purely active */
        /* E already built at top of loop */

        {
            const double y_dot_zeta = y[0] * zeta[0] + y[1] * zeta[1] + y[2] * zeta[2];
            const double sterile_massless = fabs(E[3][3]);
            if (fabs(y_dot_zeta) > 1e-9 || sterile_massless > 1e-9 || !check_orthonormal_columns_4x4_real(E, 1e-8)) {
                fprintf(out,
                    "%d,0,-1,-1,%.10e,%.10e,%.10e,nan,nan,nan,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,nan,nan\n",
                    sample + 1,
                    dm41,
                    dm21_target,
                    dm31_target,
                    zeta_norm,
                    zeta_direction_deg,
                    theta14,
                    theta24,
                    theta34,
                    delta_cp_sterile,
                    f[0][0], f[0][1], f[1][0], f[1][1], det_f,
                    M1, M2);
                continue;
            }
        }

        /* Construct 4x4 mass matrix target in reduced basis (diagonal, masses in eV) */
        double M_reduced[4][4] = {
            {m2, 0.0, 0.0, 0.0},
            {0.0, m3, 0.0, 0.0},
            {0.0, 0.0, m4, 0.0},
            {0.0,   0.0,   0.0,   0.0}
        };

        /* Transform to flavor basis: Mtarget4 = E @ M_reduced @ E^T (all real) */
        double Mtarget4[4][4] = {{0.0}};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double s = 0.0;
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                        s += E[i][a] * M_reduced[a][b] * E[j][b];
                    }
                }
                Mtarget4[i][j] = s;
            }
        }

        double diff4[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                diff4[i][j] = result.m_light_4x4_eV[i][j] - Mtarget4[i][j];
            }
        }
        const double denom = fmax(frob_norm_4x4(Mtarget4), 1e-20);
        const double mL_rel_err = frob_norm_4x4(diff4) / denom;
        double eta_abs_3x3[3][3] = {{0.0}};
        const int eta_pass = eta_constraints_satisfied_3p1(&result, cfg, eta_abs_3x3);

        fprintf(out,
            "%d,1,%d,%d,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,"
                "%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e\n",
                sample + 1,
            eta_pass ? 1 : 0,
            pmns_pass,
                dm41,
                dm21_target,
                dm31_target,
                result.dm21_eV2,
                result.dm31_eV2,
                result.dm41_eV2,
                zeta_norm,
                zeta_direction_deg,
                theta14,
                theta24,
                theta34,
                delta_cp_sterile,
                f[0][0], f[0][1], f[1][0], f[1][1], det_f,
                M1, M2,
                pmns_rms,
                mL_rel_err);
    }

    fclose(out);
    printf("CSV genere (construction (2,3) 3+1): %s\n", cfg->output_inverse_construct_23_csv_path);
    // printf("Points resolus avec succes: %d / %d\n", solved_ok, cfg->inverse_construct_23_samples);
    return 0;
}

typedef struct {
    double mD_3x2_GeV[3][2];
    double M_2x2_GeV[2][2];
    double mu_H_2x2_eV[2][2];
    double mu_H0_2x2_eV[2][2];
    double mu00_2x2_eV[2][2];
} Iss24Input;

typedef struct {
    double m_full_9x9_eV[9][9];
    double masses_full_eV[9];
    double mixing_9x9[9][9];
    double masses_eV[5];
    double mixing_5x5[5][5];
    int sterile_idx[2];
    double dm21_eV2;
    double dm31_eV2;
    double dm41_eV2;
    double dm51_eV2;
    double theta14_deg;
    double theta24_deg;
    double theta34_deg;
    double theta15_deg;
    double theta25_deg;
    double theta35_deg;
} Iss24Result;

static int inverse_2x2_iss24(const double m[2][2], double out[2][2]) {
    const double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (fabs(det) < 1e-30) return 1;
    out[0][0] = m[1][1] / det;
    out[0][1] = -m[0][1] / det;
    out[1][0] = -m[1][0] / det;
    out[1][1] = m[0][0] / det;
    return 0;
}

static void jacobi_sym_n(int n, double *a, double *vec) {
    memset(vec, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; ++i) vec[i * n + i] = 1.0;

    for (int it = 0; it < 800; ++it) {
        int p = 0, q = 1;
        double mx = fabs(a[p * n + q]);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double v = fabs(a[i * n + j]);
                if (v > mx) {
                    mx = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (mx < 1e-12) break;

        const double app = a[p * n + p];
        const double aqq = a[q * n + q];
        const double apq = a[p * n + q];
        const double phi = 0.5 * atan2(2.0 * apq, (aqq - app));
        const double c = cos(phi), s = sin(phi);

        for (int k = 0; k < n; ++k) {
            if (k != p && k != q) {
                const double aik = a[k * n + p];
                const double akq = a[k * n + q];
                a[k * n + p] = c * aik - s * akq;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * aik + c * akq;
                a[q * n + k] = a[k * n + q];
            }
        }

        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vip = vec[k * n + p];
            const double viq = vec[k * n + q];
            vec[k * n + p] = c * vip - s * viq;
            vec[k * n + q] = s * vip + c * viq;
        }
    }
}

static void sort_pairs_abs_n(int n, double *vals, double *vec) {
    for (int i = 0; i < n - 1; ++i) {
        int best = i;
        double bestv = fabs(vals[i]);
        for (int j = i + 1; j < n; ++j) {
            const double v = fabs(vals[j]);
            if (v < bestv) {
                best = j;
                bestv = v;
            }
        }
        if (best != i) {
            const double t = vals[i];
            vals[i] = vals[best];
            vals[best] = t;
            for (int r = 0; r < n; ++r) {
                const double tv = vec[r * n + i];
                vec[r * n + i] = vec[r * n + best];
                vec[r * n + best] = tv;
            }
        }
    }
}

static int solve_iss24(const Iss24Input *input, Iss24Result *result) {
    if (!input || !result) return 1;
    memset(result, 0, sizeof(*result));

    double invM[2][2];
    if (inverse_2x2_iss24(input->M_2x2_GeV, invM) != 0) return 2;

    double x[3][2] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            x[i][j] = input->mD_3x2_GeV[i][0] * invM[0][j] + input->mD_3x2_GeV[i][1] * invM[1][j];
        }
    }

    double xmu[3][2] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            xmu[i][j] = x[i][0] * input->mu_H_2x2_eV[0][j] + x[i][1] * input->mu_H_2x2_eV[1][j];
        }
    }

    double mlight[5][5] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mlight[i][j] = xmu[i][0] * x[j][0] + xmu[i][1] * x[j][1];
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int s = 0; s < 2; ++s) {
            const double v = -(x[i][0] * input->mu_H0_2x2_eV[0][s] + x[i][1] * input->mu_H0_2x2_eV[1][s]);
            mlight[i][3 + s] = v;
            mlight[3 + s][i] = v;
        }
    }
    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
            mlight[3 + s][3 + t] = input->mu00_2x2_eV[s][t];
        }
    }

    memset(result->m_full_9x9_eV, 0, sizeof(result->m_full_9x9_eV));
    for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
            const double v = input->mD_3x2_GeV[a][i] * 1.0e9;
            result->m_full_9x9_eV[a][5 + i] = v;
            result->m_full_9x9_eV[5 + i][a] = v;
        }
    }
    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
            result->m_full_9x9_eV[3 + s][3 + t] = input->mu00_2x2_eV[s][t];
            result->m_full_9x9_eV[3 + s][7 + t] = input->mu_H0_2x2_eV[t][s];
            result->m_full_9x9_eV[7 + t][3 + s] = input->mu_H0_2x2_eV[t][s];
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            result->m_full_9x9_eV[5 + i][7 + j] = input->M_2x2_GeV[j][i] * 1.0e9;
            result->m_full_9x9_eV[7 + i][5 + j] = input->M_2x2_GeV[i][j] * 1.0e9;
            result->m_full_9x9_eV[7 + i][7 + j] = input->mu_H_2x2_eV[i][j];
        }
    }

    {
        double a[81], vec[81], vals[9];
        memcpy(a, result->m_full_9x9_eV, sizeof(a));
        jacobi_sym_n(9, a, vec);
        for (int i = 0; i < 9; ++i) vals[i] = a[i * 9 + i];
        sort_pairs_abs_n(9, vals, vec);
        for (int i = 0; i < 9; ++i) {
            result->masses_full_eV[i] = fabs(vals[i]);
            for (int r = 0; r < 9; ++r) result->mixing_9x9[r][i] = vec[r * 9 + i];
        }
    }

    {
        double a[25], vec[25], vals[5];
        memcpy(a, mlight, sizeof(a));
        jacobi_sym_n(5, a, vec);
        for (int i = 0; i < 5; ++i) vals[i] = a[i * 5 + i];
        sort_pairs_abs_n(5, vals, vec);
        for (int i = 0; i < 5; ++i) {
            result->masses_eV[i] = fabs(vals[i]);
            for (int r = 0; r < 5; ++r) result->mixing_5x5[r][i] = vec[r * 5 + i];
        }
    }

    int s1 = 0, s2 = 1;
    double n1 = hypot(fabs(result->mixing_5x5[3][0]), fabs(result->mixing_5x5[4][0]));
    double n2 = hypot(fabs(result->mixing_5x5[3][1]), fabs(result->mixing_5x5[4][1]));
    if (n2 > n1) {
        s1 = 1; s2 = 0;
        const double t = n1; n1 = n2; n2 = t;
    }
    for (int i = 2; i < 5; ++i) {
        const double n = hypot(fabs(result->mixing_5x5[3][i]), fabs(result->mixing_5x5[4][i]));
        if (n > n1) {
            s2 = s1; n2 = n1; s1 = i; n1 = n;
        } else if (n > n2) {
            s2 = i; n2 = n;
        }
    }
    if (s1 > s2) {
        const int t = s1; s1 = s2; s2 = t;
    }
    result->sterile_idx[0] = s1;
    result->sterile_idx[1] = s2;

    int aidx[3], fill = 0;
    for (int i = 0; i < 5; ++i) if (i != s1 && i != s2) aidx[fill++] = i;
    const double m1 = result->masses_eV[aidx[0]];
    const double m2 = result->masses_eV[aidx[1]];
    const double m3 = result->masses_eV[aidx[2]];
    const double m4 = result->masses_eV[s1];
    const double m5 = result->masses_eV[s2];
    result->dm21_eV2 = fabs(m2 * m2 - m1 * m1);
    result->dm31_eV2 = fabs(m3 * m3 - m1 * m1);
    result->dm41_eV2 = fabs(m4 * m4 - m1 * m1);
    result->dm51_eV2 = fabs(m5 * m5 - m1 * m1);

    result->theta14_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[0][s1]))) * 180.0 / M_PI;
    result->theta24_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[1][s1]))) * 180.0 / M_PI;
    result->theta34_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[2][s1]))) * 180.0 / M_PI;
    result->theta15_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[0][s2]))) * 180.0 / M_PI;
    result->theta25_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[1][s2]))) * 180.0 / M_PI;
    result->theta35_deg = asin(clamp_unit_iss24(fabs(result->mixing_5x5[2][s2]))) * 180.0 / M_PI;

    return 0;
}

static int pmns_pass_iss24(const Iss24Result *res, const SimulationConfig *cfg, double abs_u3[3][3]) {
    double ster[5];
    for (int i = 0; i < 5; ++i) ster[i] = hypot(fabs(res->mixing_5x5[3][i]), fabs(res->mixing_5x5[4][i]));
    int act[3] = {-1, -1, -1};
    for (int k = 0; k < 3; ++k) {
        int best = -1;
        double bestv = 1e100;
        for (int i = 0; i < 5; ++i) {
            int used = 0;
            for (int j = 0; j < k; ++j) if (act[j] == i) used = 1;
            if (used) continue;
            if (ster[i] < bestv) {
                bestv = ster[i];
                best = i;
            }
        }
        if (best < 0) return 0;
        act[k] = best;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double v = fabs(res->mixing_5x5[i][act[j]]);
            abs_u3[i][j] = v;
            if (v < cfg->inverse_pmns_abs_min_3x3[i][j] || v > cfg->inverse_pmns_abs_max_3x3[i][j]) return 0;
        }
    }
    return 1;
}

static int eta_pass_iss24(const Iss24Result *res, const SimulationConfig *cfg, double eta_abs_3x3[3][3]) {
    const double dm_sterile_ref_eV2 = fmin(res->dm41_eV2, res->dm51_eV2);
    const double (*eta4_max)[3] = cfg->inverse_eta_abs_max_nonunitarity_3x3;
    if (dm_sterile_ref_eV2 >= cfg->inverse_eta_dm41_low_min_eV2 && dm_sterile_ref_eV2 <= cfg->inverse_eta_dm41_low_max_eV2) {
        eta4_max = cfg->inverse_eta_abs_max_light_lowdm_3x3;
    } else if (dm_sterile_ref_eV2 >= cfg->inverse_eta_dm41_high_min_eV2) {
        eta4_max = cfg->inverse_eta_abs_max_light_highdm_3x3;
    }

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double eta4s = 0.0;
            eta4s += res->mixing_9x9[a][res->sterile_idx[0]] * res->mixing_9x9[b][res->sterile_idx[0]];
            eta4s += res->mixing_9x9[a][res->sterile_idx[1]] * res->mixing_9x9[b][res->sterile_idx[1]];
            const double eta4 = 0.5 * eta4s;
            eta_abs_3x3[a][b] = fabs(eta4);
            if (fabs(eta4) > eta4_max[a][b]) return 0;
        }
    }
    return 1;
}

static double br_mu_to_e_gamma_iss24(const Iss24Result *res) {
    const double alpha_em = 1.0 / 137.035999084;
    const double m_w_eV = 80.379e9;
    double amp = 0.0;
    for (int i = 0; i < 9; ++i) {
        const double x = (res->masses_full_eV[i] * res->masses_full_eV[i]) / (m_w_eV * m_w_eV);
        const double g = loop_function_g_gamma_iss24(x);
        amp += res->mixing_9x9[1][i] * res->mixing_9x9[0][i] * g;
    }
    return (3.0 * alpha_em / (32.0 * M_PI)) * amp * amp;
}

static void print_progress_bar_iss24(long long done, long long total, long long kept, long long eta_kept) {
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

int run_scan_inverse_pmns_filter_3p2(const SimulationConfig *cfg) {
    if (!cfg || cfg->output_inverse_pmns_filter_csv_path[0] == '\0') return 1;

    if (cfg->inverse_random_seed > 0) srand((unsigned int)cfg->inverse_random_seed);
    else srand((unsigned int)time(NULL));

    ensure_directory_exists("data");
    ensure_directory_exists(cfg->inverse_kept_points_dir);

    FILE *out = fopen(cfg->output_inverse_pmns_filter_csv_path, "w");
    if (!out) return 2;

    fprintf(out, "point_id,sample_id,dm21_eV2,dm31_eV2,dm41_eV2,dm51_eV2,theta14_deg,theta24_deg,theta34_deg,theta15_deg,theta25_deg,theta35_deg,br_muegamma,eta_pass\n");

    int next_id = 1;
    long long tested = 0;
    long long kept = 0;
    long long eta_kept = 0;
    int last_percent_printed = -1;

    const double mr_min_gev = cfg->inverse_random_MR_min_eV * 1e-9;
    const double mr_max_gev = cfg->inverse_random_MR_max_eV * 1e-9;
    const double md_min_gev = cfg->inverse_random_mD_min_eV * 1e-9;
    const double md_max_gev = cfg->inverse_random_mD_max_eV * 1e-9;

    for (int s = 0; s < cfg->inverse_random_samples; ++s) {
        Iss24Input in;
        Iss24Result res;
        ++tested;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                in.M_2x2_GeV[i][j] = uniform_random(mr_min_gev, mr_max_gev);
                in.mu_H0_2x2_eV[i][j] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
                in.mu00_2x2_eV[i][j] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
            }
        }
        in.mu_H_2x2_eV[0][0] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        in.mu_H_2x2_eV[1][1] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        in.mu_H_2x2_eV[0][1] = uniform_random(cfg->inverse_random_mu_min_eV, cfg->inverse_random_mu_max_eV);
        in.mu_H_2x2_eV[1][0] = in.mu_H_2x2_eV[0][1];
        in.mu00_2x2_eV[1][0] = in.mu00_2x2_eV[0][1];

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) in.mD_3x2_GeV[i][j] = uniform_random(md_min_gev, md_max_gev);
        }

        if (solve_iss24(&in, &res) != 0) {
            const long long done = (long long)s + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (int)((done * 100) / cfg->inverse_random_samples)
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar_iss24(done, cfg->inverse_random_samples, kept, eta_kept);
                last_percent_printed = percent;
            }
            continue;
        }

        double abs_u3[3][3];
        if (!pmns_pass_iss24(&res, cfg, abs_u3)) {
            const long long done = (long long)s + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (int)((done * 100) / cfg->inverse_random_samples)
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar_iss24(done, cfg->inverse_random_samples, kept, eta_kept);
                last_percent_printed = percent;
            }
            continue;
        }
        ++kept;

        double eta_abs_3x3[3][3] = {{0.0}};
        const int eta_pass = eta_pass_iss24(&res, cfg, eta_abs_3x3);
        if (eta_pass) ++eta_kept;
        const double br = br_mu_to_e_gamma_iss24(&res);

        fprintf(out, "%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d\n",
                next_id, s,
                res.dm21_eV2, res.dm31_eV2, res.dm41_eV2, res.dm51_eV2,
                res.theta14_deg, res.theta24_deg, res.theta34_deg,
                res.theta15_deg, res.theta25_deg, res.theta35_deg,
                br,
                eta_pass ? 1 : 0);

        {
            char pth[512];
            snprintf(pth, sizeof(pth), "%s/%d.txt", cfg->inverse_kept_points_dir, next_id);
            FILE *f = fopen(pth, "w");
            if (f) {
                fprintf(f, "=== METADATA ===\npoint_id = %d\neta_pass = %d\n\n", next_id, eta_pass ? 1 : 0);
                fprintf(f, "=== INPUT PARAMETERS ===\n");
                fprintf(f, "M_2x2_GeV = [%.10e, %.10e; %.10e, %.10e]\n", in.M_2x2_GeV[0][0], in.M_2x2_GeV[0][1], in.M_2x2_GeV[1][0], in.M_2x2_GeV[1][1]);
                fprintf(f, "mD_3x2_GeV = [%.10e, %.10e; %.10e, %.10e; %.10e, %.10e]\n", in.mD_3x2_GeV[0][0], in.mD_3x2_GeV[0][1], in.mD_3x2_GeV[1][0], in.mD_3x2_GeV[1][1], in.mD_3x2_GeV[2][0], in.mD_3x2_GeV[2][1]);
                fprintf(f, "mu_H_2x2_eV = [%.10e, %.10e; %.10e, %.10e]\n", in.mu_H_2x2_eV[0][0], in.mu_H_2x2_eV[0][1], in.mu_H_2x2_eV[1][0], in.mu_H_2x2_eV[1][1]);
                fprintf(f, "mu_H0_2x2_eV = [%.10e, %.10e; %.10e, %.10e]\n", in.mu_H0_2x2_eV[0][0], in.mu_H0_2x2_eV[0][1], in.mu_H0_2x2_eV[1][0], in.mu_H0_2x2_eV[1][1]);
                fprintf(f, "mu00_2x2_eV = [%.10e, %.10e; %.10e, %.10e]\n\n", in.mu00_2x2_eV[0][0], in.mu00_2x2_eV[0][1], in.mu00_2x2_eV[1][0], in.mu00_2x2_eV[1][1]);
                fprintf(f, "=== FILTER OBSERVABLES (EFFECTIVE 5x5) ===\n");
                fprintf(f, "masses_eV = [%.10e, %.10e, %.10e, %.10e, %.10e]\n", res.masses_eV[0], res.masses_eV[1], res.masses_eV[2], res.masses_eV[3], res.masses_eV[4]);
                fprintf(f, "dm21_eV2 = %.10e\n", res.dm21_eV2);
                fprintf(f, "dm31_eV2 = %.10e\n", res.dm31_eV2);
                fprintf(f, "dm41_eV2 = %.10e\n", res.dm41_eV2);
                fprintf(f, "dm51_eV2 = %.10e\n", res.dm51_eV2);
                fprintf(f, "theta14_deg = %.10e\n", res.theta14_deg);
                fprintf(f, "theta24_deg = %.10e\n", res.theta24_deg);
                fprintf(f, "theta34_deg = %.10e\n", res.theta34_deg);
                fprintf(f, "theta15_deg = %.10e\n", res.theta15_deg);
                fprintf(f, "theta25_deg = %.10e\n", res.theta25_deg);
                fprintf(f, "theta35_deg = %.10e\n", res.theta35_deg);
                fprintf(f, "sterile_state_indices = [%d, %d]\n", res.sterile_idx[0], res.sterile_idx[1]);
                fprintf(f, "PMNS_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n",
                        abs_u3[0][0], abs_u3[0][1], abs_u3[0][2], abs_u3[1][0], abs_u3[1][1], abs_u3[1][2], abs_u3[2][0], abs_u3[2][1], abs_u3[2][2]);
                fprintf(f, "eta_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n", eta_abs_3x3[0][0], eta_abs_3x3[0][1], eta_abs_3x3[0][2], eta_abs_3x3[1][0], eta_abs_3x3[1][1], eta_abs_3x3[1][2], eta_abs_3x3[2][0], eta_abs_3x3[2][1], eta_abs_3x3[2][2]);
                fprintf(f, "br_muegamma = %.10e\n\n", br);
                fprintf(f, "=== MATRICES ===\n");
                fprintf(f, "mixing_9x9 =\n");
                for (int r = 0; r < 9; ++r) {
                    fprintf(f, "  [%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]\n",
                            res.mixing_9x9[r][0], res.mixing_9x9[r][1], res.mixing_9x9[r][2], res.mixing_9x9[r][3], res.mixing_9x9[r][4], res.mixing_9x9[r][5], res.mixing_9x9[r][6], res.mixing_9x9[r][7], res.mixing_9x9[r][8]);
                }
                fclose(f);
            }
        }

        ++next_id;

        {
            const long long done = (long long)s + 1;
            const int percent = (cfg->inverse_random_samples > 0)
                                    ? (int)((done * 100) / cfg->inverse_random_samples)
                                    : 100;
            if (percent != last_percent_printed || done == cfg->inverse_random_samples) {
                print_progress_bar_iss24(done, cfg->inverse_random_samples, kept, eta_kept);
                last_percent_printed = percent;
            }
        }
    }

    fclose(out);
    printf("\n");
    printf("CSV genere (inverse PMNS filtre 3+2): %s\n", cfg->output_inverse_pmns_filter_csv_path);
    printf("Scan inverse PMNS 3+2: %lld tested, %lld kept, %lld eta_pass\n", tested, kept, eta_kept);
    return 0;
}

int run_scan_inverse_seesaw_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->output_inverse_csv_path[0] == '\0' || cfg->inverse_mu00_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->baseline_km <= 0.0) {
        return 1;
    }

    ensure_directory_exists("data");

    FILE *out = fopen(cfg->output_inverse_csv_path, "w");
    if (!out) {
        return 2;
    }

    fprintf(out,
            "energy_GeV,mu00_eV,dm21_eV2,dm31_eV2,dm41_eV2,theta14_deg,theta24_deg,theta34_deg,P_mumu_disappearance,P_mue_appearance\n");

    CasasIbarraInput3x2 ci_input;
    ci_input.m_light_eV[0] = cfg->inverse_ci_m_light_eV[0];
    ci_input.m_light_eV[1] = cfg->inverse_ci_m_light_eV[1];
    ci_input.m_light_eV[2] = cfg->inverse_ci_m_light_eV[2];
    ci_input.M_heavy_GeV[0] = cfg->inverse_ci_M_heavy_GeV[0];
    ci_input.M_heavy_GeV[1] = cfg->inverse_ci_M_heavy_GeV[1];
    ci_input.theta12_deg = cfg->theta12_deg;
    ci_input.theta13_deg = cfg->theta13_deg;
    ci_input.theta23_deg = cfg->theta23_deg;
    ci_input.delta_cp_deg = cfg->delta_cp_deg;
    ci_input.alpha21_deg = cfg->inverse_ci_alpha21_deg;
    ci_input.alpha31_deg = cfg->inverse_ci_alpha31_deg;
    ci_input.z_real = cfg->inverse_ci_z_real;
    ci_input.z_imag = cfg->inverse_ci_z_imag;
    ci_input.normal_ordering = cfg->inverse_ci_normal_ordering;

    double complex mD_casas_3x2[3][2];
    if (casas_ibarra_build_md_3x2(&ci_input, mD_casas_3x2) != 0) {
        fclose(out);
        return 4;
    }

    /* Export mD dans un CSV séparé si un chemin est fourni */
    if (cfg->output_inverse_md_csv_path[0] != '\0') {
        ensure_directory_exists("data");
        FILE *md_out = fopen(cfg->output_inverse_md_csv_path, "w");
        if (md_out) {
            fprintf(md_out, "row,col,re_GeV,im_GeV,abs_GeV\n");
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 2; ++j) {
                    fprintf(md_out, "%d,%d,%.10e,%.10e,%.10e\n",
                            i, j,
                            creal(mD_casas_3x2[i][j]),
                            cimag(mD_casas_3x2[i][j]),
                            cabs(mD_casas_3x2[i][j]));
                }
            }
            fclose(md_out);
            printf("CSV genere (matrice mD Casas-Ibarra): %s\n", cfg->output_inverse_md_csv_path);
        }
    }

    for (int idx_mu = 0; idx_mu < cfg->inverse_mu00_count; ++idx_mu) {
        InverseSeesaw3p1Input input;
        InverseSeesaw3p1Result result;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                input.mD_3x2_GeV[i][j] = cabs(mD_casas_3x2[i][j]);
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                input.M_2x2_GeV[i][j] = cfg->inverse_M_2x2_GeV[i][j];
                input.mu_H_2x2_eV[i][j] = cfg->inverse_mu_H_2x2_eV[i][j];
            }
            input.mu_H0_2x1_eV[i] = cfg->inverse_mu_H0_2x1_eV[i];
        }

        input.mu00_eV = cfg->inverse_mu00_values_eV[idx_mu];

        if (inverse_seesaw_solve_3p1(&input, &result) != 0) {
            fclose(out);
            return 3;
        }

        print_mixing_matrix_4x4(&result, input.mu00_eV);

        int ordered_mass_index[4] = {-1, -1, -1, -1};
        double mass_sq[4] = {0.0};
        build_ordered_mass_indices_3p1(&result, ordered_mass_index);
        for (int i = 0; i < 3; ++i) {
            mass_sq[i] = result.masses_eV[ordered_mass_index[i]] * result.masses_eV[ordered_mass_index[i]];
        }
        mass_sq[3] = result.masses_eV[ordered_mass_index[3]] * result.masses_eV[ordered_mass_index[3]];

        double complex u[4][4];
        for (int flavor = 0; flavor < 4; ++flavor) {
            for (int mass = 0; mass < 4; ++mass) {
                const int source_mass = ordered_mass_index[mass];
                u[flavor][mass] = result.mixing_4x4[flavor][source_mass] + 0.0 * I;
            }
        }

        for (double energy = cfg->energy_min_GeV; energy <= cfg->energy_max_GeV + 1e-12; energy += cfg->energy_step_GeV) {
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

            fprintf(out,
                    "%.6f,%.10g,%.10e,%.10e,%.10e,%.10g,%.10g,%.10g,%.10e,%.10e\n",
                    energy,
                    input.mu00_eV,
                    result.dm21_eV2,
                    result.dm31_eV2,
                    result.dm41_eV2,
                    result.theta14_deg,
                    result.theta24_deg,
                    result.theta34_deg,
                    p_mumu,
                    p_mue);
        }
    }

    fclose(out);
    printf("CSV genere (inverse seesaw 3+1): %s\n", cfg->output_inverse_csv_path);
    return 0;
}
