#include "scan.h"

#include "constants.h"
#include "inverse_seesaw/inverse_seesaw.h"
#include "inverse_seesaw/oscillation.h"
#include "inverse_seesaw/pmns.h"
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

#if defined(__GNUC__)
#define SCAN_UNUSED __attribute__((unused))
#else
#define SCAN_UNUSED
#endif

#ifndef GEV_TO_EV
#define GEV_TO_EV 1.0e9
#endif

#define CONSTRUCT24_FULL_DIM 9
#define CONSTRUCT24_LIGHT_DIM 5
#define CONSTRUCT24_TAKAGI_REAL_DIM (2 * CONSTRUCT24_FULL_DIM)

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

static double log_uniform_random(double min_value, double max_value) {
    if (min_value <= 0.0 || max_value <= 0.0 || max_value <= min_value) {
        return uniform_random(min_value, max_value);
    }
    const double log_min = log(min_value);
    const double log_max = log(max_value);
    return exp(uniform_random(log_min, log_max));
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

static double complex_phase_deg_construct23(double complex z) {
    const double re = creal(z);
    const double im = cimag(z);
    if (!isfinite(re) || !isfinite(im)) {
        return NAN;
    }
    double phase = atan2(im, re) * 180.0 / M_PI;
    if (phase < 0.0) {
        phase += 360.0;
    }
    return phase;
}


static void set_col_4x4(double U[4][4], int col, const double active[3], double bottom) {
    U[0][col] = active[0];
    U[1][col] = active[1];
    U[2][col] = active[2];
    U[3][col] = bottom;
}

static void set_col_4x4_complex(double complex U[4][4], int col, const double complex active[3], double complex bottom) {
    U[0][col] = active[0];
    U[1][col] = active[1];
    U[2][col] = active[2];
    U[3][col] = bottom;
}

static void SCAN_UNUSED build_u3p1_from_zeta_real(const double p1[3],
                                      const double p2[3],
                                      const double y[3],
                                      const double zeta[3],
                                      int k0,
                                      double out_u[4][4]) {
    const double s2 = zeta[0]*zeta[0] + zeta[1]*zeta[1] + zeta[2]*zeta[2];
    if (s2 >= 1.0) {
        fprintf(stderr, "ERROR: ||zeta||^2 >= 1 : %.16e\n", s2);
        exit(EXIT_FAILURE);
    }
    const double dot_zy = zeta[0]*y[0] + zeta[1]*y[1] + zeta[2]*y[2];
    if (fabs(dot_zy) > 1e-10) {
        fprintf(stderr, "ERROR: zeta is not orthogonal to y: %.16e\n", dot_zy);
        exit(EXIT_FAILURE);
    }
    const double s = sqrt(fmax(0.0, s2));
    const double c0 = sqrt(fmax(0.0, 1.0 - s2));
    double A[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    if (s > 1e-14) {
        const double w[3] = {zeta[0] / s, zeta[1] / s, zeta[2] / s};
        const double alpha = 1.0 - c0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A[i][j] -= alpha * w[i] * w[j];
            }
        }
    }
    double col_p1[3];
    double col_p2[3];
    double col_y[3] = {y[0], y[1], y[2]};
    double col_s[3] = {zeta[0], zeta[1], zeta[2]};
    mat3_vec_mul(A, p1, col_p1);
    mat3_vec_mul(A, p2, col_p2);
    const double bottom_p1 = -(zeta[0]*p1[0] + zeta[1]*p1[1] + zeta[2]*p1[2]);
    const double bottom_p2 = -(zeta[0]*p2[0] + zeta[1]*p2[1] + zeta[2]*p2[2]);
    const double bottom_y = 0.0;
    const double bottom_s = c0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out_u[i][j] = 0.0;
    /* columns 0,1,2 are nu1,nu2,nu3; column 3 is nu4. k0 is the massless active column. */
    set_col_4x4(out_u, k0, col_y, bottom_y);
    if (k0 == 0) {
        set_col_4x4(out_u, 1, col_p1, bottom_p1);
        set_col_4x4(out_u, 2, col_p2, bottom_p2);
    } else if (k0 == 2) {
        set_col_4x4(out_u, 0, col_p1, bottom_p1);
        set_col_4x4(out_u, 1, col_p2, bottom_p2);
    } else {
        fprintf(stderr, "ERROR: unsupported k0 = %d\n", k0);
        exit(EXIT_FAILURE);
    }
    set_col_4x4(out_u, 3, col_s, bottom_s);
}

static void build_u3p1_from_zeta_complex(const double p1[3],
                                         const double p2[3],
                                         const double y[3],
                                         const double complex zeta[3],
                                         int k0,
                                         double complex out_u[4][4]) {
    double s2 = 0.0;
    double complex dot_zy = 0.0;
    for (int i = 0; i < 3; ++i) {
        s2 += pow(cabs(zeta[i]), 2.0);
        dot_zy += conj(zeta[i]) * y[i];
    }
    if (s2 >= 1.0) {
        fprintf(stderr, "ERROR: ||zeta||^2 >= 1 : %.16e\n", s2);
        exit(EXIT_FAILURE);
    }
    if (cabs(dot_zy) > 1e-10) {
        fprintf(stderr, "ERROR: zeta is not orthogonal to y: %.16e\n", cabs(dot_zy));
        exit(EXIT_FAILURE);
    }

    double complex A[3][3] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        A[i][i] = 1.0;
    }
    const double c0 = sqrt(fmax(0.0, 1.0 - s2));
    if (s2 > 1e-28) {
        const double alpha = 1.0 - c0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A[i][j] -= alpha * zeta[i] * conj(zeta[j]) / s2;
            }
        }
    }

    double complex col_p1[3] = {0.0, 0.0, 0.0};
    double complex col_p2[3] = {0.0, 0.0, 0.0};
    double complex col_y[3] = {y[0], y[1], y[2]};
    double complex col_s[3] = {zeta[0], zeta[1], zeta[2]};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            col_p1[i] += A[i][j] * p1[j];
            col_p2[i] += A[i][j] * p2[j];
        }
    }

    double complex bottom_p1 = 0.0;
    double complex bottom_p2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        bottom_p1 -= conj(zeta[i]) * p1[i];
        bottom_p2 -= conj(zeta[i]) * p2[i];
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out_u[i][j] = 0.0;
        }
    }

    set_col_4x4_complex(out_u, k0, col_y, 0.0);
    if (k0 == 0) {
        set_col_4x4_complex(out_u, 1, col_p1, bottom_p1);
        set_col_4x4_complex(out_u, 2, col_p2, bottom_p2);
    } else if (k0 == 2) {
        set_col_4x4_complex(out_u, 0, col_p1, bottom_p1);
        set_col_4x4_complex(out_u, 1, col_p2, bottom_p2);
    } else {
        fprintf(stderr, "ERROR: unsupported k0 = %d\n", k0);
        exit(EXIT_FAILURE);
    }
    set_col_4x4_complex(out_u, 3, col_s, c0);
}

static int SCAN_UNUSED check_orthonormal_columns_4x4_real(const double u[4][4], double tol) {
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

static int check_unitary_columns_4x4_complex(const double complex u[4][4], double tol) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double complex dot = 0.0;
            for (int r = 0; r < 4; ++r) {
                dot += conj(u[r][i]) * u[r][j];
            }
            if (i == j) {
                if (fabs(creal(dot) - 1.0) + fabs(cimag(dot)) > tol) {
                    return 0;
                }
            } else if (cabs(dot) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

static int SCAN_UNUSED inverse_2x2_real(const double m[2][2], double inv[2][2]) {
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

static int inverse_2x2_complex_construct23(const double complex m[2][2], double complex inv[2][2]) {
    const double complex det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (cabs(det) < 1e-18) {
        return 1;
    }

    inv[0][0] = m[1][1] / det;
    inv[0][1] = -m[0][1] / det;
    inv[1][0] = -m[1][0] / det;
    inv[1][1] = m[0][0] / det;
    return 0;
}

static void singular_values_2x2_complex(
    const double complex a[2][2],
    double *sigma_min_out,
    double *sigma_max_out) {

    const double trace =
        pow(cabs(a[0][0]), 2.0) + pow(cabs(a[0][1]), 2.0) +
        pow(cabs(a[1][0]), 2.0) + pow(cabs(a[1][1]), 2.0);
    const double complex det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    const double det_b = pow(cabs(det_a), 2.0);
    const double disc = fmax(0.0, trace * trace - 4.0 * det_b);
    const double lambda_max = 0.5 * (trace + sqrt(disc));
    const double lambda_min = 0.5 * (trace - sqrt(disc));
    const double sigma_max = sqrt(fmax(0.0, lambda_max));
    const double sigma_min = sqrt(fmax(0.0, lambda_min));

    if (sigma_min_out) {
        *sigma_min_out = sigma_min;
    }
    if (sigma_max_out) {
        *sigma_max_out = sigma_max;
    }
}

static double condition_number_2x2_complex(const double complex a[2][2]) {
    double sigma_min = 0.0;
    double sigma_max = 0.0;
    singular_values_2x2_complex(a, &sigma_min, &sigma_max);

    if (sigma_min < 1e-18) {
        return 1e30;
    }
    return sigma_max / sigma_min;
}

static void SCAN_UNUSED mat3_mul(const double a[3][3], const double b[3][3], double out[3][3]) {
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

static void SCAN_UNUSED mat3_transpose(const double in[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = in[j][i];
        }
    }
}

static void mat3_mul_complex(const double complex a[3][3],
                             const double complex b[3][3],
                             double complex out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double complex s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
}

static void mat3_transpose_complex(const double complex in[3][3], double complex out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = in[j][i];
        }
    }
}

static double SCAN_UNUSED frob_norm_4x4(const double m[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            s += m[i][j] * m[i][j];
        }
    }
    return sqrt(s);
}

static double frob_norm_4x4_complex(const double complex m[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            s += pow(cabs(m[i][j]), 2.0);
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

static int solver_pmns_pass_construct23(
    const InverseSeesaw3p1Result *result,
    const SimulationConfig *cfg,
    const int ordered_mass_index[4],
    const double pmns_target_abs[3][3],
    double *rms_abs_error) {

    double rms = 0.0;
    int pass = 1;

    if (!result || !cfg || !ordered_mass_index || !pmns_target_abs) {
        if (rms_abs_error) {
            *rms_abs_error = NAN;
        }
        return 0;
    }

    for (int flavor = 0; flavor < 3; ++flavor) {
        for (int mass = 0; mass < 3; ++mass) {
            const int solver_col = ordered_mass_index[mass];
            if (solver_col < 0 || solver_col >= 4) {
                if (rms_abs_error) {
                    *rms_abs_error = NAN;
                }
                return 0;
            }

            const double v = fabs(result->mixing_4x4[flavor][solver_col]);
            const double diff = v - pmns_target_abs[flavor][mass];
            rms += diff * diff;
            if (v < cfg->inverse_pmns_abs_min_3x3[flavor][mass] ||
                v > cfg->inverse_pmns_abs_max_3x3[flavor][mass]) {
                pass = 0;
            }
        }
    }

    if (rms_abs_error) {
        *rms_abs_error = sqrt(rms / 9.0);
    }
    return pass;
}

static int eta_constraints_satisfied_3p1(const InverseSeesaw3p1Result *result,
                                         const SimulationConfig *cfg,
                                         double dm41_ref_eV2,
                                         const double complex zeta[3],
                                         double eta_abs_3x3[3][3]) {
    const double dm41_eV2 = dm41_ref_eV2;
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

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double complex eta4_sum = 0.0;
            double complex etaH_sum = 0.0;

            eta4_sum = zeta[a] * conj(zeta[b]);
            for (int heavy_col = 4; heavy_col < 8; ++heavy_col) {
                const double complex contrib =
                    result->mixing_8x8_complex[a][heavy_col] *
                    conj(result->mixing_8x8_complex[b][heavy_col]);
                etaH_sum += contrib;
            }

            {
                const double complex eta4 = 0.5 * eta4_sum;
                const double complex etaH = 0.5 * etaH_sum;
                const double complex eta_total = eta4 + etaH;
                const double eta4_abs = cabs(eta4);
                const double etaH_abs = cabs(etaH);

                eta_abs_3x3[a][b] = cabs(eta_total);

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

static int file_exists_construct23(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

static void clear_txt_files_in_dir_construct23(const char *dir_path) {
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

static int find_next_kept_point_index_in_dir_construct23(const char *dir_path) {
    int idx = 1;
    char path[512];

    while (1) {
        snprintf(path, sizeof(path), "%s/%d.txt", dir_path, idx);
        if (!file_exists_construct23(path)) {
            return idx;
        }
        ++idx;
    }
}

static void write_kept_points_csv_header_construct23(FILE *fout) {
    if (!fout) {
        return;
    }

    fprintf(fout,
            "point_id,sample_id,solve_ok,f_pass,pmns_pass,eta_pass,"
            "dm41_target_eV2,dm21_target_eV2,dm31_target_eV2,"
            "dm21_calc_eV2,dm31_calc_eV2,dm41_calc_eV2,"
            "pmns_rms_abs_error,mL_rel_frob_error,U4_abs_rms_error,"
            "zeta_norm,zeta_direction_deg,zeta_phase_deg,majorana_alpha21_deg,majorana_alpha31_deg,"
            "theta14_deg,theta24_deg,theta34_deg,delta_cp_sterile_deg,"
            "f11,f12,f21,f22,"
            "f11_phase_deg,f12_phase_deg,f21_phase_deg,f22_phase_deg,"
            "det_f,kappa_f,f_sigma_min,M1_GeV,M2_GeV,"
            "eta11_abs,eta12_abs,eta13_abs,eta21_abs,eta22_abs,eta23_abs,eta31_abs,eta32_abs,eta33_abs,"
            "mu3_11_eV,mu3_12_eV,mu3_13_eV,mu3_21_eV,mu3_22_eV,mu3_23_eV,mu3_31_eV,mu3_32_eV,mu3_33_eV,"
            "muH11_eV,muH12_eV,muH21_eV,muH22_eV,muH01_eV,muH02_eV,mu00_eV");

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout, ",E_constructed_%d%d", r + 1, c + 1);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout,
                    ",E_constructed_re_%d%d,E_constructed_im_%d%d,E_constructed_phase_deg_%d%d",
                    r + 1,
                    c + 1,
                    r + 1,
                    c + 1,
                    r + 1,
                    c + 1);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout, ",U_solver_%d%d", r + 1, c + 1);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout,
                    ",U_solver_re_%d%d,U_solver_im_%d%d,U_solver_phase_deg_%d%d",
                    r + 1,
                    c + 1,
                    r + 1,
                    c + 1,
                    r + 1,
                    c + 1);
        }
    }
    fprintf(fout, "\n");
}

static void append_kept_point_csv_construct23(
    FILE *fout,
    int point_id,
    int sample_id,
    int solve_ok,
    int f_pass,
    int pmns_pass,
    int eta_pass,
    double dm41_target,
    double dm21_target,
    double dm31_target,
    double dm21_calc,
    double dm31_calc,
    double dm41_calc,
    double zeta_norm,
    double zeta_direction_deg,
    double zeta_phase_deg,
    double alpha21_deg,
    double alpha31_deg,
    double theta14,
    double theta24,
    double theta34,
    double delta_cp_sterile,
    const double f[2][2],
    const double f_phase_deg[2][2],
    double det_f,
    double kappa_f,
    double f_sigma_min,
    double M1,
    double M2,
    double pmns_rms,
    double mL_rel_err,
    double u4_abs_rms_err,
    const double mu3[3][3],
    const double E_constructed[4][4],
    const double complex E_constructed_complex[4][4],
    const double U_solver[4][4],
    const double complex U_solver_complex[4][4],
    const double eta_abs_3x3[3][3]) {
    if (!fout) {
        return;
    }

    fprintf(fout,
            "%d,%d,%d,%d,%d,%d,"
            "%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e",
            point_id,
            sample_id,
            solve_ok ? 1 : 0,
            f_pass ? 1 : 0,
            pmns_pass ? 1 : 0,
            eta_pass ? 1 : 0,
            dm41_target,
            dm21_target,
            dm31_target,
            dm21_calc,
            dm31_calc,
            dm41_calc,
            pmns_rms,
            mL_rel_err,
            u4_abs_rms_err,
            zeta_norm,
            zeta_direction_deg,
            zeta_phase_deg,
            alpha21_deg,
            alpha31_deg,
            theta14,
            theta24,
            theta34,
            delta_cp_sterile,
            f[0][0],
            f[0][1],
            f[1][0],
            f[1][1],
            f_phase_deg[0][0],
            f_phase_deg[0][1],
            f_phase_deg[1][0],
            f_phase_deg[1][1],
            det_f,
            kappa_f,
            f_sigma_min,
            M1,
            M2,
            eta_abs_3x3[0][0],
            eta_abs_3x3[0][1],
            eta_abs_3x3[0][2],
            eta_abs_3x3[1][0],
            eta_abs_3x3[1][1],
            eta_abs_3x3[1][2],
            eta_abs_3x3[2][0],
            eta_abs_3x3[2][1],
            eta_abs_3x3[2][2],
            mu3[0][0],
            mu3[0][1],
            mu3[0][2],
            mu3[1][0],
            mu3[1][1],
            mu3[1][2],
            mu3[2][0],
            mu3[2][1],
            mu3[2][2],
            mu3[0][0],
            mu3[0][1],
            mu3[1][0],
            mu3[1][1],
            mu3[0][2],
            mu3[1][2],
            mu3[2][2]);

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout, ",%.10e", E_constructed[r][c]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const double complex z = E_constructed_complex[r][c];
            fprintf(fout,
                    ",%.10e,%.10e,%.10e",
                    creal(z),
                    cimag(z),
                    complex_phase_deg_construct23(z));
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fprintf(fout, ",%.10e", U_solver[r][c]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const double complex z = U_solver_complex[r][c];
            fprintf(fout,
                    ",%.10e,%.10e,%.10e",
                    creal(z),
                    cimag(z),
                    complex_phase_deg_construct23(z));
        }
    }
    fprintf(fout, "\n");
}

static void write_kept_point_details_construct23(
    int point_id,
    int sample_id,
    const char *kept_points_dir,
    int eta_pass,
    double dm41_target,
    double dm21_target,
    double dm31_target,
    double dm21_calc,
    double dm31_calc,
    double dm41_calc,
    double zeta_norm,
    double zeta_direction_deg,
    double zeta_phase_deg,
    double alpha21_deg,
    double alpha31_deg,
    double theta14,
    double theta24,
    double theta34,
    double delta_cp_sterile,
    const double f[2][2],
    const double f_phase_deg[2][2],
    double det_f,
    double kappa_f,
    double f_sigma_min,
    double M1,
    double M2,
    double pmns_rms,
    double mL_rel_err,
    double u4_abs_rms_err,
    const double mu3[3][3],
    const double E_constructed[4][4],
    const double complex E_constructed_complex[4][4],
    const double U_solver[4][4],
    const double complex U_solver_complex[4][4],
    const double eta_abs_3x3[3][3]) {

    char path[512];
    snprintf(path, sizeof(path), "%s/%d.txt", kept_points_dir, point_id);

    FILE *fout = fopen(path, "w");
    if (!fout) {
        return;
    }

    fprintf(fout, "=== METADATA ===\n");
    fprintf(fout, "point_id = %d\n", point_id);
    fprintf(fout, "sample_id = %d\n", sample_id);
    fprintf(fout, "solve_ok = 1\n");
    fprintf(fout, "f_pass = 1\n");
    fprintf(fout, "pmns_pass = 1\n");
    fprintf(fout, "eta_pass = %d\n\n", eta_pass ? 1 : 0);

    fprintf(fout, "=== TARGETS / RECONSTRUCTION ===\n");
    fprintf(fout, "dm41_target_eV2 = %.10e\n", dm41_target);
    fprintf(fout, "dm21_target_eV2 = %.10e\n", dm21_target);
    fprintf(fout, "dm31_target_eV2 = %.10e\n", dm31_target);
    fprintf(fout, "dm21_calc_eV2 = %.10e\n", dm21_calc);
    fprintf(fout, "dm31_calc_eV2 = %.10e\n", dm31_calc);
    fprintf(fout, "dm41_calc_eV2 = %.10e\n", dm41_calc);
    fprintf(fout, "pmns_rms_abs_error = %.10e\n", pmns_rms);
    fprintf(fout, "mL_rel_frob_error = %.10e\n\n", mL_rel_err);
    fprintf(fout, "U4_abs_rms_error = %.10e\n\n", u4_abs_rms_err);

    fprintf(fout, "=== STERILE MIXING ===\n");
    fprintf(fout, "zeta_norm = %.10e\n", zeta_norm);
    fprintf(fout, "zeta_direction_deg = %.10e\n", zeta_direction_deg);
    fprintf(fout, "zeta_phase_deg = %.10e\n", zeta_phase_deg);
    fprintf(fout, "majorana_alpha21_deg = %.10e\n", alpha21_deg);
    fprintf(fout, "majorana_alpha31_deg = %.10e\n", alpha31_deg);
    fprintf(fout, "theta14_deg = %.10e\n", theta14);
    fprintf(fout, "theta24_deg = %.10e\n", theta24);
    fprintf(fout, "theta34_deg = %.10e\n", theta34);
    fprintf(fout, "delta_cp_sterile_deg = %.10e\n\n", delta_cp_sterile);

    fprintf(fout, "=== FREE PARAMETERS f, M_R ===\n");
    fprintf(fout, "f11 = %.10e\n", f[0][0]);
    fprintf(fout, "f12 = %.10e\n", f[0][1]);
    fprintf(fout, "f21 = %.10e\n", f[1][0]);
    fprintf(fout, "f22 = %.10e\n", f[1][1]);
    fprintf(fout, "f11_phase_deg = %.10e\n", f_phase_deg[0][0]);
    fprintf(fout, "f12_phase_deg = %.10e\n", f_phase_deg[0][1]);
    fprintf(fout, "f21_phase_deg = %.10e\n", f_phase_deg[1][0]);
    fprintf(fout, "f22_phase_deg = %.10e\n", f_phase_deg[1][1]);
    fprintf(fout, "det_f = %.10e\n", det_f);
    fprintf(fout, "kappa_f = %.10e\n", kappa_f);
    fprintf(fout, "f_sigma_min = %.10e\n", f_sigma_min);
    fprintf(fout, "M1_GeV = %.10e\n", M1);
    fprintf(fout, "M2_GeV = %.10e\n\n", M2);

    fprintf(fout, "=== ETA ===\n");
    fprintf(fout,
            "eta_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n",
            eta_abs_3x3[0][0], eta_abs_3x3[0][1], eta_abs_3x3[0][2],
            eta_abs_3x3[1][0], eta_abs_3x3[1][1], eta_abs_3x3[1][2],
            eta_abs_3x3[2][0], eta_abs_3x3[2][1], eta_abs_3x3[2][2]);

        fprintf(fout, "\n=== MU MATRIX (BLOCK FORM) ===\n");
        fprintf(fout, "mu3_eV = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n",
            mu3[0][0], mu3[0][1], mu3[0][2],
            mu3[1][0], mu3[1][1], mu3[1][2],
            mu3[2][0], mu3[2][1], mu3[2][2]);
        fprintf(fout, "mu_H_2x2_eV = [%.10e, %.10e; %.10e, %.10e]\n",
            mu3[0][0], mu3[0][1],
            mu3[1][0], mu3[1][1]);
        fprintf(fout, "mu_H0_2x1_eV = [%.10e, %.10e]\n", mu3[0][2], mu3[1][2]);
        fprintf(fout, "mu00_eV = %.10e\n", mu3[2][2]);
        fprintf(fout, "mu_form = [[mu_H, mu_H0], [mu_H0^T, mu00]]\n");

        fprintf(fout, "\n=== 4x4 MIXING EXPRESSION ===\n");
        fprintf(fout, "U4x4_expression = R_zeta * blockdiag(U_nu, 1) * diag(e^{i alpha21/2}, e^{i alpha31/2}, 1, 1)\n");
        fprintf(fout, "majorana_alpha21_deg = %.10e\n", alpha21_deg);
        fprintf(fout, "majorana_alpha31_deg = %.10e\n", alpha31_deg);
        fprintf(fout, "U4x4_constructed =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            E_constructed[r][0], E_constructed[r][1], E_constructed[r][2], E_constructed[r][3]);
        }
        fprintf(fout, "U4x4_constructed_re =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            creal(E_constructed_complex[r][0]), creal(E_constructed_complex[r][1]),
            creal(E_constructed_complex[r][2]), creal(E_constructed_complex[r][3]));
        }
        fprintf(fout, "U4x4_constructed_im =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            cimag(E_constructed_complex[r][0]), cimag(E_constructed_complex[r][1]),
            cimag(E_constructed_complex[r][2]), cimag(E_constructed_complex[r][3]));
        }
        fprintf(fout, "U4x4_constructed_phase_deg =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            complex_phase_deg_construct23(E_constructed_complex[r][0]),
            complex_phase_deg_construct23(E_constructed_complex[r][1]),
            complex_phase_deg_construct23(E_constructed_complex[r][2]),
            complex_phase_deg_construct23(E_constructed_complex[r][3]));
        }
        fprintf(fout, "U4x4_solver =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            U_solver[r][0], U_solver[r][1], U_solver[r][2], U_solver[r][3]);
        }
        fprintf(fout, "U4x4_solver_re =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            creal(U_solver_complex[r][0]), creal(U_solver_complex[r][1]),
            creal(U_solver_complex[r][2]), creal(U_solver_complex[r][3]));
        }
        fprintf(fout, "U4x4_solver_im =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            cimag(U_solver_complex[r][0]), cimag(U_solver_complex[r][1]),
            cimag(U_solver_complex[r][2]), cimag(U_solver_complex[r][3]));
        }
        fprintf(fout, "U4x4_solver_phase_deg =\n");
        for (int r = 0; r < 4; ++r) {
        fprintf(fout, "  [%.10e, %.10e, %.10e, %.10e]\n",
            complex_phase_deg_construct23(U_solver_complex[r][0]),
            complex_phase_deg_construct23(U_solver_complex[r][1]),
            complex_phase_deg_construct23(U_solver_complex[r][2]),
            complex_phase_deg_construct23(U_solver_complex[r][3]));
        }

    fclose(fout);
}

int run_scan_inverse_construct_23_3p1(const SimulationConfig *cfg) {
    if (!cfg || cfg->inverse_construct_23_samples <= 0) {
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
    ensure_directory_exists(cfg->inverse_kept_points_dir);
    if (cfg->inverse_clear_kept_points_dir) {
        clear_txt_files_in_dir_construct23(cfg->inverse_kept_points_dir);
    }

    char kept_points_csv_path[512];
    if (cfg->output_inverse_construct_23_csv_path[0] != '\0') {
        snprintf(kept_points_csv_path,
                 sizeof(kept_points_csv_path),
                 "%s",
                 cfg->output_inverse_construct_23_csv_path);
    } else {
        snprintf(kept_points_csv_path,
                 sizeof(kept_points_csv_path),
                 "%s/inverse_construct_23_kept_points.csv",
                 cfg->inverse_kept_points_dir);
    }
    if (cfg->inverse_clear_kept_points_dir) {
        remove(kept_points_csv_path);
    }

    const int kept_points_csv_exists = file_exists_construct23(kept_points_csv_path);
    FILE *kept_points_csv = fopen(kept_points_csv_path, kept_points_csv_exists ? "a" : "w");
    if (!kept_points_csv) {
        fprintf(stderr, "Erreur: impossible d'ouvrir le CSV de sortie: %s\n", kept_points_csv_path);
        return 3;
    }
    setvbuf(kept_points_csv, NULL, _IOLBF, 0);
    if (kept_points_csv && !kept_points_csv_exists) {
        write_kept_points_csv_header_construct23(kept_points_csv);
        fflush(kept_points_csv);
    }

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
    int kept_pmns = 0;
    int kept_eta = 0;
    int next_point_id = cfg->inverse_clear_kept_points_dir
                            ? 1
                            : find_next_kept_point_index_in_dir_construct23(cfg->inverse_kept_points_dir);

    int last_progress = -1;

    for (int sample = 0; sample < cfg->inverse_construct_23_samples; ++sample) {
        int solve_ok = 0;
        int f_pass = 0;
        int pmns_pass = 0;
        int eta_pass = 0;
        double dm21_calc = NAN;
        double dm31_calc = NAN;
        double dm41_calc = NAN;
        double pmns_rms = NAN;
        double mL_rel_err = NAN;
        double u4_abs_rms_err = NAN;
        double zeta_norm = NAN;
        double phi = NAN;
        double zeta_phase_deg = NAN;
        double alpha21_deg = NAN;
        double alpha31_deg = NAN;
        double theta14 = NAN;
        double theta24 = NAN;
        double theta34 = NAN;
        double delta_cp_sterile = NAN;
        double det_f = NAN;
        double kappa_f = NAN;
        double f_sigma_min = NAN;
        double M1 = NAN;
        double M2 = NAN;
        double dm41 = NAN;
        double complex zeta[3] = {NAN, NAN, NAN};
        double f[2][2] = {{NAN, NAN}, {NAN, NAN}};
        double f_phase_deg[2][2] = {{NAN, NAN}, {NAN, NAN}};
        double mu3[3][3] = {
            {NAN, NAN, NAN},
            {NAN, NAN, NAN},
            {NAN, NAN, NAN}
        };
        double complex mu3_complex[3][3] = {
            {NAN, NAN, NAN},
            {NAN, NAN, NAN},
            {NAN, NAN, NAN}
        };
        double E[4][4] = {
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN}
        };
        double complex E_complex[4][4] = {
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN}
        };
        double U_solver_snapshot[4][4] = {
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN}
        };
        double complex U_solver_complex_snapshot[4][4] = {
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN},
            {NAN, NAN, NAN, NAN}
        };
        double eta_abs_3x3[3][3] = {
            {NAN, NAN, NAN},
            {NAN, NAN, NAN},
            {NAN, NAN, NAN}
        };

        int progress = (100 * (sample + 1)) / cfg->inverse_construct_23_samples;
        if (progress != last_progress && (progress % 2 == 0 || progress == 100)) {
            printf("\rProgress: %3d%% [%d/%d]", progress, sample + 1, cfg->inverse_construct_23_samples);
            fflush(stdout);
            last_progress = progress;
        }

        // Tirage des masses et angles
        dm41 = cfg->inverse_construct_23_dm41_logspace
                   ? log_uniform_random(cfg->inverse_construct_23_dm41_min_eV2, cfg->inverse_construct_23_dm41_max_eV2)
                   : uniform_random(cfg->inverse_construct_23_dm41_min_eV2, cfg->inverse_construct_23_dm41_max_eV2);
        const double m4 = sqrt(dm41);
        const double m2 = sqrt(dm21_target);
        const double m3 = sqrt(dm31_target);

        // Tirage de zeta
        zeta_norm = uniform_random(cfg->inverse_construct_23_zeta_norm_min, cfg->inverse_construct_23_zeta_norm_max);
        phi = uniform_random(
            deg_to_rad(cfg->inverse_construct_23_zeta_direction_min_deg),
            deg_to_rad(cfg->inverse_construct_23_zeta_direction_max_deg));
        zeta_phase_deg = uniform_random(
            cfg->inverse_construct_23_zeta_phase_min_deg,
            cfg->inverse_construct_23_zeta_phase_max_deg);
        const double zeta_phase = deg_to_rad(zeta_phase_deg);
        const double c1 = cos(phi);
        const double c2 = sin(phi);
        for (int i = 0; i < 3; ++i) {
            zeta[i] = zeta_norm * (c1 * p1[i] + cexp(I * zeta_phase) * c2 * p2[i]);
        }
        // Test orthogonalité zeta^dagger * y = 0
        double complex zeta_dot_y = 0.0;
        for (int i = 0; i < 3; ++i) {
            zeta_dot_y += conj(zeta[i]) * y[i];
        }
        (void)zeta_dot_y;

        alpha21_deg = uniform_random(cfg->inverse_construct_23_alpha21_min_deg, cfg->inverse_construct_23_alpha21_max_deg);
        alpha31_deg = uniform_random(cfg->inverse_construct_23_alpha31_min_deg, cfg->inverse_construct_23_alpha31_max_deg);

        const double ue4 = cabs(zeta[0]);
        const double umu4 = cabs(zeta[1]);
        const double utau4 = cabs(zeta[2]);
        const double c14 = sqrt(fmax(0.0, 1.0 - ue4 * ue4));
        const double sin24 = (c14 > 1e-14) ? umu4 / c14 : 0.0;
        const double c24 = sqrt(fmax(0.0, 1.0 - sin24 * sin24));
        const double denom34 = c14 * c24;
        const double sin34 = (denom34 > 1e-14) ? utau4 / denom34 : 0.0;

        theta14 = asin(clamp_unit_real(ue4)) * 180.0 / M_PI;
        theta24 = asin(clamp_unit_real(sin24)) * 180.0 / M_PI;
        theta34 = asin(clamp_unit_real(sin34)) * 180.0 / M_PI;
        delta_cp_sterile = zeta_phase_deg;

        // Construction de la matrice E
        int k0 = 0; // NH
        build_u3p1_from_zeta_complex(p1, p2, y, zeta, k0, E_complex);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                E[r][c] = cabs(E_complex[r][c]);
            }
        }
        // printf("[DEBUG] E après build_u3p1_from_zeta_real (sample=%d):\n", sample+1);
        // for (int r = 0; r < 4; ++r) {
        //     printf("  [");
        //     for (int c = 0; c < 4; ++c) {
        //         printf("% .6e%s", E[r][c], (c < 3 ? ", " : ""));
        //     }
        //     printf("]\n");
        // }

        int have_f = 0;
        for (int trial = 0; trial < 128; ++trial) {
            f[0][0] = uniform_random(cfg->inverse_construct_23_f11_min, cfg->inverse_construct_23_f11_max);
            f[0][1] = uniform_random(cfg->inverse_construct_23_f12_min, cfg->inverse_construct_23_f12_max);
            f[1][0] = uniform_random(cfg->inverse_construct_23_f21_min, cfg->inverse_construct_23_f21_max);
            f[1][1] = uniform_random(cfg->inverse_construct_23_f22_min, cfg->inverse_construct_23_f22_max);
            f_phase_deg[0][0] = uniform_random(cfg->inverse_construct_23_f11_phase_min_deg, cfg->inverse_construct_23_f11_phase_max_deg);
            f_phase_deg[0][1] = uniform_random(cfg->inverse_construct_23_f12_phase_min_deg, cfg->inverse_construct_23_f12_phase_max_deg);
            f_phase_deg[1][0] = uniform_random(cfg->inverse_construct_23_f21_phase_min_deg, cfg->inverse_construct_23_f21_phase_max_deg);
            f_phase_deg[1][1] = uniform_random(cfg->inverse_construct_23_f22_phase_min_deg, cfg->inverse_construct_23_f22_phase_max_deg);

            const double complex f11c = f[0][0] * cexp(I * deg_to_rad(f_phase_deg[0][0]));
            const double complex f12c = f[0][1] * cexp(I * deg_to_rad(f_phase_deg[0][1]));
            const double complex f21c = f[1][0] * cexp(I * deg_to_rad(f_phase_deg[1][0]));
            const double complex f22c = f[1][1] * cexp(I * deg_to_rad(f_phase_deg[1][1]));
            const double complex f_complex[2][2] = {
                {f11c, f12c},
                {f21c, f22c}
            };
            det_f = cabs(f11c * f22c - f12c * f21c);
            singular_values_2x2_complex(f_complex, &f_sigma_min, NULL);
            kappa_f = condition_number_2x2_complex(f_complex);
            f_pass =
                isfinite(det_f) &&
                isfinite(f_sigma_min) &&
                isfinite(kappa_f) &&
                fabs(det_f) >= cfg->inverse_construct_23_f_det_min_abs &&
                fabs(det_f) <= cfg->inverse_construct_23_f_det_max_abs &&
                f_sigma_min >= cfg->inverse_construct_23_f_sigma_min_min &&
                kappa_f <= cfg->inverse_construct_23_kappa_f_max;
            if (f_pass) {
                have_f = 1;
                break;
            }
        }
        if (!have_f) {
            continue;
        }


        M1 = uniform_random(cfg->inverse_construct_23_M1_min_GeV, cfg->inverse_construct_23_M1_max_GeV);
        M2 = uniform_random(cfg->inverse_construct_23_M2_min_GeV, cfg->inverse_construct_23_M2_max_GeV);

        const double complex f_complex_final[2][2] = {
            {
                f[0][0] * cexp(I * deg_to_rad(f_phase_deg[0][0])),
                f[0][1] * cexp(I * deg_to_rad(f_phase_deg[0][1]))
            },
            {
                f[1][0] * cexp(I * deg_to_rad(f_phase_deg[1][0])),
                f[1][1] * cexp(I * deg_to_rad(f_phase_deg[1][1]))
            }
        };

        double complex finv[2][2];
        if (inverse_2x2_complex_construct23(f_complex_final, finv) != 0) {
            goto append_sample_row;
        }

        double complex F[3][2] = {{0.0}};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                F[i][j] = p1[i] * f_complex_final[0][j] + p2[i] * f_complex_final[1][j];
            }
        }

        InverseSeesaw3p1Input input;
        memset(&input, 0, sizeof(input));

        input.M_2x2_GeV[0][0] = M1;
        input.M_2x2_GeV[1][1] = M2;
        input.M_2x2_GeV[0][1] = 0.0;
        input.M_2x2_GeV[1][0] = 0.0;
        input.use_complex = 1;

        for (int a = 0; a < 3; ++a) {
            input.mD_3x2_complex_GeV[a][0] = F[a][0] * M1;
            input.mD_3x2_complex_GeV[a][1] = F[a][1] * M2;
            input.mD_3x2_GeV[a][0] = creal(input.mD_3x2_complex_GeV[a][0]);
            input.mD_3x2_GeV[a][1] = creal(input.mD_3x2_complex_GeV[a][1]);
        }

        double complex U4[4][4];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) U4[i][j] = E_complex[i][j];
        int massless_col = k0; // convention: la colonne k0 est sans masse
        double complex U_r[4][3];
        for (int i = 0; i < 4; ++i) {
            int col = 0;
            for (int j = 0; j < 4; ++j) {
                if (j == massless_col) continue;
                U_r[i][col++] = U4[i][j];
            }
        }
        double complex m_r[3][3] = {{0.0}};
        m_r[0][0] = m2;
        m_r[1][1] = m3;
        m_r[2][2] = m4;

        /* M_L^target = U_r m_r U_r^T for a Majorana mass matrix. */
        double complex M_L_target[4][4] = {{0.0}};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double complex s = 0.0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        s += U_r[i][a] * m_r[a][b] * U_r[j][b];
                    }
                }
                M_L_target[i][j] = s;
            }
        }

        /*
         * Active 3x3 block in flavor basis, then rotate to the orthonormal
         * basis B = [p1 p2 y] before applying the construct_23 inversion.
         *
         * If M3 is not rotated with B^T M3 B, applying only f^{-1} mixes
         * incompatible bases and biases dm21/dm31 reconstruction.
         */
        double complex M3[3][3] = {{0.0}};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                M3[i][j] = M_L_target[i][j];
            }
        }

        double complex B[3][3] = {
            {p1[0], p2[0], y[0]},
            {p1[1], p2[1], y[1]},
            {p1[2], p2[2], y[2]}
        };

        double complex Bt[3][3], tmp_mb[3][3], M3_basis[3][3];
        mat3_transpose_complex(B, Bt);
        mat3_mul_complex(M3, B, tmp_mb);      // M3 * B
        mat3_mul_complex(Bt, tmp_mb, M3_basis); // B^T * M3 * B

        double complex Qinv[3][3] = {
            {finv[0][0], finv[0][1], 0.0},
            {finv[1][0], finv[1][1], 0.0},
            {0.0, 0.0, -1.0}
        };

        double complex QinvT[3][3], tmp3[3][3];
        mat3_transpose_complex(Qinv, QinvT);
        mat3_mul_complex(Qinv, M3_basis, tmp3);
        mat3_mul_complex(tmp3, QinvT, mu3_complex);

        /*
         * Derive mu_H0 and mu00 from the full 4x4 light block target:
         *   M_as = -F * mu_H0,  mu00 = (M_L)_44.
         * Keep mu_H from the active 3x3 reconstruction above.
         */
        {
            const double complex m_as[3] = {
                0.5 * (M_L_target[0][3] + M_L_target[3][0]),
                0.5 * (M_L_target[1][3] + M_L_target[3][1]),
                0.5 * (M_L_target[2][3] + M_L_target[3][2])
            };

            double complex gram[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            double complex rhs[2] = {0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
                gram[0][0] += conj(F[a][0]) * F[a][0];
                gram[0][1] += conj(F[a][0]) * F[a][1];
                gram[1][0] += conj(F[a][1]) * F[a][0];
                gram[1][1] += conj(F[a][1]) * F[a][1];

                rhs[0] += conj(F[a][0]) * m_as[a];
                rhs[1] += conj(F[a][1]) * m_as[a];
            }

            double complex gram_inv[2][2];
            if (inverse_2x2_complex_construct23(gram, gram_inv) != 0) {
                goto append_sample_row;
            }

            mu3_complex[0][2] = -(
                gram_inv[0][0] * rhs[0] +
                gram_inv[0][1] * rhs[1]
            );
            mu3_complex[1][2] = -(
                gram_inv[1][0] * rhs[0] +
                gram_inv[1][1] * rhs[1]
            );
            mu3_complex[2][0] = mu3_complex[0][2];
            mu3_complex[2][1] = mu3_complex[1][2];
            mu3_complex[2][2] = M_L_target[3][3];
        }

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                mu3[a][b] = cabs(mu3_complex[a][b]);
            }
        }

        input.mu_H_2x2_complex_eV[0][0] = mu3_complex[0][0];
        input.mu_H_2x2_complex_eV[0][1] = mu3_complex[0][1];
        input.mu_H_2x2_complex_eV[1][0] = mu3_complex[1][0];
        input.mu_H_2x2_complex_eV[1][1] = mu3_complex[1][1];
        input.mu_H0_2x1_complex_eV[0] = mu3_complex[0][2];
        input.mu_H0_2x1_complex_eV[1] = mu3_complex[1][2];
        input.mu00_complex_eV = mu3_complex[2][2];

        input.mu_H_2x2_eV[0][0] = creal(mu3_complex[0][0]);
        input.mu_H_2x2_eV[0][1] = creal(mu3_complex[0][1]);
        input.mu_H_2x2_eV[1][0] = creal(mu3_complex[1][0]);
        input.mu_H_2x2_eV[1][1] = creal(mu3_complex[1][1]);
        input.mu_H0_2x1_eV[0] = creal(mu3_complex[0][2]);
        input.mu_H0_2x1_eV[1] = creal(mu3_complex[1][2]);
        input.mu00_eV = creal(mu3_complex[2][2]);

        InverseSeesaw3p1Result result;
        const int solve_ret = inverse_seesaw_solve_3p1(&input, &result);
        if (solve_ret != 0) {
            goto append_sample_row;
        }

        solve_ok = 1;
        ++solved_ok;
        dm21_calc = result.dm21_eV2;
        dm31_calc = result.dm31_eV2;
        dm41_calc = result.dm41_eV2;

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                U_solver_snapshot[r][c] = result.mixing_4x4[r][c];
                U_solver_complex_snapshot[r][c] = result.mixing_4x4_complex[r][c];
            }
        }
        {
            double s = 0.0;
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    const double d = result.mixing_4x4[r][c] - E[r][c];
                    s += d * d;
                }
            }
            u4_abs_rms_err = sqrt(s / 16.0);
        }

        int ordered_mass_index[4] = {-1, -1, -1, -1};
        build_ordered_mass_indices_3p1(&result, ordered_mass_index);

        /* Build U_{3+1} = R_zeta * blockdiag(U_nu, 1), with massless column purely active */
        /* E already built at top of loop */

        {
            const double complex y_dot_zeta = conj(zeta[0]) * y[0] + conj(zeta[1]) * y[1] + conj(zeta[2]) * y[2];
            const double sterile_massless = cabs(E_complex[3][0]);
            int unitary_fail = !check_unitary_columns_4x4_complex(E_complex, 1e-8);
            if (cabs(y_dot_zeta) > 1e-9 || sterile_massless > 1e-9 || unitary_fail) {
                goto append_sample_row;
            }
        }

        double complex diff4[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                diff4[i][j] = result.m_light_4x4_complex_eV[i][j] - M_L_target[i][j];
            }
        }
        const double denom = fmax(frob_norm_4x4_complex(M_L_target), 1e-20);
        mL_rel_err = frob_norm_4x4_complex(diff4) / denom;
        eta_pass = eta_constraints_satisfied_3p1(&result, cfg, dm41, zeta, eta_abs_3x3);

        pmns_pass = solver_pmns_pass_construct23(
            &result,
            cfg,
            ordered_mass_index,
            pmns_target_abs,
            &pmns_rms);

        if (!pmns_pass) {
            goto append_sample_row;
        }

        ++kept_pmns;
        if (eta_pass) {
            ++kept_eta;
        }

        {
            const int point_id = next_point_id++;
            write_kept_point_details_construct23(
                point_id,
                sample + 1,
                cfg->inverse_kept_points_dir,
                eta_pass,
                dm41,
                dm21_target,
                dm31_target,
                result.dm21_eV2,
                result.dm31_eV2,
                result.dm41_eV2,
                zeta_norm,
                phi * 180.0 / M_PI,
                zeta_phase_deg,
                alpha21_deg,
                alpha31_deg,
                theta14,
                theta24,
                theta34,
                delta_cp_sterile,
                f,
                f_phase_deg,
                det_f,
                kappa_f,
                f_sigma_min,
                M1,
                M2,
                pmns_rms,
                mL_rel_err,
                u4_abs_rms_err,
                mu3,
                E,
                E_complex,
                result.mixing_4x4,
                result.mixing_4x4_complex,
                eta_abs_3x3);
        }

append_sample_row:
        append_kept_point_csv_construct23(
            kept_points_csv,
            sample + 1,
            sample + 1,
            solve_ok,
            f_pass,
            pmns_pass,
            eta_pass,
            dm41,
            dm21_target,
            dm31_target,
            dm21_calc,
            dm31_calc,
            dm41_calc,
            zeta_norm,
            phi * 180.0 / M_PI,
            zeta_phase_deg,
            alpha21_deg,
            alpha31_deg,
            theta14,
            theta24,
            theta34,
            delta_cp_sterile,
            f,
            f_phase_deg,
            det_f,
            kappa_f,
            f_sigma_min,
            M1,
            M2,
            pmns_rms,
            mL_rel_err,
            u4_abs_rms_err,
            mu3,
            E,
            E_complex,
            U_solver_snapshot,
            U_solver_complex_snapshot,
            eta_abs_3x3);
        fflush(kept_points_csv);
    }

    if (kept_points_csv) {
        fclose(kept_points_csv);
    }

    printf("Scan construct_23 3+1: solved=%d/%d, kept(PMNS)=%d, kept(PMNS+eta)=%d\n",
           solved_ok, cfg->inverse_construct_23_samples, kept_pmns, kept_eta);
    printf("Points conserves ecrits dans: %s\n", cfg->inverse_kept_points_dir);
    printf("CSV unique des points: %s\n", kept_points_csv_path);
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

static double br_mu_to_e_gamma_construct24(
    const double masses_full_eV[CONSTRUCT24_FULL_DIM],
    const double complex mixing_full[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM]) {

    const double alpha_em = 1.0 / 137.035999084;
    const double m_w_eV = 80.379e9;
    double complex amp = 0.0;

    for (int i = 0; i < CONSTRUCT24_FULL_DIM; ++i) {
        const double x = (masses_full_eV[i] * masses_full_eV[i]) / (m_w_eV * m_w_eV);
        const double g = loop_function_g_gamma_iss24(x);
        amp += conj(mixing_full[1][i]) * mixing_full[0][i] * g;
    }

    return (3.0 * alpha_em / (32.0 * M_PI)) * pow(cabs(amp), 2.0);
}

static void build_unitary2_construct24(double theta_deg,
                                       double alpha_deg,
                                       double beta_deg,
                                       double gamma_deg,
                                       double complex u[2][2]) {
    const double th = deg_to_rad(theta_deg);
    const double a = deg_to_rad(alpha_deg);
    const double b = deg_to_rad(beta_deg);
    const double g = deg_to_rad(gamma_deg);
    const double c = cos(th);
    const double s = sin(th);
    const double complex eg = cexp(I * g);
    u[0][0] = eg * c * cexp(I * a);
    u[0][1] = eg * s * cexp(I * b);
    u[1][0] = -eg * s * cexp(-I * b);
    u[1][1] = eg * c * cexp(-I * a);
}

static void mat2_mul_complex(const double complex a[2][2],
                             const double complex b[2][2],
                             double complex out[2][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
}

static void mat2_dagger_complex(const double complex in[2][2], double complex out[2][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            out[i][j] = conj(in[j][i]);
        }
    }
}

static void build_c_svd_construct24(double s1,
                                    double s2,
                                    const double complex v[2][2],
                                    const double complex w[2][2],
                                    double complex cmat[2][2]) {
    double complex sd[2][2] = {{s1, 0.0}, {0.0, s2}};
    double complex tmp[2][2];
    mat2_mul_complex(v, sd, tmp);
    mat2_mul_complex(tmp, w, cmat);
}

static void build_u3p2_from_c_construct24(const double p1[3],
                                          const double p2[3],
                                          const double y[3],
                                          const double complex cmat[2][2],
                                          const double complex v[2][2],
                                          const double complex w[2][2],
                                          double s1,
                                          double s2,
                                          double alpha21_deg,
                                          double alpha31_deg,
                                          double complex z[3][2],
                                          double complex u5[5][5]) {
    double complex sqrt_active_2[2][2];
    double complex sqrt_sterile_2[2][2];
    double complex vdag[2][2], wdag[2][2], tmp[2][2], d2[2][2];
    const double d1 = sqrt(fmax(0.0, 1.0 - s1 * s1));
    const double d2v = sqrt(fmax(0.0, 1.0 - s2 * s2));

    for (int a = 0; a < 3; ++a) {
        z[a][0] = p1[a] * cmat[0][0] + p2[a] * cmat[1][0];
        z[a][1] = p1[a] * cmat[0][1] + p2[a] * cmat[1][1];
    }

    mat2_dagger_complex(v, vdag);
    mat2_dagger_complex(w, wdag);
    d2[0][0] = d1; d2[0][1] = 0.0;
    d2[1][0] = 0.0; d2[1][1] = d2v;
    mat2_mul_complex(v, d2, tmp);
    mat2_mul_complex(tmp, vdag, sqrt_active_2);
    mat2_mul_complex(wdag, d2, tmp);
    mat2_mul_complex(tmp, w, sqrt_sterile_2);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            u5[i][j] = 0.0;
        }
    }

    double complex a3[3][3] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            a3[i][j] =
                y[i] * y[j] +
                (p1[i] * sqrt_active_2[0][0] + p2[i] * sqrt_active_2[1][0]) * p1[j] +
                (p1[i] * sqrt_active_2[0][1] + p2[i] * sqrt_active_2[1][1]) * p2[j];
        }
    }

    const double complex maj[3] = {
        1.0,
        cexp(I * 0.5 * deg_to_rad(alpha21_deg)),
        cexp(I * 0.5 * deg_to_rad(alpha31_deg))
    };
    const double active_basis[3][3] = {
        {y[0], p1[0], p2[0]},
        {y[1], p1[1], p2[1]},
        {y[2], p1[2], p2[2]}
    };

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double complex sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += a3[row][k] * active_basis[k][col];
            }
            u5[row][col] = sum * maj[col];
        }
        u5[row][3] = z[row][0];
        u5[row][4] = z[row][1];
    }

    for (int s = 0; s < 2; ++s) {
        for (int col = 0; col < 3; ++col) {
            double complex sum = 0.0;
            for (int a = 0; a < 3; ++a) {
                sum -= conj(z[a][s]) * active_basis[a][col];
            }
            u5[3 + s][col] = sum * maj[col];
        }
        for (int t = 0; t < 2; ++t) {
            u5[3 + s][3 + t] = sqrt_sterile_2[s][t];
        }
    }
}

static int pmns_pass_construct24(const double complex u5[5][5],
                                 const SimulationConfig *cfg,
                                 double *rms_abs_error) {
    double rms = 0.0;
    int pass = 1;
    for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 3; ++i) {
            const double v = cabs(u5[a][i]);
            const double target = 0.5 * (cfg->inverse_pmns_abs_min_3x3[a][i] + cfg->inverse_pmns_abs_max_3x3[a][i]);
            const double d = v - target;
            rms += d * d;
            if (v < cfg->inverse_pmns_abs_min_3x3[a][i] || v > cfg->inverse_pmns_abs_max_3x3[a][i]) {
                pass = 0;
            }
        }
    }
    if (rms_abs_error) {
        *rms_abs_error = sqrt(rms / 9.0);
    }
    return pass;
}

static int eta_pass_construct24(const SimulationConfig *cfg,
                                double dm41_eV2,
                                double dm51_eV2,
                                const double complex z[3][2],
                                double eta_abs_3x3[3][3]) {
    const double dm_ref = fmin(dm41_eV2, dm51_eV2);
    const double (*eta_max)[3] = cfg->inverse_eta_abs_max_nonunitarity_3x3;
    if (dm_ref >= cfg->inverse_eta_dm41_low_min_eV2 && dm_ref <= cfg->inverse_eta_dm41_low_max_eV2) {
        eta_max = cfg->inverse_eta_abs_max_light_lowdm_3x3;
    } else if (dm_ref >= cfg->inverse_eta_dm41_high_min_eV2) {
        eta_max = cfg->inverse_eta_abs_max_light_highdm_3x3;
    }
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double complex zz = z[a][0] * conj(z[b][0]) + z[a][1] * conj(z[b][1]);
            eta_abs_3x3[a][b] = 0.5 * cabs(zz);
            if (eta_abs_3x3[a][b] > eta_max[a][b]) {
                return 0;
            }
        }
    }
    return 1;
}

static void write_complex_columns_construct24(FILE *out, const char *prefix, int n) {
    for (int i = 0; i < n; ++i) {
        fprintf(out,
                ",%s%d_abs,%s%d_re,%s%d_im,%s%d_phase_deg",
                prefix, i + 1,
                prefix, i + 1,
                prefix, i + 1,
                prefix, i + 1);
    }
}

static void write_complex_values_construct24(FILE *out, const double complex *values, int n) {
    for (int i = 0; i < n; ++i) {
        fprintf(out,
                ",%.10e,%.10e,%.10e,%.10e",
                cabs(values[i]),
                creal(values[i]),
                cimag(values[i]),
                complex_phase_deg_construct23(values[i]));
    }
}

static void jacobi_symmetric_real_matrix_construct24(
    double a[CONSTRUCT24_TAKAGI_REAL_DIM][CONSTRUCT24_TAKAGI_REAL_DIM],
    double eigenvalues[CONSTRUCT24_TAKAGI_REAL_DIM],
    double eigenvectors[CONSTRUCT24_TAKAGI_REAL_DIM][CONSTRUCT24_TAKAGI_REAL_DIM]) {

    for (int i = 0; i < CONSTRUCT24_TAKAGI_REAL_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_TAKAGI_REAL_DIM; ++j) {
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int iter = 0; iter < 6000; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = fabs(a[p][q]);
        for (int i = 0; i < CONSTRUCT24_TAKAGI_REAL_DIM; ++i) {
            for (int j = i + 1; j < CONSTRUCT24_TAKAGI_REAL_DIM; ++j) {
                const double off = fabs(a[i][j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < 1e-18) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double phi = 0.5 * atan2(2.0 * apq, (aqq - app));
        const double c = cos(phi);
        const double s = sin(phi);

        for (int k = 0; k < CONSTRUCT24_TAKAGI_REAL_DIM; ++k) {
            if (k != p && k != q) {
                const double akp = a[k][p];
                const double akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (int k = 0; k < CONSTRUCT24_TAKAGI_REAL_DIM; ++k) {
            const double vip = eigenvectors[k][p];
            const double viq = eigenvectors[k][q];
            eigenvectors[k][p] = c * vip - s * viq;
            eigenvectors[k][q] = s * vip + c * viq;
        }
    }

    for (int i = 0; i < CONSTRUCT24_TAKAGI_REAL_DIM; ++i) {
        eigenvalues[i] = a[i][i];
    }
}

static void sort_takagi_pairs_by_mass_construct24(
    double masses[CONSTRUCT24_FULL_DIM],
    double complex vectors[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM]) {

    for (int i = 0; i < CONSTRUCT24_FULL_DIM - 1; ++i) {
        int best = i;
        double best_value = masses[i];
        for (int j = i + 1; j < CONSTRUCT24_FULL_DIM; ++j) {
            if (masses[j] < best_value) {
                best = j;
                best_value = masses[j];
            }
        }
        if (best != i) {
            const double mt = masses[i];
            masses[i] = masses[best];
            masses[best] = mt;
            for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
                const double complex vt = vectors[row][i];
                vectors[row][i] = vectors[row][best];
                vectors[row][best] = vt;
            }
        }
    }
}

static int takagi_complex_symmetric_9x9_construct24(
    const double complex input[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM],
    double masses[CONSTRUCT24_FULL_DIM],
    double complex vectors[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM]) {

    double scale = 0.0;
    for (int i = 0; i < CONSTRUCT24_FULL_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_FULL_DIM; ++j) {
            const double v = cabs(input[i][j]);
            if (v > scale) scale = v;
        }
    }
    if (scale <= 0.0) return 1;

    double k[CONSTRUCT24_TAKAGI_REAL_DIM][CONSTRUCT24_TAKAGI_REAL_DIM] = {{0.0}};
    double evals[CONSTRUCT24_TAKAGI_REAL_DIM] = {0.0};
    double evecs[CONSTRUCT24_TAKAGI_REAL_DIM][CONSTRUCT24_TAKAGI_REAL_DIM] = {{0.0}};

    for (int i = 0; i < CONSTRUCT24_FULL_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_FULL_DIM; ++j) {
            const double complex z = input[i][j] / scale;
            const double a = creal(z);
            const double b = cimag(z);
            k[i][j] = a;
            k[i][j + CONSTRUCT24_FULL_DIM] = b;
            k[i + CONSTRUCT24_FULL_DIM][j] = b;
            k[i + CONSTRUCT24_FULL_DIM][j + CONSTRUCT24_FULL_DIM] = -a;
        }
    }

    jacobi_symmetric_real_matrix_construct24(k, evals, evecs);

    int selected[CONSTRUCT24_TAKAGI_REAL_DIM] = {0};
    int n_selected = 0;
    for (int pick = 0; pick < CONSTRUCT24_FULL_DIM; ++pick) {
        int best = -1;
        double best_eval = 0.0;
        for (int i = 0; i < CONSTRUCT24_TAKAGI_REAL_DIM; ++i) {
            if (evals[i] <= 1e-22) continue;
            int used = 0;
            for (int u = 0; u < n_selected; ++u) {
                if (selected[u] == i) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;

            double complex candidate[CONSTRUCT24_FULL_DIM];
            double norm = 0.0;
            for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
                candidate[row] = evecs[row][i] + I * evecs[row + CONSTRUCT24_FULL_DIM][i];
                norm += pow(cabs(candidate[row]), 2.0);
            }
            norm = sqrt(norm);
            if (norm < 1e-30) continue;
            for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
                candidate[row] /= norm;
            }

            double max_overlap = 0.0;
            for (int prev = 0; prev < n_selected; ++prev) {
                double complex overlap = 0.0;
                for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
                    overlap += conj(vectors[row][prev]) * candidate[row];
                }
                if (cabs(overlap) > max_overlap) max_overlap = cabs(overlap);
            }
            if (max_overlap > 0.95) continue;

            if (best < 0 || evals[i] < best_eval) {
                best = i;
                best_eval = evals[i];
            }
        }

        if (best < 0) break;
        selected[n_selected++] = best;
        masses[pick] = best_eval * scale;
        for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
            vectors[row][pick] = evecs[row][best] + I * evecs[row + CONSTRUCT24_FULL_DIM][best];
        }
    }

    if (n_selected != CONSTRUCT24_FULL_DIM) return 2;

    for (int col = 0; col < CONSTRUCT24_FULL_DIM; ++col) {
        double norm = 0.0;
        for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
            norm += pow(cabs(vectors[row][col]), 2.0);
        }
        norm = sqrt(norm);
        if (norm < 1e-30) return 3;
        for (int row = 0; row < CONSTRUCT24_FULL_DIM; ++row) {
            vectors[row][col] /= norm;
        }
    }

    sort_takagi_pairs_by_mass_construct24(masses, vectors);
    return 0;
}

static void build_full_mass_matrix_9x9_construct24(
    const double complex F[3][2],
    double M1_GeV,
    double M2_GeV,
    const double complex muH[2][2],
    const double complex muH0[2][2],
    const double complex mu00[2][2],
    double complex full[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM]) {

    for (int i = 0; i < CONSTRUCT24_FULL_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_FULL_DIM; ++j) {
            full[i][j] = 0.0;
        }
    }

    const double M_GeV[2] = {M1_GeV, M2_GeV};
    for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
            const double complex mD_eV = F[a][i] * M_GeV[i] * GEV_TO_EV;
            full[a][5 + i] = mD_eV;
            full[5 + i][a] = mD_eV;
        }
    }

    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
            full[3 + s][3 + t] = mu00[s][t];
            full[3 + s][7 + t] = muH0[t][s];
            full[7 + t][3 + s] = muH0[t][s];
            full[7 + s][7 + t] = muH[s][t];
        }
    }

    full[5][7] = M1_GeV * GEV_TO_EV;
    full[7][5] = M1_GeV * GEV_TO_EV;
    full[6][8] = M2_GeV * GEV_TO_EV;
    full[8][6] = M2_GeV * GEV_TO_EV;
}

static void build_effective_light_from_blocks_construct24(
    const double complex F[3][2],
    const double complex muH[2][2],
    const double complex muH0[2][2],
    const double complex mu00[2][2],
    double complex eff[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM]) {

    for (int i = 0; i < CONSTRUCT24_LIGHT_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_LIGHT_DIM; ++j) {
            eff[i][j] = 0.0;
        }
    }

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    eff[a][b] += F[a][i] * muH[i][j] * F[b][j];
                }
            }
        }
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < 2; ++i) {
                eff[a][3 + s] -= F[a][i] * muH0[i][s];
            }
            eff[3 + s][a] = eff[a][3 + s];
        }
    }

    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
            eff[3 + s][3 + t] = mu00[s][t];
        }
    }
}

static double relative_frobenius_5x5_construct24(
    const double complex a[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM],
    const double complex b[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM]) {

    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < CONSTRUCT24_LIGHT_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_LIGHT_DIM; ++j) {
            num += pow(cabs(a[i][j] - b[i][j]), 2.0);
            den += pow(cabs(b[i][j]), 2.0);
        }
    }
    return sqrt(num / fmax(den, 1e-300));
}

static double u5_abs_rms_error_construct24(
    const double complex solver[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM],
    const double complex target[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM]) {

    double sum = 0.0;
    for (int i = 0; i < CONSTRUCT24_LIGHT_DIM; ++i) {
        for (int j = 0; j < CONSTRUCT24_LIGHT_DIM; ++j) {
            const double d = cabs(solver[i][j]) - cabs(target[i][j]);
            sum += d * d;
        }
    }
    return sqrt(sum / 25.0);
}

static void write_complex_matrix5_text_construct24(FILE *fout,
                                                   const char *name,
                                                   const double complex m[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM]) {
    fprintf(fout, "%s_abs =\n", name);
    for (int r = 0; r < CONSTRUCT24_LIGHT_DIM; ++r) {
        fprintf(fout, "  [");
        for (int c = 0; c < CONSTRUCT24_LIGHT_DIM; ++c) {
            fprintf(fout, "%s%.10e", c ? ", " : "", cabs(m[r][c]));
        }
        fprintf(fout, "]\n");
    }
    fprintf(fout, "%s_re =\n", name);
    for (int r = 0; r < CONSTRUCT24_LIGHT_DIM; ++r) {
        fprintf(fout, "  [");
        for (int c = 0; c < CONSTRUCT24_LIGHT_DIM; ++c) {
            fprintf(fout, "%s%.10e", c ? ", " : "", creal(m[r][c]));
        }
        fprintf(fout, "]\n");
    }
    fprintf(fout, "%s_im =\n", name);
    for (int r = 0; r < CONSTRUCT24_LIGHT_DIM; ++r) {
        fprintf(fout, "  [");
        for (int c = 0; c < CONSTRUCT24_LIGHT_DIM; ++c) {
            fprintf(fout, "%s%.10e", c ? ", " : "", cimag(m[r][c]));
        }
        fprintf(fout, "]\n");
    }
    fprintf(fout, "%s_phase_deg =\n", name);
    for (int r = 0; r < CONSTRUCT24_LIGHT_DIM; ++r) {
        fprintf(fout, "  [");
        for (int c = 0; c < CONSTRUCT24_LIGHT_DIM; ++c) {
            fprintf(fout, "%s%.10e", c ? ", " : "", complex_phase_deg_construct23(m[r][c]));
        }
        fprintf(fout, "]\n");
    }
}

static void write_kept_point_details_construct24(
    int point_id,
    int sample_id,
    const char *kept_points_dir,
    int eta_pass,
    int solve_ok,
    double dm21_target,
    double dm31_target,
    double dm41_target,
    double dm51_target,
    double dm21_calc,
    double dm31_calc,
    double dm41_calc,
    double dm51_calc,
    double s1,
    double s2,
    double v_angle,
    double w_angle,
    double va,
    double vb,
    double vg,
    double wa,
    double wb,
    double wg,
    double alpha21,
    double alpha31,
    double M1_GeV,
    double M2_GeV,
    double pmns_rms,
    double mL_rel_err,
    double u5_abs_rms,
    const double complex fmat[2][2],
    const double complex C[2][2],
    const double complex Z[3][2],
    const double complex muH[2][2],
    const double complex muH0[2][2],
    const double complex mu00[2][2],
    const double complex U5[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM],
    const double complex U5_solver[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM],
    double br_muegamma,
    int br_muegamma_pass,
    const double eta_abs_3x3[3][3]) {

    char path[512];
    snprintf(path, sizeof(path), "%s/%d.txt", kept_points_dir, point_id);
    FILE *fout = fopen(path, "w");
    if (!fout) return;

    fprintf(fout, "=== METADATA ===\n");
    fprintf(fout, "point_id = %d\n", point_id);
    fprintf(fout, "sample_id = %d\n", sample_id);
    fprintf(fout, "solve_ok = %d\n", solve_ok ? 1 : 0);
    fprintf(fout, "pmns_pass = 1\n");
    fprintf(fout, "eta_pass = %d\n", eta_pass ? 1 : 0);
    fprintf(fout, "br_muegamma = %.10e\n", br_muegamma);
    fprintf(fout, "br_muegamma_pass = %d\n\n", br_muegamma_pass ? 1 : 0);

    fprintf(fout, "=== TARGETS / FULL 9x9 DIAGONALIZATION ===\n");
    fprintf(fout, "dm21_target_eV2 = %.10e\n", dm21_target);
    fprintf(fout, "dm31_target_eV2 = %.10e\n", dm31_target);
    fprintf(fout, "dm41_target_eV2 = %.10e\n", dm41_target);
    fprintf(fout, "dm51_target_eV2 = %.10e\n", dm51_target);
    fprintf(fout, "dm21_calc_eV2 = %.10e\n", dm21_calc);
    fprintf(fout, "dm31_calc_eV2 = %.10e\n", dm31_calc);
    fprintf(fout, "dm41_calc_eV2 = %.10e\n", dm41_calc);
    fprintf(fout, "dm51_calc_eV2 = %.10e\n", dm51_calc);
    fprintf(fout, "pmns_rms_abs_error = %.10e\n", pmns_rms);
    fprintf(fout, "mL_rel_frob_error = %.10e\n", mL_rel_err);
    fprintf(fout, "U5_abs_rms_error = %.10e\n\n", u5_abs_rms);

    fprintf(fout, "=== C = V diag(s1,s2) W PARAMETERS ===\n");
    fprintf(fout, "s1 = %.10e\n", s1);
    fprintf(fout, "s2 = %.10e\n", s2);
    fprintf(fout, "V_angle_deg = %.10e\n", v_angle);
    fprintf(fout, "W_angle_deg = %.10e\n", w_angle);
    fprintf(fout, "V_alpha_deg = %.10e\n", va);
    fprintf(fout, "V_beta_deg = %.10e\n", vb);
    fprintf(fout, "V_gamma_deg = %.10e\n", vg);
    fprintf(fout, "W_alpha_deg = %.10e\n", wa);
    fprintf(fout, "W_beta_deg = %.10e\n", wb);
    fprintf(fout, "W_gamma_deg = %.10e\n", wg);
    fprintf(fout, "majorana_alpha21_deg = %.10e\n", alpha21);
    fprintf(fout, "majorana_alpha31_deg = %.10e\n\n", alpha31);

    fprintf(fout, "=== FREE PARAMETERS f, M_R ===\n");
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            fprintf(fout, "f%d%d_abs = %.10e\n", r + 1, c + 1, cabs(fmat[r][c]));
            fprintf(fout, "f%d%d_re = %.10e\n", r + 1, c + 1, creal(fmat[r][c]));
            fprintf(fout, "f%d%d_im = %.10e\n", r + 1, c + 1, cimag(fmat[r][c]));
            fprintf(fout, "f%d%d_phase_deg = %.10e\n", r + 1, c + 1, complex_phase_deg_construct23(fmat[r][c]));
        }
    }
    fprintf(fout, "M1_GeV = %.10e\n", M1_GeV);
    fprintf(fout, "M2_GeV = %.10e\n\n", M2_GeV);

    fprintf(fout, "=== ETA ===\n");
    fprintf(fout,
            "eta_abs_3x3 = [%.10e, %.10e, %.10e; %.10e, %.10e, %.10e; %.10e, %.10e, %.10e]\n\n",
            eta_abs_3x3[0][0], eta_abs_3x3[0][1], eta_abs_3x3[0][2],
            eta_abs_3x3[1][0], eta_abs_3x3[1][1], eta_abs_3x3[1][2],
            eta_abs_3x3[2][0], eta_abs_3x3[2][1], eta_abs_3x3[2][2]);

    fprintf(fout, "=== C MATRIX ===\n");
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            fprintf(fout, "C%d%d_abs = %.10e, C%d%d_re = %.10e, C%d%d_im = %.10e, C%d%d_phase_deg = %.10e\n",
                    r + 1, c + 1, cabs(C[r][c]),
                    r + 1, c + 1, creal(C[r][c]),
                    r + 1, c + 1, cimag(C[r][c]),
                    r + 1, c + 1, complex_phase_deg_construct23(C[r][c]));
        }
    }

    fprintf(fout, "\n=== Z ACTIVE-STERILE MATRIX ===\n");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {
            fprintf(fout, "Z%d%d_abs = %.10e, Z%d%d_re = %.10e, Z%d%d_im = %.10e, Z%d%d_phase_deg = %.10e\n",
                    r + 1, c + 1, cabs(Z[r][c]),
                    r + 1, c + 1, creal(Z[r][c]),
                    r + 1, c + 1, cimag(Z[r][c]),
                    r + 1, c + 1, complex_phase_deg_construct23(Z[r][c]));
        }
    }

    fprintf(fout, "\n=== MU MATRIX BLOCKS ===\n");
    fprintf(fout, "mu_H_2x2_abs = [%.10e, %.10e; %.10e, %.10e]\n",
            cabs(muH[0][0]), cabs(muH[0][1]), cabs(muH[1][0]), cabs(muH[1][1]));
    fprintf(fout, "mu_H0_2x2_abs = [%.10e, %.10e; %.10e, %.10e]\n",
            cabs(muH0[0][0]), cabs(muH0[0][1]), cabs(muH0[1][0]), cabs(muH0[1][1]));
    fprintf(fout, "mu00_2x2_abs = [%.10e, %.10e; %.10e, %.10e]\n\n",
            cabs(mu00[0][0]), cabs(mu00[0][1]), cabs(mu00[1][0]), cabs(mu00[1][1]));

    fprintf(fout, "=== 5x5 LIGHT MIXING ===\n");
    write_complex_matrix5_text_construct24(fout, "U5_constructed", U5);
    fprintf(fout, "\n");
    write_complex_matrix5_text_construct24(fout, "U5_solver", U5_solver);

    fclose(fout);
}

int run_scan_inverse_construct_24_3p2(const SimulationConfig *cfg) {
    if (!cfg || cfg->inverse_construct_24_samples <= 0) {
        return 1;
    }
    if (cfg->inverse_construct_24_seed > 0) srand((unsigned int)cfg->inverse_construct_24_seed);
    else srand((unsigned int)time(NULL));

    ensure_directory_exists("data");
    ensure_directory_exists(cfg->inverse_kept_points_dir);
    if (cfg->inverse_clear_kept_points_dir) {
        clear_txt_files_in_dir_construct23(cfg->inverse_kept_points_dir);
    }

    FILE *out = fopen(cfg->output_inverse_construct_24_csv_path, "w");
    if (!out) {
        return 2;
    }

    fprintf(out,
            "point_id,sample_id,pmns_pass,eta_pass,"
            "dm41_target_eV2,dm51_target_eV2,dm54_target_eV2,"
            "s1,s2,V_angle_deg,W_angle_deg,"
            "V_alpha_deg,V_beta_deg,V_gamma_deg,W_alpha_deg,W_beta_deg,W_gamma_deg,"
            "majorana_alpha21_deg,majorana_alpha31_deg,"
            "M1_GeV,M2_GeV,"
            "theta14_deg,theta24_deg,theta34_deg,theta15_deg,theta25_deg,theta35_deg,"
            "pmns_rms_abs_error,solve_ok,coherence_pass,br_muegamma,br_muegamma_pass,dm21_calc_eV2,dm31_calc_eV2,dm41_calc_eV2,dm51_calc_eV2,"
            "mL_rel_frob_error,U5_abs_rms_error,"
            "f11_abs,f12_abs,f21_abs,f22_abs,f11_phase_deg,f12_phase_deg,f21_phase_deg,f22_phase_deg,"
            "eta11_abs,eta12_abs,eta13_abs,eta21_abs,eta22_abs,eta23_abs,eta31_abs,eta32_abs,eta33_abs,"
            "muH11_abs,muH12_abs,muH21_abs,muH22_abs,"
            "muH0_11_abs,muH0_12_abs,muH0_21_abs,muH0_22_abs,"
            "mu00_11_abs,mu00_12_abs,mu00_21_abs,mu00_22_abs");
    write_complex_columns_construct24(out, "C", 4);
    write_complex_columns_construct24(out, "Z", 6);
    write_complex_columns_construct24(out, "U5", 25);
    write_complex_columns_construct24(out, "U5_solver", 25);
    fprintf(out, "\n");

    double y[3], p1[3], p2[3];
    build_basis_from_pmns_exact(cfg, y, p1, p2);

    int kept_pmns = 0;
    int kept_eta = 0;
    int kept_details = 0;
    int next_point_id = 1;
    int last_progress = -1;

    for (int sample = 0; sample < cfg->inverse_construct_24_samples; ++sample) {
        int progress = (100 * (sample + 1)) / cfg->inverse_construct_24_samples;
        if (progress != last_progress && (progress % 2 == 0 || progress == 100)) {
            printf("\rProgress: %3d%% [%d/%d]", progress, sample + 1, cfg->inverse_construct_24_samples);
            fflush(stdout);
            last_progress = progress;
        }

        double dm41 = cfg->inverse_construct_24_dm41_logspace
                          ? log_uniform_random(cfg->inverse_construct_24_dm41_min_eV2, cfg->inverse_construct_24_dm41_max_eV2)
                          : uniform_random(cfg->inverse_construct_24_dm41_min_eV2, cfg->inverse_construct_24_dm41_max_eV2);
        double dm51 = cfg->inverse_construct_24_dm51_logspace
                          ? log_uniform_random(cfg->inverse_construct_24_dm51_min_eV2, cfg->inverse_construct_24_dm51_max_eV2)
                          : uniform_random(cfg->inverse_construct_24_dm51_min_eV2, cfg->inverse_construct_24_dm51_max_eV2);
        if (dm51 < dm41) {
            const double t = dm41; dm41 = dm51; dm51 = t;
        }
        const double dm54 = dm51 - dm41;
        const double masses[5] = {
            0.0,
            sqrt(cfg->dm21_eV2),
            sqrt(cfg->dm31_eV2),
            sqrt(dm41),
            sqrt(dm51)
        };

        const double s1 = uniform_random(cfg->inverse_construct_24_s1_min, cfg->inverse_construct_24_s1_max);
        const double s2 = uniform_random(cfg->inverse_construct_24_s2_min, cfg->inverse_construct_24_s2_max);
        const double v_angle = uniform_random(cfg->inverse_construct_24_v_angle_min_deg, cfg->inverse_construct_24_v_angle_max_deg);
        const double w_angle = uniform_random(cfg->inverse_construct_24_w_angle_min_deg, cfg->inverse_construct_24_w_angle_max_deg);
        const double va = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double vb = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double vg = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double wa = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double wb = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double wg = uniform_random(cfg->inverse_construct_24_phase_min_deg, cfg->inverse_construct_24_phase_max_deg);
        const double alpha21 = uniform_random(cfg->inverse_construct_24_alpha21_min_deg, cfg->inverse_construct_24_alpha21_max_deg);
        const double alpha31 = uniform_random(cfg->inverse_construct_24_alpha31_min_deg, cfg->inverse_construct_24_alpha31_max_deg);
        const double M1_GeV = uniform_random(cfg->inverse_construct_23_M1_min_GeV, cfg->inverse_construct_23_M1_max_GeV);
        const double M2_GeV = uniform_random(cfg->inverse_construct_23_M2_min_GeV, cfg->inverse_construct_23_M2_max_GeV);

        double complex V[2][2], W[2][2], C[2][2], Z[3][2], U5[5][5];
        build_unitary2_construct24(v_angle, va, vb, vg, V);
        build_unitary2_construct24(w_angle, wa, wb, wg, W);
        build_c_svd_construct24(s1, s2, V, W, C);
        build_u3p2_from_c_construct24(p1, p2, y, C, V, W, s1, s2, alpha21, alpha31, Z, U5);

        double pmns_rms = NAN;
        const int pmns_pass = pmns_pass_construct24(U5, cfg, &pmns_rms);
        double eta_abs[3][3] = {{0.0}};
        const int eta_pass = eta_pass_construct24(cfg, dm41, dm51, Z, eta_abs);
        if (pmns_pass) ++kept_pmns;
        if (pmns_pass && eta_pass) ++kept_eta;

        double complex Mlight[5][5] = {{0.0}};
        for (int a = 0; a < 5; ++a) {
            for (int b = 0; b < 5; ++b) {
                for (int k = 0; k < 5; ++k) {
                    Mlight[a][b] += U5[a][k] * masses[k] * U5[b][k];
                }
            }
        }

        double complex fmat[2][2];
        for (int trial = 0; trial < 128; ++trial) {
            fmat[0][0] = uniform_random(cfg->inverse_construct_23_f11_min, cfg->inverse_construct_23_f11_max) *
                         cexp(I * deg_to_rad(uniform_random(cfg->inverse_construct_23_f11_phase_min_deg, cfg->inverse_construct_23_f11_phase_max_deg)));
            fmat[0][1] = uniform_random(cfg->inverse_construct_23_f12_min, cfg->inverse_construct_23_f12_max) *
                         cexp(I * deg_to_rad(uniform_random(cfg->inverse_construct_23_f12_phase_min_deg, cfg->inverse_construct_23_f12_phase_max_deg)));
            fmat[1][0] = uniform_random(cfg->inverse_construct_23_f21_min, cfg->inverse_construct_23_f21_max) *
                         cexp(I * deg_to_rad(uniform_random(cfg->inverse_construct_23_f21_phase_min_deg, cfg->inverse_construct_23_f21_phase_max_deg)));
            fmat[1][1] = uniform_random(cfg->inverse_construct_23_f22_min, cfg->inverse_construct_23_f22_max) *
                         cexp(I * deg_to_rad(uniform_random(cfg->inverse_construct_23_f22_phase_min_deg, cfg->inverse_construct_23_f22_phase_max_deg)));
            double sigma_min = 0.0;
            singular_values_2x2_complex(fmat, &sigma_min, NULL);
            if (cabs(fmat[0][0] * fmat[1][1] - fmat[0][1] * fmat[1][0]) >= cfg->inverse_construct_23_f_det_min_abs &&
                cabs(fmat[0][0] * fmat[1][1] - fmat[0][1] * fmat[1][0]) <= cfg->inverse_construct_23_f_det_max_abs &&
                sigma_min >= cfg->inverse_construct_23_f_sigma_min_min &&
                condition_number_2x2_complex(fmat) <= cfg->inverse_construct_23_kappa_f_max) {
                break;
            }
        }

        double complex finv[2][2], finvT[2][2];
        double complex muH[2][2] = {{NAN}}, muH0[2][2] = {{NAN}}, mu00[2][2] = {{0.0}};
        double complex F[3][2] = {{0.0}};
        if (inverse_2x2_complex_construct23(fmat, finv) == 0) {
            double complex B[3][3] = {{p1[0], p2[0], y[0]}, {p1[1], p2[1], y[1]}, {p1[2], p2[2], y[2]}};
            double complex M3[3][3], Bt[3][3], tmp3[3][3], Mbasis[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M3[i][j] = Mlight[i][j];
            mat3_transpose_complex(B, Bt);
            mat3_mul_complex(M3, B, tmp3);
            mat3_mul_complex(Bt, tmp3, Mbasis);
            finvT[0][0] = finv[0][0]; finvT[0][1] = finv[1][0];
            finvT[1][0] = finv[0][1]; finvT[1][1] = finv[1][1];
            double complex top2[2][2] = {{Mbasis[0][0], Mbasis[0][1]}, {Mbasis[1][0], Mbasis[1][1]}};
            double complex tmp2[2][2];
            mat2_mul_complex(finv, top2, tmp2);
            mat2_mul_complex(tmp2, finvT, muH);

            for (int i = 0; i < 3; ++i) {
                F[i][0] = p1[i] * fmat[0][0] + p2[i] * fmat[1][0];
                F[i][1] = p1[i] * fmat[0][1] + p2[i] * fmat[1][1];
            }
            double complex gram[2][2] = {{0.0}}, gram_inv[2][2], rhs[2][2] = {{0.0}};
            for (int a = 0; a < 3; ++a) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        gram[i][j] += conj(F[a][i]) * F[a][j];
                    }
                    for (int s = 0; s < 2; ++s) {
                        rhs[i][s] += conj(F[a][i]) * Mlight[a][3 + s];
                    }
                }
            }
            if (inverse_2x2_complex_construct23(gram, gram_inv) == 0) {
                mat2_mul_complex(gram_inv, rhs, muH0);
                for (int i = 0; i < 2; ++i) {
                    for (int s = 0; s < 2; ++s) {
                        muH0[i][s] = -muH0[i][s];
                    }
                }
            }
        }
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) mu00[i][j] = Mlight[3 + i][3 + j];

        double complex Meff_check[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM];
        build_effective_light_from_blocks_construct24(F, muH, muH0, mu00, Meff_check);
        const double mL_rel_err = relative_frobenius_5x5_construct24(Meff_check, Mlight);

        double complex full9[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM];
        double full_masses[CONSTRUCT24_FULL_DIM] = {NAN};
        double complex full_vectors[CONSTRUCT24_FULL_DIM][CONSTRUCT24_FULL_DIM] = {{0.0}};
        double complex U5_solver[CONSTRUCT24_LIGHT_DIM][CONSTRUCT24_LIGHT_DIM] = {{0.0}};
        build_full_mass_matrix_9x9_construct24(F, M1_GeV, M2_GeV, muH, muH0, mu00, full9);
        const int solve_ok = (takagi_complex_symmetric_9x9_construct24(full9, full_masses, full_vectors) == 0);
        double dm21_calc = NAN;
        double dm31_calc = NAN;
        double dm41_calc = NAN;
        double dm51_calc = NAN;
        double u5_abs_rms = NAN;
        if (solve_ok) {
            for (int r = 0; r < CONSTRUCT24_LIGHT_DIM; ++r) {
                for (int c = 0; c < CONSTRUCT24_LIGHT_DIM; ++c) {
                    U5_solver[r][c] = full_vectors[r][c];
                }
            }
            const double m1sq = full_masses[0] * full_masses[0];
            dm21_calc = full_masses[1] * full_masses[1] - m1sq;
            dm31_calc = full_masses[2] * full_masses[2] - m1sq;
            dm41_calc = full_masses[3] * full_masses[3] - m1sq;
            dm51_calc = full_masses[4] * full_masses[4] - m1sq;
            u5_abs_rms = u5_abs_rms_error_construct24(U5_solver, U5);
        }
        const double dm21_rel_err = solve_ok ? fabs(dm21_calc - cfg->dm21_eV2) / fmax(cfg->dm21_eV2, 1e-300) : INFINITY;
        const double dm31_rel_err = solve_ok ? fabs(dm31_calc - cfg->dm31_eV2) / fmax(cfg->dm31_eV2, 1e-300) : INFINITY;
        const double dm41_rel_err = solve_ok ? fabs(dm41_calc - dm41) / fmax(dm41, 1e-300) : INFINITY;
        const double dm51_rel_err = solve_ok ? fabs(dm51_calc - dm51) / fmax(dm51, 1e-300) : INFINITY;
        const int coherent_9x9 =
            solve_ok &&
            isfinite(u5_abs_rms) && u5_abs_rms < 5.0e-2 &&
            dm21_rel_err < 2.0e-2 &&
            dm31_rel_err < 2.0e-2 &&
            dm41_rel_err < 2.0e-2 &&
            dm51_rel_err < 2.0e-2;
        const double br_muegamma = solve_ok ? br_mu_to_e_gamma_construct24(full_masses, full_vectors) : NAN;
        const int br_muegamma_pass = isfinite(br_muegamma) && br_muegamma <= cfg->inverse_br_muegamma_max;

        const int point_id = next_point_id++;
        fprintf(out,
                "%d,%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%d,%d,%.10e,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e,"
                "%.10e,%.10e,%.10e,%.10e",
                point_id, sample + 1, pmns_pass ? 1 : 0, eta_pass ? 1 : 0,
                dm41, dm51, dm54, s1, s2, v_angle, w_angle,
                va, vb, vg, wa, wb, wg, alpha21, alpha31,
                M1_GeV, M2_GeV,
                asin(clamp_unit_iss24(cabs(U5[0][3]))) * 180.0 / M_PI,
                asin(clamp_unit_iss24(cabs(U5[1][3]))) * 180.0 / M_PI,
                asin(clamp_unit_iss24(cabs(U5[2][3]))) * 180.0 / M_PI,
                asin(clamp_unit_iss24(cabs(U5[0][4]))) * 180.0 / M_PI,
                asin(clamp_unit_iss24(cabs(U5[1][4]))) * 180.0 / M_PI,
                asin(clamp_unit_iss24(cabs(U5[2][4]))) * 180.0 / M_PI,
                pmns_rms,
                solve_ok ? 1 : 0, coherent_9x9 ? 1 : 0, br_muegamma, br_muegamma_pass ? 1 : 0,
                dm21_calc, dm31_calc, dm41_calc, dm51_calc,
                mL_rel_err, u5_abs_rms,
                cabs(fmat[0][0]), cabs(fmat[0][1]), cabs(fmat[1][0]), cabs(fmat[1][1]),
                complex_phase_deg_construct23(fmat[0][0]),
                complex_phase_deg_construct23(fmat[0][1]),
                complex_phase_deg_construct23(fmat[1][0]),
                complex_phase_deg_construct23(fmat[1][1]),
                eta_abs[0][0], eta_abs[0][1], eta_abs[0][2],
                eta_abs[1][0], eta_abs[1][1], eta_abs[1][2],
                eta_abs[2][0], eta_abs[2][1], eta_abs[2][2],
                cabs(muH[0][0]), cabs(muH[0][1]), cabs(muH[1][0]), cabs(muH[1][1]),
                cabs(muH0[0][0]), cabs(muH0[0][1]), cabs(muH0[1][0]), cabs(muH0[1][1]),
                cabs(mu00[0][0]), cabs(mu00[0][1]), cabs(mu00[1][0]), cabs(mu00[1][1]));
        write_complex_values_construct24(out, &C[0][0], 4);
        write_complex_values_construct24(out, &Z[0][0], 6);
        write_complex_values_construct24(out, &U5[0][0], 25);
        write_complex_values_construct24(out, &U5_solver[0][0], 25);
        fprintf(out, "\n");

        if (pmns_pass && coherent_9x9) {
            ++kept_details;
            write_kept_point_details_construct24(
                point_id,
                sample + 1,
                cfg->inverse_kept_points_dir,
                eta_pass,
                solve_ok,
                cfg->dm21_eV2,
                cfg->dm31_eV2,
                dm41,
                dm51,
                dm21_calc,
                dm31_calc,
                dm41_calc,
                dm51_calc,
                s1,
                s2,
                v_angle,
                w_angle,
                va,
                vb,
                vg,
                wa,
                wb,
                wg,
                alpha21,
                alpha31,
                M1_GeV,
                M2_GeV,
                pmns_rms,
                mL_rel_err,
                u5_abs_rms,
                fmat,
                C,
                Z,
                muH,
                muH0,
                mu00,
                U5,
                U5_solver,
                br_muegamma,
                br_muegamma_pass,
                eta_abs);
        }
    }

    fclose(out);
    printf("\nScan construct_24 3+2: samples=%d, kept(PMNS)=%d, kept(PMNS+eta)=%d, detailed_9x9=%d\n",
           cfg->inverse_construct_24_samples, kept_pmns, kept_eta, kept_details);
    printf("CSV unique des points: %s\n", cfg->output_inverse_construct_24_csv_path);
    return 0;
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
