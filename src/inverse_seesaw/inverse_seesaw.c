#include "inverse_seesaw/inverse_seesaw.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ISS_LIGHT_DIM 4
#define ISS_HEAVY_DIM 4
#define ISS_FULL_DIM 8
#define ISS_TAKAGI_REAL_DIM (2 * ISS_FULL_DIM)
#define GEV_TO_EV 1.0e9

#if defined(__GNUC__)
#define ISS_UNUSED __attribute__((unused))
#else
#define ISS_UNUSED
#endif

static double clamp_unit(double value) {
    if (value > 1.0) return 1.0;
    if (value < -1.0) return -1.0;
    return value;
}

static int inverse_2x2(const double matrix[2][2], double out[2][2]) {
    const double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if (fabs(det) < 1e-30) {
        return 1;
    }

    out[0][0] = matrix[1][1] / det;
    out[0][1] = -matrix[0][1] / det;
    out[1][0] = -matrix[1][0] / det;
    out[1][1] = matrix[0][0] / det;
    return 0;
}

static double complex input_mD_value(const InverseSeesaw3p1Input *input, int row, int col) {
    return input->use_complex ? input->mD_3x2_complex_GeV[row][col] : input->mD_3x2_GeV[row][col];
}

static double complex input_muH_value(const InverseSeesaw3p1Input *input, int row, int col) {
    return input->use_complex ? input->mu_H_2x2_complex_eV[row][col] : input->mu_H_2x2_eV[row][col];
}

static double complex input_muH0_value(const InverseSeesaw3p1Input *input, int row) {
    return input->use_complex ? input->mu_H0_2x1_complex_eV[row] : input->mu_H0_2x1_eV[row];
}

static double complex input_mu00_value(const InverseSeesaw3p1Input *input) {
    return input->use_complex ? input->mu00_complex_eV : input->mu00_eV;
}

static int inverse_2x2_complex(const double complex matrix[2][2], double complex out[2][2]) {
    const double complex det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if (cabs(det) < 1e-30) {
        return 1;
    }

    out[0][0] = matrix[1][1] / det;
    out[0][1] = -matrix[0][1] / det;
    out[1][0] = -matrix[1][0] / det;
    out[1][1] = matrix[0][0] / det;
    return 0;
}

static int select_sterile_state_index_robust(const InverseSeesaw3p1Result *result) {
    int best_idx = 0;
    double best_mass = result->masses_eV[0];
    double best_sterile_component = fabs(result->mixing_4x4[3][0]);

    for (int i = 1; i < ISS_LIGHT_DIM; ++i) {
        const double mass = result->masses_eV[i];
        const double sterile_component = fabs(result->mixing_4x4[3][i]);

        if (mass > best_mass + 1e-18 ||
            (fabs(mass - best_mass) <= 1e-18 && sterile_component > best_sterile_component + 1e-14)) {
            best_idx = i;
            best_mass = mass;
            best_sterile_component = sterile_component;
        }
    }

    return best_idx;
}

static void get_sorted_active_indices_3p1(const InverseSeesaw3p1Result *result,
                                           int sterile_idx,
                                           int active_indices[3]) {
    int fill = 0;
    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        if (i != sterile_idx) {
            active_indices[fill++] = i;
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
}

static int ISS_UNUSED build_effective_light_mass_matrix_legacy(
    const InverseSeesaw3p1Input *input,
    double m_light_4x4_eV[ISS_LIGHT_DIM][ISS_LIGHT_DIM]) {

    double inv_M[2][2];
    if (inverse_2x2(input->M_2x2_GeV, inv_M) != 0) {
        return 1;
    }

    double x[3][2] = {{0.0}};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            x[row][col] =
                input->mD_3x2_GeV[row][0] * inv_M[0][col] +
                input->mD_3x2_GeV[row][1] * inv_M[1][col];
        }
    }

    double x_muH[3][2] = {{0.0}};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            x_muH[row][col] =
                x[row][0] * input->mu_H_2x2_eV[0][col] +
                x[row][1] * input->mu_H_2x2_eV[1][col];
        }
    }

    double active_block[3][3] = {{0.0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            active_block[i][j] = x_muH[i][0] * x[j][0] + x_muH[i][1] * x[j][1];
        }
    }

    double active_sterile_block[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
        active_sterile_block[i] = -(
            x[i][0] * input->mu_H0_2x1_eV[0] +
            x[i][1] * input->mu_H0_2x1_eV[1]);
    }

    memset(m_light_4x4_eV, 0, sizeof(double) * ISS_LIGHT_DIM * ISS_LIGHT_DIM);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m_light_4x4_eV[i][j] = active_block[i][j];
        }
        m_light_4x4_eV[i][3] = active_sterile_block[i];
        m_light_4x4_eV[3][i] = active_sterile_block[i];
    }
    m_light_4x4_eV[3][3] = input->mu00_eV;

    return 0;
}

static int build_effective_light_mass_matrix_complex(
    const InverseSeesaw3p1Input *input,
    double complex m_light_4x4_eV[ISS_LIGHT_DIM][ISS_LIGHT_DIM]) {

    double complex M[2][2] = {
        {input->M_2x2_GeV[0][0], input->M_2x2_GeV[0][1]},
        {input->M_2x2_GeV[1][0], input->M_2x2_GeV[1][1]}
    };
    double complex inv_M[2][2];
    if (inverse_2x2_complex(M, inv_M) != 0) {
        return 1;
    }

    double complex x[3][2] = {{0.0}};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            x[row][col] =
                input_mD_value(input, row, 0) * inv_M[0][col] +
                input_mD_value(input, row, 1) * inv_M[1][col];
        }
    }

    double complex x_muH[3][2] = {{0.0}};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            x_muH[row][col] =
                x[row][0] * input_muH_value(input, 0, col) +
                x[row][1] * input_muH_value(input, 1, col);
        }
    }

    memset(m_light_4x4_eV, 0, sizeof(double complex) * ISS_LIGHT_DIM * ISS_LIGHT_DIM);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m_light_4x4_eV[i][j] = x_muH[i][0] * x[j][0] + x_muH[i][1] * x[j][1];
        }
        m_light_4x4_eV[i][3] = -(x[i][0] * input_muH0_value(input, 0) +
                                 x[i][1] * input_muH0_value(input, 1));
        m_light_4x4_eV[3][i] = m_light_4x4_eV[i][3];
    }
    m_light_4x4_eV[3][3] = input_mu00_value(input);

    return 0;
}

static int check_symmetric_8x8(const double matrix[ISS_FULL_DIM][ISS_FULL_DIM]) {
    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = i + 1; j < ISS_FULL_DIM; ++j) {
            if (fabs(matrix[i][j] - matrix[j][i]) > 1e-10) {
                return 1;
            }
        }
    }
    return 0;
}

static int check_complex_symmetric_8x8(const double complex matrix[ISS_FULL_DIM][ISS_FULL_DIM]) {
    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = i + 1; j < ISS_FULL_DIM; ++j) {
            if (cabs(matrix[i][j] - matrix[j][i]) > 1e-8) {
                return 1;
            }
        }
    }
    return 0;
}

static void build_full_mass_matrix_8x8_eV(
    const InverseSeesaw3p1Input *input,
    double m_full_8x8_eV[ISS_FULL_DIM][ISS_FULL_DIM]) {

    memset(m_full_8x8_eV, 0, sizeof(double) * ISS_FULL_DIM * ISS_FULL_DIM);

    for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
            const double value_eV = input->mD_3x2_GeV[a][i] * GEV_TO_EV;
            m_full_8x8_eV[a][4 + i] = value_eV;
            m_full_8x8_eV[4 + i][a] = value_eV;
        }
    }

    m_full_8x8_eV[3][3] = input->mu00_eV;

    for (int i = 0; i < 2; ++i) {
        m_full_8x8_eV[3][6 + i] = input->mu_H0_2x1_eV[i];
        m_full_8x8_eV[6 + i][3] = input->mu_H0_2x1_eV[i];
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            m_full_8x8_eV[4 + i][6 + j] = input->M_2x2_GeV[j][i] * GEV_TO_EV;
            m_full_8x8_eV[6 + i][4 + j] = input->M_2x2_GeV[i][j] * GEV_TO_EV;
            m_full_8x8_eV[6 + i][6 + j] = input->mu_H_2x2_eV[i][j];
        }
    }
}

static void build_full_mass_matrix_8x8_complex_eV(
    const InverseSeesaw3p1Input *input,
    double complex m_full_8x8_eV[ISS_FULL_DIM][ISS_FULL_DIM]) {

    memset(m_full_8x8_eV, 0, sizeof(double complex) * ISS_FULL_DIM * ISS_FULL_DIM);

    for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 2; ++i) {
            const double complex value_eV = input_mD_value(input, a, i) * GEV_TO_EV;
            m_full_8x8_eV[a][4 + i] = value_eV;
            m_full_8x8_eV[4 + i][a] = value_eV;
        }
    }

    m_full_8x8_eV[3][3] = input_mu00_value(input);

    for (int i = 0; i < 2; ++i) {
        m_full_8x8_eV[3][6 + i] = input_muH0_value(input, i);
        m_full_8x8_eV[6 + i][3] = input_muH0_value(input, i);
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            m_full_8x8_eV[4 + i][6 + j] = input->M_2x2_GeV[j][i] * GEV_TO_EV;
            m_full_8x8_eV[6 + i][4 + j] = input->M_2x2_GeV[i][j] * GEV_TO_EV;
            m_full_8x8_eV[6 + i][6 + j] = input_muH_value(input, i, j);
        }
    }
}

static void ISS_UNUSED takagi_real_symmetric_8x8(
    const double input[ISS_FULL_DIM][ISS_FULL_DIM],
    double eigenvalues[ISS_FULL_DIM],
    double eigenvectors[ISS_FULL_DIM][ISS_FULL_DIM]) {

    double a[ISS_FULL_DIM][ISS_FULL_DIM];
    memcpy(a, input, sizeof(a));

    memset(eigenvectors, 0, sizeof(double) * ISS_FULL_DIM * ISS_FULL_DIM);
    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    for (int iter = 0; iter < 320; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = fabs(a[p][q]);

        for (int i = 0; i < ISS_FULL_DIM; ++i) {
            for (int j = i + 1; j < ISS_FULL_DIM; ++j) {
                const double off = fabs(a[i][j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < 1e-12) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];

        const double phi = 0.5 * atan2(2.0 * apq, (aqq - app));
        const double c = cos(phi);
        const double s = sin(phi);

        for (int k = 0; k < ISS_FULL_DIM; ++k) {
            if (k != p && k != q) {
                const double aik = a[k][p];
                const double akq = a[k][q];
                a[k][p] = c * aik - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * aik + c * akq;
                a[q][k] = a[k][q];
            }
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (int k = 0; k < ISS_FULL_DIM; ++k) {
            const double vip = eigenvectors[k][p];
            const double viq = eigenvectors[k][q];
            eigenvectors[k][p] = c * vip - s * viq;
            eigenvectors[k][q] = s * vip + c * viq;
        }
    }

    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        eigenvalues[i] = a[i][i];
    }
}

static void jacobi_symmetric_real_matrix(
    int n,
    double a[ISS_TAKAGI_REAL_DIM][ISS_TAKAGI_REAL_DIM],
    double eigenvalues[ISS_TAKAGI_REAL_DIM],
    double eigenvectors[ISS_TAKAGI_REAL_DIM][ISS_TAKAGI_REAL_DIM]) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int iter = 0; iter < 1600; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = fabs(a[p][q]);

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double off = fabs(a[i][j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < 1e-10) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double phi = 0.5 * atan2(2.0 * apq, (aqq - app));
        const double c = cos(phi);
        const double s = sin(phi);

        for (int k = 0; k < n; ++k) {
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

        for (int k = 0; k < n; ++k) {
            const double vip = eigenvectors[k][p];
            const double viq = eigenvectors[k][q];
            eigenvectors[k][p] = c * vip - s * viq;
            eigenvectors[k][q] = s * vip + c * viq;
        }
    }

    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = a[i][i];
    }
}

static void sort_takagi_pairs_by_mass_8x8(
    double masses[ISS_FULL_DIM],
    double complex vectors[ISS_FULL_DIM][ISS_FULL_DIM]) {

    for (int i = 0; i < ISS_FULL_DIM - 1; ++i) {
        int best = i;
        double best_value = masses[i];
        for (int j = i + 1; j < ISS_FULL_DIM; ++j) {
            if (masses[j] < best_value) {
                best = j;
                best_value = masses[j];
            }
        }

        if (best != i) {
            const double mt = masses[i];
            masses[i] = masses[best];
            masses[best] = mt;

            for (int row = 0; row < ISS_FULL_DIM; ++row) {
                const double complex vt = vectors[row][i];
                vectors[row][i] = vectors[row][best];
                vectors[row][best] = vt;
            }
        }
    }
}

static int takagi_complex_symmetric_8x8(
    const double complex input[ISS_FULL_DIM][ISS_FULL_DIM],
    double masses[ISS_FULL_DIM],
    double complex vectors[ISS_FULL_DIM][ISS_FULL_DIM]) {

    double k[ISS_TAKAGI_REAL_DIM][ISS_TAKAGI_REAL_DIM] = {{0.0}};
    double evals[ISS_TAKAGI_REAL_DIM] = {0.0};
    double evecs[ISS_TAKAGI_REAL_DIM][ISS_TAKAGI_REAL_DIM] = {{0.0}};

    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = 0; j < ISS_FULL_DIM; ++j) {
            const double a = creal(input[i][j]);
            const double b = cimag(input[i][j]);
            k[i][j] = a;
            k[i][j + ISS_FULL_DIM] = b;
            k[i + ISS_FULL_DIM][j] = b;
            k[i + ISS_FULL_DIM][j + ISS_FULL_DIM] = -a;
        }
    }

    jacobi_symmetric_real_matrix(ISS_TAKAGI_REAL_DIM, k, evals, evecs);

    int selected[ISS_TAKAGI_REAL_DIM];
    int n_selected = 0;
    for (int pick = 0; pick < ISS_FULL_DIM; ++pick) {
        int best = -1;
        double best_eval = 0.0;
        for (int i = 0; i < ISS_TAKAGI_REAL_DIM; ++i) {
            if (evals[i] <= 1e-12) {
                continue;
            }
            int used = 0;
            for (int u = 0; u < n_selected; ++u) {
                if (selected[u] == i) {
                    used = 1;
                    break;
                }
            }
            if (used) {
                continue;
            }
            {
                double complex candidate[ISS_FULL_DIM];
                double norm = 0.0;
                double max_overlap = 0.0;
                for (int row = 0; row < ISS_FULL_DIM; ++row) {
                    candidate[row] = evecs[row][i] + I * evecs[row + ISS_FULL_DIM][i];
                    norm += pow(cabs(candidate[row]), 2.0);
                }
                norm = sqrt(norm);
                if (norm < 1e-30) {
                    continue;
                }
                for (int row = 0; row < ISS_FULL_DIM; ++row) {
                    candidate[row] /= norm;
                }
                for (int prev = 0; prev < n_selected; ++prev) {
                    double complex overlap = 0.0;
                    for (int row = 0; row < ISS_FULL_DIM; ++row) {
                        overlap += conj(vectors[row][prev]) * candidate[row];
                    }
                    if (cabs(overlap) > max_overlap) {
                        max_overlap = cabs(overlap);
                    }
                }
                if (max_overlap > 0.5) {
                    continue;
                }
            }
            if (best < 0 || evals[i] < best_eval) {
                best = i;
                best_eval = evals[i];
            }
        }
        if (best < 0) {
            break;
        }
        selected[n_selected++] = best;
        masses[pick] = best_eval;
        for (int row = 0; row < ISS_FULL_DIM; ++row) {
            vectors[row][pick] =
                evecs[row][best] + I * evecs[row + ISS_FULL_DIM][best];
        }
    }

    if (n_selected != ISS_FULL_DIM) {
        return 1;
    }

    for (int col = 0; col < ISS_FULL_DIM; ++col) {
        double norm = 0.0;
        for (int row = 0; row < ISS_FULL_DIM; ++row) {
            norm += pow(cabs(vectors[row][col]), 2.0);
        }
        norm = sqrt(norm);
        if (norm < 1e-30) {
            return 2;
        }
        for (int row = 0; row < ISS_FULL_DIM; ++row) {
            vectors[row][col] /= norm;
        }
    }

    sort_takagi_pairs_by_mass_8x8(masses, vectors);
    return 0;
}

static void ISS_UNUSED sort_eigenpairs_by_abs_value_8x8(
    double values[ISS_FULL_DIM],
    double vectors[ISS_FULL_DIM][ISS_FULL_DIM]) {

    for (int i = 0; i < ISS_FULL_DIM - 1; ++i) {
        int best = i;
        double best_value = fabs(values[i]);
        for (int j = i + 1; j < ISS_FULL_DIM; ++j) {
            const double candidate = fabs(values[j]);
            if (candidate < best_value) {
                best = j;
                best_value = candidate;
            }
        }

        if (best != i) {
            const double tmp = values[i];
            values[i] = values[best];
            values[best] = tmp;

            for (int row = 0; row < ISS_FULL_DIM; ++row) {
                const double t = vectors[row][i];
                vectors[row][i] = vectors[row][best];
                vectors[row][best] = t;
            }
        }
    }
}

static int ISS_UNUSED check_orthonormal_columns_8x8(
    const double vectors[ISS_FULL_DIM][ISS_FULL_DIM],
    double *max_diag_deviation,
    double *max_offdiag_abs) {

    double max_diag = 0.0;
    double max_offdiag = 0.0;

    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = 0; j < ISS_FULL_DIM; ++j) {
            double dot = 0.0;
            for (int k = 0; k < ISS_FULL_DIM; ++k) {
                dot += vectors[k][i] * vectors[k][j];
            }

            if (i == j) {
                const double dev = fabs(dot - 1.0);
                if (dev > max_diag) {
                    max_diag = dev;
                }
            } else {
                const double off = fabs(dot);
                if (off > max_offdiag) {
                    max_offdiag = off;
                }
            }
        }
    }

    if (max_diag_deviation) {
        *max_diag_deviation = max_diag;
    }
    if (max_offdiag_abs) {
        *max_offdiag_abs = max_offdiag;
    }

    return (max_diag < 1e-8 && max_offdiag < 1e-3) ? 0 : 1;
}

static int check_unitary_columns_complex_8x8(
    const double complex vectors[ISS_FULL_DIM][ISS_FULL_DIM],
    double *max_diag_deviation,
    double *max_offdiag_abs) {

    double max_diag = 0.0;
    double max_offdiag = 0.0;

    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = 0; j < ISS_FULL_DIM; ++j) {
            double complex dot = 0.0;
            for (int k = 0; k < ISS_FULL_DIM; ++k) {
                dot += conj(vectors[k][i]) * vectors[k][j];
            }

            if (i == j) {
                const double dev = fabs(creal(dot) - 1.0) + fabs(cimag(dot));
                if (dev > max_diag) {
                    max_diag = dev;
                }
            } else {
                const double off = cabs(dot);
                if (off > max_offdiag) {
                    max_offdiag = off;
                }
            }
        }
    }

    if (max_diag_deviation) {
        *max_diag_deviation = max_diag;
    }
    if (max_offdiag_abs) {
        *max_offdiag_abs = max_offdiag;
    }

    return (max_diag < 1e-8 && max_offdiag < 1e-3) ? 0 : 1;
}

static void jacobi_symmetric_4x4(
    const double input[ISS_LIGHT_DIM][ISS_LIGHT_DIM],
    double eigenvalues[ISS_LIGHT_DIM],
    double eigenvectors[ISS_LIGHT_DIM][ISS_LIGHT_DIM]) {

    double a[ISS_LIGHT_DIM][ISS_LIGHT_DIM];
    memcpy(a, input, sizeof(a));

    memset(eigenvectors, 0, sizeof(double) * ISS_LIGHT_DIM * ISS_LIGHT_DIM);
    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    for (int iter = 0; iter < 120; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = fabs(a[p][q]);

        for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
            for (int j = i + 1; j < ISS_LIGHT_DIM; ++j) {
                const double off = fabs(a[i][j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < 1e-14) {
            break;
        }

        {
            const double app = a[p][p];
            const double aqq = a[q][q];
            const double apq = a[p][q];
            const double phi = 0.5 * atan2(2.0 * apq, (aqq - app));
            const double c = cos(phi);
            const double s = sin(phi);

            for (int k = 0; k < ISS_LIGHT_DIM; ++k) {
                if (k != p && k != q) {
                    const double aik = a[k][p];
                    const double akq = a[k][q];
                    a[k][p] = c * aik - s * akq;
                    a[p][k] = a[k][p];
                    a[k][q] = s * aik + c * akq;
                    a[q][k] = a[k][q];
                }
            }

            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for (int k = 0; k < ISS_LIGHT_DIM; ++k) {
                const double vip = eigenvectors[k][p];
                const double viq = eigenvectors[k][q];
                eigenvectors[k][p] = c * vip - s * viq;
                eigenvectors[k][q] = s * vip + c * viq;
            }
        }
    }

    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        eigenvalues[i] = a[i][i];
    }
}

static void sort_eigenpairs_by_abs_value_4x4(
    double values[ISS_LIGHT_DIM],
    double vectors[ISS_LIGHT_DIM][ISS_LIGHT_DIM]) {

    for (int i = 0; i < ISS_LIGHT_DIM - 1; ++i) {
        int best = i;
        double best_value = fabs(values[i]);
        for (int j = i + 1; j < ISS_LIGHT_DIM; ++j) {
            const double candidate = fabs(values[j]);
            if (candidate < best_value) {
                best = j;
                best_value = candidate;
            }
        }

        if (best != i) {
            const double tmp = values[i];
            values[i] = values[best];
            values[best] = tmp;

            for (int row = 0; row < ISS_LIGHT_DIM; ++row) {
                const double t = vectors[row][i];
                vectors[row][i] = vectors[row][best];
                vectors[row][best] = t;
            }
        }
    }
}

static int check_orthonormal_columns_4x4(
    const double vectors[ISS_LIGHT_DIM][ISS_LIGHT_DIM],
    double *max_diag_deviation,
    double *max_offdiag_abs) {

    double max_diag = 0.0;
    double max_offdiag = 0.0;

    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        for (int j = 0; j < ISS_LIGHT_DIM; ++j) {
            double dot = 0.0;
            for (int k = 0; k < ISS_LIGHT_DIM; ++k) {
                dot += vectors[k][i] * vectors[k][j];
            }

            if (i == j) {
                const double dev = fabs(dot - 1.0);
                if (dev > max_diag) {
                    max_diag = dev;
                }
            } else {
                const double off = fabs(dot);
                if (off > max_offdiag) {
                    max_offdiag = off;
                }
            }
        }
    }

    if (max_diag_deviation) {
        *max_diag_deviation = max_diag;
    }
    if (max_offdiag_abs) {
        *max_offdiag_abs = max_offdiag;
    }

    return (max_diag < 1e-8 && max_offdiag < 1e-8) ? 0 : 1;
}

static void extract_light_heavy_from_full(
    const double full_masses[ISS_FULL_DIM],
    const double full_mixing[ISS_FULL_DIM][ISS_FULL_DIM],
    const double complex full_mixing_complex[ISS_FULL_DIM][ISS_FULL_DIM],
    InverseSeesaw3p1Result *result) {

    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        result->light_state_indices[i] = i;
        result->heavy_state_indices[i] = ISS_LIGHT_DIM + i;
        result->light_masses_eV[i] = full_masses[i];
        result->heavy_masses_eV[i] = full_masses[ISS_LIGHT_DIM + i];
    }

    for (int row = 0; row < ISS_LIGHT_DIM; ++row) {
        for (int col = 0; col < ISS_LIGHT_DIM; ++col) {
            result->mixing_4x4[row][col] = full_mixing[row][result->light_state_indices[col]];
            result->mixing_4x4_complex[row][col] = full_mixing_complex[row][result->light_state_indices[col]];
        }
    }

    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        result->masses_eV[i] = result->light_masses_eV[i];
    }

    for (int a = 0; a < 3; ++a) {
        for (int h = 0; h < ISS_HEAVY_DIM; ++h) {
            const int heavy_col = result->heavy_state_indices[h];
            result->active_heavy_mixing_abs_3x4[a][h] = cabs(full_mixing_complex[a][heavy_col]);
        }
    }
}

int inverse_seesaw_solve_3p1(
    const InverseSeesaw3p1Input *input,
    InverseSeesaw3p1Result *result) {

    if (!input || !result) {
        return 1;
    }

    memset(result, 0, sizeof(*result));

    build_full_mass_matrix_8x8_eV(input, result->m_full_8x8_eV);
    build_full_mass_matrix_8x8_complex_eV(input, result->m_full_8x8_complex_eV);
    for (int i = 0; i < ISS_FULL_DIM; ++i) {
        for (int j = 0; j < ISS_FULL_DIM; ++j) {
            result->m_full_8x8_eV[i][j] = creal(result->m_full_8x8_complex_eV[i][j]);
        }
    }

    if (check_complex_symmetric_8x8(result->m_full_8x8_complex_eV) != 0 ||
        (!input->use_complex && check_symmetric_8x8(result->m_full_8x8_eV) != 0)) {
        return 2;
    }

    if (build_effective_light_mass_matrix_complex(input, result->m_light_4x4_complex_eV) != 0) {
        return 3;
    }
    for (int i = 0; i < ISS_LIGHT_DIM; ++i) {
        for (int j = 0; j < ISS_LIGHT_DIM; ++j) {
            result->m_light_4x4_eV[i][j] = creal(result->m_light_4x4_complex_eV[i][j]);
        }
    }

    {
        double takagi_masses[ISS_FULL_DIM];
        double complex takagi_vectors[ISS_FULL_DIM][ISS_FULL_DIM];

        if (takagi_complex_symmetric_8x8(result->m_full_8x8_complex_eV, takagi_masses, takagi_vectors) != 0) {
            return 4;
        }

        {
            double max_diag_deviation = 0.0;
            double max_offdiag_abs = 0.0;
            if (check_unitary_columns_complex_8x8(takagi_vectors, &max_diag_deviation, &max_offdiag_abs) != 0) {
                fprintf(stderr,
                        "Erreur: vecteurs Takagi 8x8 non unitaires (max |u_i.u_i-1|=%.3e, max |u_i.u_j|=%.3e)\n",
                        max_diag_deviation,
                        max_offdiag_abs);
                return 5;
            }
        }

        for (int i = 0; i < ISS_FULL_DIM; ++i) {
            result->masses_full_eV[i] = fabs(takagi_masses[i]);
            for (int a = 0; a < ISS_FULL_DIM; ++a) {
                result->mixing_8x8_complex[a][i] = takagi_vectors[a][i];
                result->mixing_8x8[a][i] = cabs(takagi_vectors[a][i]);
            }
        }
    }

    extract_light_heavy_from_full(result->masses_full_eV, result->mixing_8x8, result->mixing_8x8_complex, result);

    if (!input->use_complex) {
        double raw_light_eigs[ISS_LIGHT_DIM];
        double light_vectors[ISS_LIGHT_DIM][ISS_LIGHT_DIM];
        jacobi_symmetric_4x4(result->m_light_4x4_eV, raw_light_eigs, light_vectors);
        sort_eigenpairs_by_abs_value_4x4(raw_light_eigs, light_vectors);

        {
            double max_diag_deviation = 0.0;
            double max_offdiag_abs = 0.0;
            if (check_orthonormal_columns_4x4(light_vectors, &max_diag_deviation, &max_offdiag_abs) != 0) {
                fprintf(stderr,
                        "Erreur: vecteurs propres 4x4 non orthonormes (max |v_i.v_i-1|=%.3e, max |v_i.v_j|=%.3e)\n",
                        max_diag_deviation,
                        max_offdiag_abs);
                return 5;
            }
        }

    }

    /*
     * Keep light masses/mixing from the full 8x8 Takagi diagonalization
     * (already stored by extract_light_heavy_from_full).
     *
     * The legacy 4x4 eigensystem above is only a consistency diagnostic for
     * m_light_4x4_eV and must not overwrite physical observables.
     */

    int sterile_idx = select_sterile_state_index_robust(result);
    result->sterile_state_index = sterile_idx;

    int active_indices[3];
    get_sorted_active_indices_3p1(result, sterile_idx, active_indices);

    {
        const double m1 = result->masses_eV[active_indices[0]];
        const double m2 = result->masses_eV[active_indices[1]];
        const double m3 = result->masses_eV[active_indices[2]];

        result->dm21_eV2 = m2 * m2 - m1 * m1;
        result->dm31_eV2 = m3 * m3 - m1 * m1;
    }

    {
        /* dm41 is explicitly taken from the 8x8 light/sterile extracted states. */
        const double m1_8x8 = result->masses_eV[active_indices[0]];
        const double m4_8x8 = result->masses_eV[sterile_idx];
        result->dm41_eV2 = fabs(m4_8x8 * m4_8x8 - m1_8x8 * m1_8x8);
    }

    if (result->dm21_eV2 < 0.0) result->dm21_eV2 = fabs(result->dm21_eV2);
    if (result->dm31_eV2 < 0.0) result->dm31_eV2 = fabs(result->dm31_eV2);
    if (result->dm41_eV2 < 1e-18) result->dm41_eV2 = 1e-18;

    {
        const double ue4 = fabs(result->mixing_4x4[0][sterile_idx]);
        const double umu4 = fabs(result->mixing_4x4[1][sterile_idx]);
        const double utau4 = fabs(result->mixing_4x4[2][sterile_idx]);

        const double c14 = sqrt(fmax(0.0, 1.0 - ue4 * ue4));
        const double sin24 = (c14 > 1e-14) ? umu4 / c14 : 0.0;
        const double c24 = sqrt(fmax(0.0, 1.0 - sin24 * sin24));
        const double denom34 = c14 * c24;
        const double sin34 = (denom34 > 1e-14) ? utau4 / denom34 : 0.0;

        result->theta14_deg = asin(clamp_unit(ue4)) * 180.0 / M_PI;
        result->theta24_deg = asin(clamp_unit(sin24)) * 180.0 / M_PI;
        result->theta34_deg = asin(clamp_unit(sin34)) * 180.0 / M_PI;
    }

    return 0;
}
