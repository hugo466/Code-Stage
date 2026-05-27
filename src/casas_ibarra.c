#include "casas_ibarra.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double deg_to_rad_local(double degrees) {
    return degrees * (M_PI / 180.0);
}

static void pmns_build_3x3_with_majorana(
    double theta12,
    double theta13,
    double theta23,
    double delta_cp,
    double alpha21,
    double alpha31,
    double complex u[3][3]) {

    const double c12 = cos(theta12);
    const double s12 = sin(theta12);
    const double c13 = cos(theta13);
    const double s13 = sin(theta13);
    const double c23 = cos(theta23);
    const double s23 = sin(theta23);

    u[0][0] = c12 * c13;
    u[0][1] = s12 * c13;
    u[0][2] = s13 * cexp(-I * delta_cp);

    u[1][0] = -s12 * c23 - c12 * s23 * s13 * cexp(I * delta_cp);
    u[1][1] = c12 * c23 - s12 * s23 * s13 * cexp(I * delta_cp);
    u[1][2] = s23 * c13;

    u[2][0] = s12 * s23 - c12 * c23 * s13 * cexp(I * delta_cp);
    u[2][1] = -c12 * s23 - s12 * c23 * s13 * cexp(I * delta_cp);
    u[2][2] = c23 * c13;

    const double complex p1 = 1.0;
    const double complex p2 = cexp(I * 0.5 * alpha21);
    const double complex p3 = cexp(I * 0.5 * alpha31);

    for (int row = 0; row < 3; ++row) {
        u[row][0] *= p1;
        u[row][1] *= p2;
        u[row][2] *= p3;
    }
}

static int build_r_3x2(
    const CasasIbarraInput3x2 *input,
    double complex r[3][2]) {

    const double complex z = input->z_real + I * input->z_imag;
    const double complex cz = ccos(z);
    const double complex sz = csin(z);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            r[i][j] = 0.0;
        }
    }

    if (input->normal_ordering) {
        r[1][0] = cz;
        r[1][1] = -sz;
        r[2][0] = sz;
        r[2][1] = cz;
    } else {
        r[0][0] = cz;
        r[0][1] = -sz;
        r[1][0] = sz;
        r[1][1] = cz;
    }

    return 0;
}

int casas_ibarra_build_md_3x2(
    const CasasIbarraInput3x2 *input,
    double complex mD_3x2_GeV[3][2]) {

    if (!input || !mD_3x2_GeV) {
        return 1;
    }

    for (int i = 0; i < 3; ++i) {
        if (input->m_light_eV[i] < 0.0) {
            return 2;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (input->M_heavy_GeV[i] <= 0.0) {
            return 3;
        }
    }

    double complex u[3][3];
    pmns_build_3x3_with_majorana(
        deg_to_rad_local(input->theta12_deg),
        deg_to_rad_local(input->theta13_deg),
        deg_to_rad_local(input->theta23_deg),
        deg_to_rad_local(input->delta_cp_deg),
        deg_to_rad_local(input->alpha21_deg),
        deg_to_rad_local(input->alpha31_deg),
        u);

    double complex r[3][2];
    build_r_3x2(input, r);

    double sqrt_m_eV[3];
    for (int i = 0; i < 3; ++i) {
        sqrt_m_eV[i] = sqrt(input->m_light_eV[i]);
    }

    double sqrt_M_eV[2];
    for (int i = 0; i < 2; ++i) {
        sqrt_M_eV[i] = sqrt(input->M_heavy_GeV[i] * 1e9);
    }

    for (int alpha = 0; alpha < 3; ++alpha) {
        for (int k = 0; k < 2; ++k) {
            double complex sum = 0.0;
            for (int i = 0; i < 3; ++i) {
                sum += conj(u[alpha][i]) * sqrt_m_eV[i] * r[i][k];
            }

            const double complex mD_eV = I * sum * sqrt_M_eV[k];
            mD_3x2_GeV[alpha][k] = mD_eV * 1e-9;
        }
    }

    return 0;
}

int casas_ibarra_build_yukawa_3x2(
    const CasasIbarraInput3x2 *input,
    double complex yukawa_3x2[3][2]) {

    if (!input || !yukawa_3x2) {
        return 1;
    }

    double complex mD_3x2_GeV[3][2];
    const int status = casas_ibarra_build_md_3x2(input, mD_3x2_GeV);
    if (status != 0) {
        return status;
    }

    const double prefactor = sqrt(2.0) / 246.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            yukawa_3x2[i][j] = prefactor * mD_3x2_GeV[i][j];
        }
    }

    return 0;
}
