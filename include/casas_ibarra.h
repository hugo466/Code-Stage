#ifndef CASAS_IBARRA_H
#define CASAS_IBARRA_H

#include <complex.h>

typedef struct {
    double m_light_eV[3];
    double M_heavy_GeV[2];

    double theta12_deg;
    double theta13_deg;
    double theta23_deg;
    double delta_cp_deg;

    double alpha21_deg;
    double alpha31_deg;

    double z_real;
    double z_imag;

    int normal_ordering;
} CasasIbarraInput3x2;

int casas_ibarra_build_md_3x2(
    const CasasIbarraInput3x2 *input,
    double complex mD_3x2_GeV[3][2]);

int casas_ibarra_build_yukawa_3x2(
    const CasasIbarraInput3x2 *input,
    double complex yukawa_3x2[3][2]);

#endif
