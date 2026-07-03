#ifndef INVERSE_SEESAW_H
#define INVERSE_SEESAW_H

#include <complex.h>

typedef struct {
    double mD_3x2_GeV[3][2];
    double complex mD_3x2_complex_GeV[3][2];
    double M_2x2_GeV[2][2];
    double mu_H_2x2_eV[2][2];
    double complex mu_H_2x2_complex_eV[2][2];
    double mu_H0_2x1_eV[2];
    double complex mu_H0_2x1_complex_eV[2];
    double mu00_eV;
    double complex mu00_complex_eV;
    int use_complex;
} InverseSeesaw3p1Input;

typedef struct {
    double m_full_8x8_eV[8][8];
    double complex m_full_8x8_complex_eV[8][8];
    double masses_full_eV[8];
    double mixing_8x8[8][8];
    double complex mixing_8x8_complex[8][8];
    int light_state_indices[4];
    int heavy_state_indices[4];
    double light_masses_eV[4];
    double heavy_masses_eV[4];
    double active_heavy_mixing_abs_3x4[3][4];

    double m_light_4x4_eV[4][4];
    double complex m_light_4x4_complex_eV[4][4];
    double masses_eV[4];
    double mixing_4x4[4][4];
    double complex mixing_4x4_complex[4][4];
    int sterile_state_index;

    double dm21_eV2;
    double dm31_eV2;
    double dm41_eV2;

    double theta14_deg;
    double theta24_deg;
    double theta34_deg;
} InverseSeesaw3p1Result;

int inverse_seesaw_solve_3p1(
    const InverseSeesaw3p1Input *input,
    InverseSeesaw3p1Result *result);

#endif
