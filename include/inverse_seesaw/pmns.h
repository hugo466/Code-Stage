#ifndef PMNS_H
#define PMNS_H

#include "config.h"

#include <complex.h>

void pmns_build_3x3(
    double theta12,
    double theta13,
    double theta23,
    double delta_cp,
    double complex u3[3][3]);

int pmns_build_general(const SimulationConfig *cfg, int n_sterile, double complex u[3 + n_sterile][3 + n_sterile]);
int pmns_build_3plus1(const SimulationConfig *cfg, double complex u[4][4]);
int pmns_build_3plus2(const SimulationConfig *cfg, double complex u[5][5]);

#endif