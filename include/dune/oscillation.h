#ifndef DUNE_OSCILLATION_H
#define DUNE_OSCILLATION_H

#include "dune/dune.h"

DuneStatus dune_osc_engine_build_kernel(DuneRunContext *ctx);
DuneStatus dune_exact_light_probability(
    const DuneTheoryPoint *point,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    double *probability);

#endif
