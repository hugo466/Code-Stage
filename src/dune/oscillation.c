#include "dune/oscillation.h"

#include <complex.h>
#include <math.h>

DuneStatus dune_osc_engine_build_kernel(DuneRunContext *ctx) {
    if (!ctx) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }
    if (ctx->theory.dm41_eV2 <= 0.0) {
        ctx->kernel.regime = DUNE_REGIME_UNKNOWN;
        return DUNE_STATUS_MISSING_INPUT;
    }
    ctx->kernel.regime = DUNE_REGIME_EXACT_LIGHT;
    return DUNE_STATUS_OK;
}

DuneStatus dune_exact_light_probability(
    const DuneTheoryPoint *point,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    double *probability) {
    if (!point || !probability || energy_GeV <= 0.0 || baseline_km < 0.0 ||
        alpha < 0 || beta < 0 || alpha >= point->n_light || beta >= point->n_light) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    double p = (alpha == beta) ? 1.0 : 0.0;
    for (int i = 0; i < point->n_light; ++i) {
        const double mi2 = point->light_masses_eV[i] * point->light_masses_eV[i];
        for (int j = i + 1; j < point->n_light; ++j) {
            const double mj2 = point->light_masses_eV[j] * point->light_masses_eV[j];
            const double phase = 1.267 * (mj2 - mi2) * baseline_km / energy_GeV;
            const double complex a =
                point->mixing[alpha][i] * conj(point->mixing[beta][i]) *
                conj(point->mixing[alpha][j]) * point->mixing[beta][j];
            p -= 4.0 * creal(a) * sin(phase) * sin(phase);
            p += 2.0 * cimag(a) * sin(2.0 * phase);
        }
    }

    if (p < 0.0 && p > -1e-12) p = 0.0;
    if (p > 1.0 && p < 1.0 + 1e-12) p = 1.0;
    *probability = p;
    return DUNE_STATUS_OK;
}
