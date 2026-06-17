#ifndef DUNE_DUNE_H
#define DUNE_DUNE_H

#include "config.h"

#include <complex.h>

#define DUNE_ND_MAX_BINS 512

typedef enum {
    DUNE_STATUS_OK = 0,
    DUNE_STATUS_INVALID_ARGUMENT = 1,
    DUNE_STATUS_MISSING_INPUT = 2
} DuneStatus;

typedef enum {
    DUNE_REGIME_UNKNOWN = 0,
    DUNE_REGIME_EXACT_LIGHT,
    DUNE_REGIME_AVERAGED_OUT,
    DUNE_REGIME_NONUNITARITY,
    DUNE_REGIME_HNL_DIRECT
} DunePhysicsRegime;

typedef struct {
    int point_id;
    char model[32];
    int n_light;
    int n_active;
    double dm21_eV2;
    double dm31_eV2;
    double dm41_eV2;
    double light_masses_eV[8];
    double complex mixing[8][8];
    double eta_abs_3x3[3][3];
} DuneTheoryPoint;

typedef struct {
    DunePhysicsRegime regime;
    DuneTheoryPoint theory;
} DunePhysicsKernel;

typedef struct {
    char name[64];
    int n_bins;
    double bin_low_GeV[DUNE_ND_MAX_BINS];
    double bin_high_GeV[DUNE_ND_MAX_BINS];
    double mu_like[DUNE_ND_MAX_BINS];
    double e_like[DUNE_ND_MAX_BINS];
    double null_mu_like[DUNE_ND_MAX_BINS];
    double null_e_like[DUNE_ND_MAX_BINS];
} DuneSamplePrediction;

typedef struct {
    const SimulationConfig *cfg;
    DuneTheoryPoint theory;
    DunePhysicsKernel kernel;
} DuneRunContext;

const char *dune_status_message(DuneStatus status);
DuneStatus dune_run_context_init(DuneRunContext *ctx, const SimulationConfig *cfg);

#endif
