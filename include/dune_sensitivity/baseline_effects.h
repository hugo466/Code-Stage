#ifndef DUNE_SENSITIVITY_BASELINE_EFFECTS_H
#define DUNE_SENSITIVITY_BASELINE_EFFECTS_H

#include "config.h"
#include "dune/dune.h"

#define DUNE_SENSITIVITY_MAX_BINS 8192
#define DUNE_SENSITIVITY_N_NORM_PULLS 26

typedef struct {
    char detector[8];
    char panel[16];
    char component[16];
    double e_rec_GeV;
    double asimov_events;
    double test_events;
} DuneSensitivitySpectrumBin;

int dune_nd_fig4_build_sensitivity_rows(
    const SimulationConfig *cfg,
    const DuneTheoryPoint *asimov,
    const DuneTheoryPoint *test,
    DuneSensitivitySpectrumBin *rows,
    int max_rows,
    int *out_count);

int dune_fd_fig4_build_sensitivity_rows(
    const SimulationConfig *cfg,
    const DuneTheoryPoint *asimov,
    const DuneTheoryPoint *test,
    DuneSensitivitySpectrumBin *rows,
    int max_rows,
    int *out_count);

int run_dune_baseline_effects_sensitivity(const SimulationConfig *cfg);

#endif
