#include "scan.h"

#include "dune_nd/dune_nd.h"
#include "output/csv_writer.h"
#include "samples/prediction_builder.h"
#include "stats/asimov.h"
#include "stats/likelihood.h"
#include "studies/study_minimal_onaxis.h"
#include "studies/study_phase2_full_nd.h"
#include "studies/study_robust_prism.h"

#include <stdio.h>

static int dune_status_to_run_code(DuneStatus status) {
    return status == DUNE_STATUS_OK ? 0 : 1;
}

int run_dune_nd_predict_spectrum(const SimulationConfig *cfg) {
    DuneRunContext ctx;
    DuneSamplePrediction prediction = {0};
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND init failed: %s\n", dune_status_message(status));
        return dune_status_to_run_code(status);
    }

    status = dune_prediction_builder_run(&ctx, &prediction);
    if (status == DUNE_STATUS_OK && cfg->dune_spectrum_pred_csv[0] != '\0') {
        status = dune_csv_writer_write_prediction(cfg->dune_spectrum_pred_csv, &prediction);
    }
    if (status == DUNE_STATUS_OK && cfg->dune_spectrum_null_csv[0] != '\0') {
        status = dune_csv_writer_write_null(cfg->dune_spectrum_null_csv, &prediction);
    }
    if (status == DUNE_STATUS_OK && cfg->dune_residuals_csv[0] != '\0') {
        status = dune_csv_writer_write_residuals(cfg->dune_residuals_csv, &prediction);
    }
    if (status == DUNE_STATUS_OK && cfg->dune_point_observables_csv[0] != '\0') {
        status = dune_csv_writer_write_point_observables(cfg->dune_point_observables_csv, &ctx.theory);
    }
    fprintf(stderr, "DUNE ND predict spectrum: %s\n", dune_status_message(status));
    return dune_status_to_run_code(status);
}

int run_dune_nd_asimov_chi2(const SimulationConfig *cfg) {
    DuneRunContext ctx;
    DuneSamplePrediction prediction = {0};
    double chi2 = 0.0;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND init failed: %s\n", dune_status_message(status));
        return dune_status_to_run_code(status);
    }

    status = dune_prediction_builder_run(&ctx, &prediction);
    if (status == DUNE_STATUS_OK) {
        status = dune_asimov_build(&prediction);
    }
    if {
       (status == DUNE_STATUS_OK) {
        status = dune_asimov_build(&prediction); 
    }
    if (status == DUNE_STATUS_OK) {
        status = dune_likelihood_poisson_chi2(&prediction, &chi2);
    }
    fprintf(stderr, "DUNE ND Asimov chi2: %s (chi2=%.6g)\n", dune_status_message(status), chi2);
    return dune_status_to_run_code(status);
}

int run_study_minimal_onaxis(const SimulationConfig *cfg) {
    DuneRunContext ctx;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status == DUNE_STATUS_OK) {
        status = dune_study_minimal_onaxis_run(&ctx);
    }
    fprintf(stderr, "DUNE study minimal on-axis: %s\n", dune_status_message(status));
    return dune_status_to_run_code(status);
}

int run_study_robust_prism(const SimulationConfig *cfg) {
    DuneRunContext ctx;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status == DUNE_STATUS_OK) {
        status = dune_study_robust_prism_run(&ctx);
    }
    fprintf(stderr, "DUNE study robust PRISM: %s\n", dune_status_message(status));
    return dune_status_to_run_code(status);
}

int run_study_phase2_full_nd(const SimulationConfig *cfg) {
    DuneRunContext ctx;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status == DUNE_STATUS_OK) {
        status = dune_study_phase2_full_nd_run(&ctx);
    }
    fprintf(stderr, "DUNE study phase2 full ND: %s\n", dune_status_message(status));
    return dune_status_to_run_code(status);
}
