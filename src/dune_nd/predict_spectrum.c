#include "scan.h"

#include "dune/dune.h"
#include "dune/csv_writer.h"
#include "dune/flux.h"
#include "dune/oscillation.h"
#include "dune/theory.h"
#include "dune/xsec.h"

#include <stdio.h>
#include <string.h>

static int dune_status_to_run_code(DuneStatus status) {
    return status == DUNE_STATUS_OK ? 0 : 1;
}

static DuneStatus build_minimal_onaxis_prediction(DuneRunContext *ctx, DuneSamplePrediction *prediction) {
    if (!ctx || !prediction) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    DuneStatus status = dune_theory_point_load(ctx);
    if (status != DUNE_STATUS_OK) {
        return status;
    }
    ctx->kernel.theory = ctx->theory;

    status = dune_flux_provider_load(ctx);
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    status = dune_osc_engine_build_kernel(ctx);
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    status = dune_xsec_model_load_from_config(ctx->cfg);
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    int n_bins = ctx->cfg->dune_Erec_bins > 0 ? ctx->cfg->dune_Erec_bins : 55;
    if (n_bins > DUNE_ND_MAX_BINS) {
        n_bins = DUNE_ND_MAX_BINS;
    }
    const double e_min = ctx->cfg->dune_Erec_min_GeV > 0.0 ? ctx->cfg->dune_Erec_min_GeV : ctx->cfg->energy_min_GeV;
    const double e_max = ctx->cfg->dune_Erec_max_GeV > e_min ? ctx->cfg->dune_Erec_max_GeV : ctx->cfg->energy_max_GeV;
    if (e_min <= 0.0 || e_max <= e_min || n_bins <= 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    DuneSourceGeometry geometry;
    memset(&geometry, 0, sizeof(geometry));
    geometry.baseline_km = ctx->cfg->dune_detector_distance_m > 0.0
                               ? ctx->cfg->dune_detector_distance_m * 1.0e-3
                               : ctx->cfg->baseline_km;
    geometry.source_z_min_m = 0.0;
    geometry.source_z_max_m = ctx->cfg->dune_decay_pipe_length_m > 0.0
                                  ? ctx->cfg->dune_decay_pipe_length_m
                                  : 0.0;
    geometry.n_source_steps = ctx->cfg->dune_source_z_bins > 0 ? ctx->cfg->dune_source_z_bins : 1;
    geometry.model = strcmp(ctx->cfg->dune_baseline_model, "source_line") == 0
                         ? DUNE_SOURCE_GEOMETRY_SOURCE_LINE
                         : DUNE_SOURCE_GEOMETRY_FIXED;

    prediction->n_bins = n_bins;
    strncpy(prediction->name, "dune_nd_minimal_onaxis", sizeof(prediction->name) - 1);

    const double width = (e_max - e_min) / (double)n_bins;
    for (int b = 0; b < n_bins; ++b) {
        const double e0 = e_min + width * (double)b;
        const double e1 = e0 + width;
        const double e = 0.5 * (e0 + e1);
        prediction->bin_low_GeV[b] = e0;
        prediction->bin_high_GeV[b] = e1;

        double flux_numu = 0.0;
        status = dune_flux_provider_eval(ctx, DUNE_FLUX_NUMU, e, 0.0, &flux_numu);
        if (status != DUNE_STATUS_OK) return status;

        double xsec_mu_cc = 0.0;
        double xsec_e_cc = 0.0;
        status = dune_xsec_model_eval(DUNE_FLUX_NUMU, 1, e, &xsec_mu_cc);
        if (status != DUNE_STATUS_OK) return status;
        status = dune_xsec_model_eval(DUNE_FLUX_NUE, 1, e, &xsec_e_cc);
        if (status != DUNE_STATUS_OK) return status;

        double p_mumu_avg = 0.0;
        double p_mue_avg = 0.0;
        for (int iz = 0; iz < geometry.n_source_steps; ++iz) {
            const double t = ((double)iz + 0.5) / (double)geometry.n_source_steps;
            const double z = geometry.source_z_min_m + t * (geometry.source_z_max_m - geometry.source_z_min_m);
            double baseline_km = 0.0;
            status = dune_source_geometry_baseline(&geometry, z, &baseline_km);
            if (status != DUNE_STATUS_OK) return status;

            double p_mumu = 0.0;
            double p_mue = 0.0;
            status = dune_exact_light_probability(&ctx->theory, 1, 1, e, baseline_km, &p_mumu);
            if (status != DUNE_STATUS_OK) return status;
            status = dune_exact_light_probability(&ctx->theory, 1, 0, e, baseline_km, &p_mue);
            if (status != DUNE_STATUS_OK) return status;
            p_mumu_avg += p_mumu;
            p_mue_avg += p_mue;
        }
        p_mumu_avg /= (double)geometry.n_source_steps;
        p_mue_avg /= (double)geometry.n_source_steps;

        prediction->mu_like[b] = flux_numu * p_mumu_avg * xsec_mu_cc * width;
        prediction->e_like[b] = flux_numu * p_mue_avg * xsec_e_cc * width;
        prediction->null_mu_like[b] = flux_numu * xsec_mu_cc * width;
        prediction->null_e_like[b] = 0.0;
    }

    return DUNE_STATUS_OK;
}

int run_dune_nd_predict_spectrum(const SimulationConfig *cfg) {
    if (!cfg) {
        fprintf(stderr, "DUNE ND predict spectrum: configuration invalide\n");
        return dune_status_to_run_code(DUNE_STATUS_INVALID_ARGUMENT);
    }

    DuneRunContext ctx;
    DuneSamplePrediction prediction = {0};
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND init failed: %s\n", dune_status_message(status));
        return dune_status_to_run_code(status);
    }

    status = build_minimal_onaxis_prediction(&ctx, &prediction);
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
