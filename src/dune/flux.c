#include "dune/flux.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DuneFluxTable g_flux_fhc;
static DuneFluxTable g_flux_rhc;
static int g_has_fhc = 0;
static int g_has_rhc = 0;
static int g_use_table = 0;
static DuneBeamMode g_mode = DUNE_BEAM_FHC;

typedef struct {
    double norm;
    double power;
    double cutoff_GeV;
} FluxAnalyticParams;

static int flavor_to_globes_column(DuneFluxFlavor flavor) {
    switch (flavor) {
        case DUNE_FLUX_NUE: return 0;
        case DUNE_FLUX_NUMU: return 1;
        case DUNE_FLUX_NUTAU: return 2;
        case DUNE_FLUX_NUEBAR: return 3;
        case DUNE_FLUX_NUMUBAR: return 4;
        case DUNE_FLUX_NUTAUBAR: return 5;
        default: return -1;
    }
}

static int parse_globes_number(const char *token, double *out) {
    char buffer[128];
    size_t n = strlen(token);
    if (n >= sizeof(buffer)) {
        n = sizeof(buffer) - 1;
    }
    memcpy(buffer, token, n);
    buffer[n] = '\0';
    for (size_t i = 0; i < n; ++i) {
        if (buffer[i] == ',') {
            buffer[i] = '.';
        }
    }
    char *end = NULL;
    const double value = strtod(buffer, &end);
    if (end == buffer) {
        return 0;
    }
    *out = value;
    return 1;
}

double dune_flux_analytic_test(double energy_GeV) {
    if (energy_GeV <= 0.0) {
        return 0.0;
    }

    FluxAnalyticParams defaults = {1.0, 2.0, 2.5};
    const FluxAnalyticParams *p = &defaults;
    if (p->cutoff_GeV <= 0.0) {
        return 0.0;
    }

    return p->norm * pow(energy_GeV, p->power) * exp(-energy_GeV / p->cutoff_GeV);
}

DuneStatus dune_flux_table_load_globes(const char *path, DuneFluxTable *table) {
    if (!path || !table) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    FILE *in = fopen(path, "r");
    if (!in) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    memset(table, 0, sizeof(*table));

    char token[128];
    double row[7];
    int row_count = 0;
    while (fscanf(in, "%127s", token) == 1) {
        double value = 0.0;
        if (!parse_globes_number(token, &value)) {
            continue;
        }
        row[row_count++] = value;
        if (row_count == 7) {
            if (table->n_points >= DUNE_FLUX_TABLE_MAX_POINTS) {
                fclose(in);
                return DUNE_STATUS_MISSING_INPUT;
            }
            const int idx = table->n_points++;
            table->energy_GeV[idx] = row[0];
            for (int f = 0; f < 6; ++f) {
                table->flux[f][idx] = row[f + 1];
            }
            row_count = 0;
        }
    }

    fclose(in);
    return table->n_points > 0 ? DUNE_STATUS_OK : DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_flux_table_eval(
    const DuneFluxTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double *flux) {
    if (!table || !flux || table->n_points <= 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }
    const int col = flavor_to_globes_column(flavor);
    if (col < 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (energy_GeV <= table->energy_GeV[0]) {
        *flux = table->flux[col][0];
        return DUNE_STATUS_OK;
    }
    if (energy_GeV >= table->energy_GeV[table->n_points - 1]) {
        *flux = table->flux[col][table->n_points - 1];
        return DUNE_STATUS_OK;
    }

    for (int i = 0; i < table->n_points - 1; ++i) {
        const double e0 = table->energy_GeV[i];
        const double e1 = table->energy_GeV[i + 1];
        if (energy_GeV >= e0 && energy_GeV <= e1) {
            const double t = (energy_GeV - e0) / (e1 - e0);
            *flux = table->flux[col][i] + t * (table->flux[col][i + 1] - table->flux[col][i]);
            return DUNE_STATUS_OK;
        }
    }

    return DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_flux_provider_load(DuneRunContext *ctx) {
    if (!ctx) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    g_use_table = 0;
    g_has_fhc = 0;
    g_has_rhc = 0;
    g_mode = (strcmp(ctx->cfg->dune_beam_mode, "RHC") == 0 || strcmp(ctx->cfg->dune_beam_mode, "rhc") == 0)
                 ? DUNE_BEAM_RHC
                 : DUNE_BEAM_FHC;

    if (strcmp(ctx->cfg->dune_flux_model, "table") == 0) {
        g_use_table = 1;
        if (ctx->cfg->dune_flux_fhc_file[0] != '\0') {
            DuneStatus st = dune_flux_table_load_globes(ctx->cfg->dune_flux_fhc_file, &g_flux_fhc);
            if (st != DUNE_STATUS_OK) return st;
            g_has_fhc = 1;
        }
        if (ctx->cfg->dune_flux_rhc_file[0] != '\0') {
            DuneStatus st = dune_flux_table_load_globes(ctx->cfg->dune_flux_rhc_file, &g_flux_rhc);
            if (st != DUNE_STATUS_OK) return st;
            g_has_rhc = 1;
        }
        if ((g_mode == DUNE_BEAM_FHC && !g_has_fhc) || (g_mode == DUNE_BEAM_RHC && !g_has_rhc)) {
            return DUNE_STATUS_MISSING_INPUT;
        }
    }

    return DUNE_STATUS_OK;
}

DuneStatus dune_flux_provider_eval(
    const DuneRunContext *ctx,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double source_z_m,
    double *flux) {
    (void)ctx;
    (void)source_z_m;
    if (!flux) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (g_use_table) {
        const DuneFluxTable *table = (g_mode == DUNE_BEAM_RHC) ? &g_flux_rhc : &g_flux_fhc;
        return dune_flux_table_eval(table, flavor, energy_GeV, flux);
    }

    double flavor_scale = 1.0;
    switch (flavor) {
        case DUNE_FLUX_NUMU:
            flavor_scale = 1.0;
            break;
        case DUNE_FLUX_NUMUBAR:
            flavor_scale = 0.35;
            break;
        case DUNE_FLUX_NUE:
            flavor_scale = 0.05;
            break;
        case DUNE_FLUX_NUEBAR:
            flavor_scale = 0.02;
            break;
        case DUNE_FLUX_NUTAU:
        case DUNE_FLUX_NUTAUBAR:
            flavor_scale = 0.0;
            break;
        default:
            return DUNE_STATUS_INVALID_ARGUMENT;
    }

    *flux = flavor_scale * dune_flux_analytic_test(energy_GeV);
    return DUNE_STATUS_OK;
}

DuneStatus dune_source_geometry_baseline(
    const DuneSourceGeometry *geometry,
    double source_z_m,
    double *baseline_km) {
    if (!geometry || !baseline_km || geometry->baseline_km <= 0.0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (geometry->model == DUNE_SOURCE_GEOMETRY_FIXED) {
        *baseline_km = geometry->baseline_km;
        return DUNE_STATUS_OK;
    }

    if (geometry->model == DUNE_SOURCE_GEOMETRY_SOURCE_LINE) {
        const double value = geometry->baseline_km - source_z_m * 1.0e-3;
        *baseline_km = value > 0.0 ? value : 0.0;
        return DUNE_STATUS_OK;
    }

    return DUNE_STATUS_INVALID_ARGUMENT;
}
