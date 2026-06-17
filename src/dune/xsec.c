#include "dune/xsec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DuneXsecTable g_xsec_cc;
static DuneXsecTable g_xsec_nc;
static int g_has_cc = 0;
static int g_has_nc = 0;

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

DuneStatus dune_xsec_table_load_globes(const char *path, DuneXsecTable *table) {
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
            if (table->n_points >= DUNE_XSEC_TABLE_MAX_POINTS) {
                fclose(in);
                return DUNE_STATUS_MISSING_INPUT;
            }
            const int idx = table->n_points++;
            table->energy_GeV[idx] = pow(10.0, row[0]);
            for (int f = 0; f < 6; ++f) {
                table->xsec[f][idx] = row[f + 1];
            }
            row_count = 0;
        }
    }

    fclose(in);
    return table->n_points > 0 ? DUNE_STATUS_OK : DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_xsec_table_eval(
    const DuneXsecTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double *xsec) {
    if (!table || !xsec || table->n_points <= 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }
    const int col = flavor_to_globes_column(flavor);
    if (col < 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (energy_GeV <= table->energy_GeV[0]) {
        *xsec = table->xsec[col][0];
        return DUNE_STATUS_OK;
    }
    if (energy_GeV >= table->energy_GeV[table->n_points - 1]) {
        *xsec = table->xsec[col][table->n_points - 1];
        return DUNE_STATUS_OK;
    }

    for (int i = 0; i < table->n_points - 1; ++i) {
        const double e0 = table->energy_GeV[i];
        const double e1 = table->energy_GeV[i + 1];
        if (energy_GeV >= e0 && energy_GeV <= e1) {
            const double t = (energy_GeV - e0) / (e1 - e0);
            *xsec = table->xsec[col][i] + t * (table->xsec[col][i + 1] - table->xsec[col][i]);
            return DUNE_STATUS_OK;
        }
    }

    return DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_xsec_model_load_from_config(const SimulationConfig *cfg) {
    if (!cfg) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    g_has_cc = 0;
    g_has_nc = 0;
    if (strcmp(cfg->dune_xsec_model, "table") != 0) {
        return DUNE_STATUS_OK;
    }
    if (cfg->dune_xsec_cc_file[0] != '\0') {
        DuneStatus st = dune_xsec_table_load_globes(cfg->dune_xsec_cc_file, &g_xsec_cc);
        if (st != DUNE_STATUS_OK) return st;
        g_has_cc = 1;
    }
    if (cfg->dune_xsec_nc_file[0] != '\0') {
        DuneStatus st = dune_xsec_table_load_globes(cfg->dune_xsec_nc_file, &g_xsec_nc);
        if (st != DUNE_STATUS_OK) return st;
        g_has_nc = 1;
    }
    return (g_has_cc || g_has_nc) ? DUNE_STATUS_OK : DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_xsec_model_eval(
    DuneFluxFlavor flavor,
    int use_cc,
    double energy_GeV,
    double *xsec) {
    if (!xsec) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (use_cc) {
        if (!g_has_cc) return DUNE_STATUS_MISSING_INPUT;
        return dune_xsec_table_eval(&g_xsec_cc, flavor, energy_GeV, xsec);
    }

    if (!g_has_nc) return DUNE_STATUS_MISSING_INPUT;
    return dune_xsec_table_eval(&g_xsec_nc, flavor, energy_GeV, xsec);
}
