#include "dune/dk2nu_flux_z.h"

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trim(char *text) {
    size_t len = strlen(text);
    while (len > 0 && isspace((unsigned char)text[len - 1])) {
        text[--len] = '\0';
    }
    size_t start = 0;
    while (text[start] && isspace((unsigned char)text[start])) {
        ++start;
    }
    if (start > 0) {
        memmove(text, text + start, strlen(text + start) + 1);
    }
}

static int flavor_from_string(const char *text, DuneFluxFlavor *flavor) {
    char buffer[32];
    size_t n = strlen(text);
    if (n >= sizeof(buffer)) n = sizeof(buffer) - 1;
    memcpy(buffer, text, n);
    buffer[n] = '\0';
    trim(buffer);
    for (size_t i = 0; buffer[i]; ++i) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
    }

    if (strcmp(buffer, "nue") == 0 || strcmp(buffer, "12") == 0) {
        *flavor = DUNE_FLUX_NUE;
        return 1;
    }
    if (strcmp(buffer, "numu") == 0 || strcmp(buffer, "14") == 0) {
        *flavor = DUNE_FLUX_NUMU;
        return 1;
    }
    if (strcmp(buffer, "nutau") == 0 || strcmp(buffer, "16") == 0) {
        *flavor = DUNE_FLUX_NUTAU;
        return 1;
    }
    if (strcmp(buffer, "nuebar") == 0 || strcmp(buffer, "anti_nue") == 0 || strcmp(buffer, "-12") == 0) {
        *flavor = DUNE_FLUX_NUEBAR;
        return 1;
    }
    if (strcmp(buffer, "numubar") == 0 || strcmp(buffer, "anti_numu") == 0 || strcmp(buffer, "-14") == 0) {
        *flavor = DUNE_FLUX_NUMUBAR;
        return 1;
    }
    if (strcmp(buffer, "nutaubar") == 0 || strcmp(buffer, "anti_nutau") == 0 || strcmp(buffer, "-16") == 0) {
        *flavor = DUNE_FLUX_NUTAUBAR;
        return 1;
    }
    return 0;
}

static int ensure_capacity(DuneDk2nuFluxZTable *table, int need) {
    if (need <= table->capacity) {
        return 1;
    }
    int next = table->capacity > 0 ? table->capacity * 2 : 4096;
    while (next < need) next *= 2;
    DuneDk2nuFluxZRow *rows = (DuneDk2nuFluxZRow *)realloc(table->rows, (size_t)next * sizeof(*rows));
    if (!rows) {
        return 0;
    }
    table->rows = rows;
    table->capacity = next;
    return 1;
}

DuneStatus dune_dk2nu_flux_z_load_csv(const char *path, DuneDk2nuFluxZTable *table) {
    if (!path || !table) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    FILE *in = fopen(path, "r");
    if (!in) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    memset(table, 0, sizeof(*table));
    table->e_min_GeV = DBL_MAX;
    table->z_min_m = DBL_MAX;
    table->e_max_GeV = -DBL_MAX;
    table->z_max_m = -DBL_MAX;

    char line[1024];
    if (!fgets(line, sizeof(line), in)) {
        fclose(in);
        return DUNE_STATUS_MISSING_INPUT;
    }

    while (fgets(line, sizeof(line), in)) {
        char *fields[6] = {0};
        int count = 0;
        char *token = strtok(line, ",");
        while (token && count < 6) {
            fields[count++] = token;
            token = strtok(NULL, ",");
        }
        if (count != 6) {
            continue;
        }

        DuneFluxFlavor flavor;
        if (!flavor_from_string(fields[0], &flavor)) {
            continue;
        }

        DuneDk2nuFluxZRow row;
        row.flavor = flavor;
        row.e_low_GeV = strtod(fields[1], NULL);
        row.e_high_GeV = strtod(fields[2], NULL);
        row.z_low_m = strtod(fields[3], NULL);
        row.z_high_m = strtod(fields[4], NULL);
        row.weight = strtod(fields[5], NULL);
        if (row.e_high_GeV <= row.e_low_GeV || row.z_high_m <= row.z_low_m || row.weight <= 0.0) {
            continue;
        }
        if (!ensure_capacity(table, table->n_rows + 1)) {
            fclose(in);
            dune_dk2nu_flux_z_free(table);
            return DUNE_STATUS_MISSING_INPUT;
        }
        table->rows[table->n_rows++] = row;
        if (row.e_low_GeV < table->e_min_GeV) table->e_min_GeV = row.e_low_GeV;
        if (row.e_high_GeV > table->e_max_GeV) table->e_max_GeV = row.e_high_GeV;
        if (row.z_low_m < table->z_min_m) table->z_min_m = row.z_low_m;
        if (row.z_high_m > table->z_max_m) table->z_max_m = row.z_high_m;
    }

    fclose(in);
    return table->n_rows > 0 ? DUNE_STATUS_OK : DUNE_STATUS_MISSING_INPUT;
}

void dune_dk2nu_flux_z_free(DuneDk2nuFluxZTable *table) {
    if (!table) return;
    free(table->rows);
    memset(table, 0, sizeof(*table));
}

double dune_dk2nu_flux_z_weight_sum(
    const DuneDk2nuFluxZTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV) {
    if (!table || !table->rows) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < table->n_rows; ++i) {
        const DuneDk2nuFluxZRow *row = &table->rows[i];
        if (row->flavor == flavor &&
            energy_GeV >= row->e_low_GeV &&
            energy_GeV < row->e_high_GeV) {
            sum += row->weight;
        }
    }
    return sum;
}
