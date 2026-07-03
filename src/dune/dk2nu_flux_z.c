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

static int ensure_energy_bin_capacity(DuneDk2nuFluxZTable *table, int need) {
    if (need <= table->energy_bin_capacity) {
        return 1;
    }
    int next = table->energy_bin_capacity > 0 ? table->energy_bin_capacity * 2 : 128;
    while (next < need) next *= 2;
    DuneDk2nuFluxZEnergyBin *bins =
        (DuneDk2nuFluxZEnergyBin *)realloc(table->energy_bins, (size_t)next * sizeof(*bins));
    if (!bins) {
        return 0;
    }
    table->energy_bins = bins;
    table->energy_bin_capacity = next;
    return 1;
}

static int compare_rows_by_flavor_energy_z(const void *lhs, const void *rhs) {
    const DuneDk2nuFluxZRow *a = (const DuneDk2nuFluxZRow *)lhs;
    const DuneDk2nuFluxZRow *b = (const DuneDk2nuFluxZRow *)rhs;
    if (a->flavor != b->flavor) {
        return (int)a->flavor - (int)b->flavor;
    }
    if (a->e_low_GeV < b->e_low_GeV) return -1;
    if (a->e_low_GeV > b->e_low_GeV) return 1;
    if (a->e_high_GeV < b->e_high_GeV) return -1;
    if (a->e_high_GeV > b->e_high_GeV) return 1;
    if (a->z_low_m < b->z_low_m) return -1;
    if (a->z_low_m > b->z_low_m) return 1;
    if (a->z_high_m < b->z_high_m) return -1;
    if (a->z_high_m > b->z_high_m) return 1;
    return 0;
}

static int build_energy_bins(DuneDk2nuFluxZTable *table) {
    table->n_energy_bins = 0;
    if (!table->rows || table->n_rows <= 0) {
        return 0;
    }

    qsort(table->rows,
          (size_t)table->n_rows,
          sizeof(table->rows[0]),
          compare_rows_by_flavor_energy_z);

    int first = 0;
    while (first < table->n_rows) {
        const DuneDk2nuFluxZRow *ref = &table->rows[first];
        int last = first + 1;
        double weight_sum = ref->weight;
        while (last < table->n_rows) {
            const DuneDk2nuFluxZRow *row = &table->rows[last];
            if (row->flavor != ref->flavor ||
                row->e_low_GeV != ref->e_low_GeV ||
                row->e_high_GeV != ref->e_high_GeV) {
                break;
            }
            weight_sum += row->weight;
            ++last;
        }

        if (!ensure_energy_bin_capacity(table, table->n_energy_bins + 1)) {
            return 0;
        }
        DuneDk2nuFluxZEnergyBin *bin = &table->energy_bins[table->n_energy_bins++];
        bin->flavor = ref->flavor;
        bin->e_low_GeV = ref->e_low_GeV;
        bin->e_high_GeV = ref->e_high_GeV;
        bin->first_row = first;
        bin->n_rows = last - first;
        bin->weight_sum = weight_sum;
        first = last;
    }
    return table->n_energy_bins > 0;
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
    if (table->n_rows <= 0 || !build_energy_bins(table)) {
        dune_dk2nu_flux_z_free(table);
        return DUNE_STATUS_MISSING_INPUT;
    }
    return DUNE_STATUS_OK;
}

void dune_dk2nu_flux_z_free(DuneDk2nuFluxZTable *table) {
    if (!table) return;
    free(table->rows);
    free(table->energy_bins);
    memset(table, 0, sizeof(*table));
}

double dune_dk2nu_flux_z_weight_sum(
    const DuneDk2nuFluxZTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV) {
    double sum = 0.0;
    if (!dune_dk2nu_flux_z_find_energy_bin(table, flavor, energy_GeV, NULL, NULL, &sum)) {
        return 0.0;
    }
    return sum;
}

int dune_dk2nu_flux_z_find_energy_bin(
    const DuneDk2nuFluxZTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV,
    int *first_row,
    int *n_rows,
    double *weight_sum) {
    if (!table || !table->energy_bins || !table->rows) {
        return 0;
    }
    for (int i = 0; i < table->n_energy_bins; ++i) {
        const DuneDk2nuFluxZEnergyBin *bin = &table->energy_bins[i];
        if (bin->flavor == flavor &&
            energy_GeV >= bin->e_low_GeV &&
            energy_GeV < bin->e_high_GeV) {
            if (first_row) *first_row = bin->first_row;
            if (n_rows) *n_rows = bin->n_rows;
            if (weight_sum) *weight_sum = bin->weight_sum;
            return 1;
        }
    }
    return 0;
}
