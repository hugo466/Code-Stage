#include "dune/csv_writer.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define DUNE_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define DUNE_MKDIR(path) mkdir(path, 0777)
#endif

static void make_parent_dirs(const char *path) {
    char buffer[512];
    strncpy(buffer, path, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    for (char *p = buffer; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\') {
            const char saved = *p;
            *p = '\0';
            if (buffer[0] != '\0') {
                (void)DUNE_MKDIR(buffer);
            }
            *p = saved;
        }
    }
}

DuneStatus dune_csv_writer_write_prediction(const char *path, const DuneSamplePrediction *prediction) {
    if (!path || !prediction) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    make_parent_dirs(path);
    FILE *out = fopen(path, "w");
    if (!out) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    fprintf(out, "bin,E_low_GeV,E_high_GeV,mu_like,e_like,null_mu_like,null_e_like\n");
    for (int i = 0; i < prediction->n_bins; ++i) {
        fprintf(out, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                i,
                prediction->bin_low_GeV[i],
                prediction->bin_high_GeV[i],
                prediction->mu_like[i],
                prediction->e_like[i],
                prediction->null_mu_like[i],
                prediction->null_e_like[i]);
    }
    fclose(out);
    return DUNE_STATUS_OK;
}

DuneStatus dune_csv_writer_write_null(const char *path, const DuneSamplePrediction *prediction) {
    if (!path || !prediction) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    make_parent_dirs(path);
    FILE *out = fopen(path, "w");
    if (!out) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    fprintf(out, "bin,E_low_GeV,E_high_GeV,null_mu_like,null_e_like\n");
    for (int i = 0; i < prediction->n_bins; ++i) {
        fprintf(out, "%d,%.10e,%.10e,%.10e,%.10e\n",
                i,
                prediction->bin_low_GeV[i],
                prediction->bin_high_GeV[i],
                prediction->null_mu_like[i],
                prediction->null_e_like[i]);
    }
    fclose(out);
    return DUNE_STATUS_OK;
}

DuneStatus dune_csv_writer_write_residuals(const char *path, const DuneSamplePrediction *prediction) {
    if (!path || !prediction) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    make_parent_dirs(path);
    FILE *out = fopen(path, "w");
    if (!out) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    fprintf(out, "bin,E_low_GeV,E_high_GeV,residual_mu_like,residual_e_like\n");
    for (int i = 0; i < prediction->n_bins; ++i) {
        fprintf(out, "%d,%.10e,%.10e,%.10e,%.10e\n",
                i,
                prediction->bin_low_GeV[i],
                prediction->bin_high_GeV[i],
                prediction->mu_like[i] - prediction->null_mu_like[i],
                prediction->e_like[i] - prediction->null_e_like[i]);
    }
    fclose(out);
    return DUNE_STATUS_OK;
}

DuneStatus dune_csv_writer_write_point_observables(const char *path, const DuneTheoryPoint *point) {
    if (!path || !point) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    double max_eta_abs = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (point->eta_abs_3x3[i][j] > max_eta_abs) {
                max_eta_abs = point->eta_abs_3x3[i][j];
            }
        }
    }

    make_parent_dirs(path);
    FILE *out = fopen(path, "w");
    if (!out) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    fprintf(out, "point_id,n_light,n_active,dm21_eV2,dm31_eV2,dm41_eV2,max_abs_eta\n");
    fprintf(out, "%d,%d,%d,%.10e,%.10e,%.10e,%.10e\n",
            point->point_id,
            point->n_light,
            point->n_active,
            point->dm21_eV2,
            point->dm31_eV2,
            point->dm41_eV2,
            max_eta_abs);
    fclose(out);
    return DUNE_STATUS_OK;
}
