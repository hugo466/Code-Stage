#include "dune/theory.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static const char *value_after_equals(const char *line) {
    const char *eq = strchr(line, '=');
    return eq ? eq + 1 : NULL;
}

static void trim_left(const char **s) {
    while (**s && isspace((unsigned char)**s)) {
        ++(*s);
    }
}

static int parse_matrix_row4(const char *line, double row[4]) {
    const char *p = strchr(line, '[');
    if (!p) {
        return 0;
    }
    ++p;
    for (int i = 0; i < 4; ++i) {
        char *end = NULL;
        trim_left(&p);
        row[i] = strtod(p, &end);
        if (end == p) {
            return 0;
        }
        p = end;
        while (*p == ',' || isspace((unsigned char)*p)) {
            ++p;
        }
    }
    return 1;
}

static int parse_eta_row(const char *line, double eta[3][3]) {
    const char *p = strchr(line, '[');
    if (!p) {
        return 0;
    }
    ++p;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            char *end = NULL;
            trim_left(&p);
            if (starts_with(p, "nan")) {
                eta[r][c] = NAN;
                p += 3;
            } else {
                eta[r][c] = strtod(p, &end);
                if (end == p) {
                    return 0;
                }
                p = end;
            }
            while (*p == ',' || *p == ';' || isspace((unsigned char)*p)) {
                ++p;
            }
        }
    }
    return 1;
}

static void trim(char *s) {
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }
    size_t start = 0;
    while (isspace((unsigned char)s[start])) {
        ++start;
    }
    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }
}

static int split_csv_simple(char *line, char *fields[], int max_fields) {
    int n = 0;
    char *token = strtok(line, ",");
    while (token && n < max_fields) {
        trim(token);
        fields[n++] = token;
        token = strtok(NULL, ",");
    }
    return n;
}

static int is_absolute_path_local(const char *path) {
    return path && ((isalpha((unsigned char)path[0]) && path[1] == ':') || path[0] == '/' || path[0] == '\\');
}

static void build_relative_to_index(const char *index_csv, const char *point_file, char *out, int out_size) {
    if (!out || out_size <= 0) {
        return;
    }
    out[0] = '\0';
    if (!point_file) {
        return;
    }
    if (is_absolute_path_local(point_file)) {
        strncpy(out, point_file, (size_t)out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    const char *slash1 = strrchr(index_csv, '/');
    const char *slash2 = strrchr(index_csv, '\\');
    const char *slash = slash1;
    if (!slash || (slash2 && slash2 > slash)) {
        slash = slash2;
    }
    if (!slash) {
        strncpy(out, point_file, (size_t)out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    const size_t prefix_len = (size_t)(slash - index_csv + 1);
    if (prefix_len >= (size_t)out_size) {
        return;
    }
    memcpy(out, index_csv, prefix_len);
    out[prefix_len] = '\0';
    strncat(out, point_file, (size_t)out_size - strlen(out) - 1);
}

DuneStatus dune_iss23_read_point(const char *path, DuneTheoryPoint *point) {
    if (!path || !point) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    FILE *in = fopen(path, "r");
    if (!in) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    memset(point, 0, sizeof(*point));
    strncpy(point->model, "iss23", sizeof(point->model) - 1);
    point->n_active = 3;
    point->n_light = 4;

    char line[2048];
    int read_solver = 0;
    int read_constructed = 0;
    while (fgets(line, sizeof(line), in) != NULL) {
        if (starts_with(line, "point_id")) {
            const char *v = value_after_equals(line);
            if (v) point->point_id = atoi(v);
        } else if (starts_with(line, "dm21_calc_eV2")) {
            const char *v = value_after_equals(line);
            if (v) point->dm21_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "dm31_calc_eV2")) {
            const char *v = value_after_equals(line);
            if (v) point->dm31_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "dm41_calc_eV2")) {
            const char *v = value_after_equals(line);
            if (v) point->dm41_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "dm21_target_eV2") && point->dm21_eV2 <= 0.0) {
            const char *v = value_after_equals(line);
            if (v) point->dm21_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "dm31_target_eV2") && point->dm31_eV2 <= 0.0) {
            const char *v = value_after_equals(line);
            if (v) point->dm31_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "dm41_target_eV2") && point->dm41_eV2 <= 0.0) {
            const char *v = value_after_equals(line);
            if (v) point->dm41_eV2 = strtod(v, NULL);
        } else if (starts_with(line, "eta_abs_3x3")) {
            (void)parse_eta_row(line, point->eta_abs_3x3);
        } else if (starts_with(line, "U4x4_solver")) {
            for (int r = 0; r < 4; ++r) {
                double row[4] = {0.0};
                if (!fgets(line, sizeof(line), in) || !parse_matrix_row4(line, row)) {
                    fclose(in);
                    return DUNE_STATUS_MISSING_INPUT;
                }
                for (int c = 0; c < 4; ++c) {
                    point->mixing[r][c] = row[c] + 0.0 * I;
                }
            }
            read_solver = 1;
        } else if (starts_with(line, "U4x4_constructed") && !read_solver) {
            for (int r = 0; r < 4; ++r) {
                double row[4] = {0.0};
                if (!fgets(line, sizeof(line), in) || !parse_matrix_row4(line, row)) {
                    fclose(in);
                    return DUNE_STATUS_MISSING_INPUT;
                }
                for (int c = 0; c < 4; ++c) {
                    point->mixing[r][c] = row[c] + 0.0 * I;
                }
            }
            read_constructed = 1;
        }
    }

    fclose(in);
    point->light_masses_eV[0] = 0.0;
    point->light_masses_eV[1] = point->dm21_eV2 > 0.0 ? sqrt(point->dm21_eV2) : 0.0;
    point->light_masses_eV[2] = point->dm31_eV2 > 0.0 ? sqrt(point->dm31_eV2) : 0.0;
    point->light_masses_eV[3] = point->dm41_eV2 > 0.0 ? sqrt(point->dm41_eV2) : 0.0;
    return (point->dm41_eV2 > 0.0 && (read_solver || read_constructed))
               ? DUNE_STATUS_OK
               : DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_theory_index_find_entry(const char *index_csv, int point_id, DuneTheoryIndexEntry *entry) {
    if (!index_csv || !entry) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    FILE *in = fopen(index_csv, "r");
    if (!in) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    char line[2048];
    while (fgets(line, sizeof(line), in) != NULL) {
        char work[2048];
        char *fields[16] = {0};
        strncpy(work, line, sizeof(work) - 1);
        work[sizeof(work) - 1] = '\0';

        const int n = split_csv_simple(work, fields, 16);
        if (n < 3) {
            continue;
        }
        if (strcmp(fields[0], "point_id") == 0) {
            continue;
        }

        const int id = atoi(fields[0]);
        if (id != point_id) {
            continue;
        }

        memset(entry, 0, sizeof(*entry));
        entry->point_id = id;
        strncpy(entry->model, fields[1], sizeof(entry->model) - 1);
        strncpy(entry->point_file, fields[2], sizeof(entry->point_file) - 1);
        if (n > 3) entry->n_light = atoi(fields[3]);
        if (n > 4) entry->n_heavy = atoi(fields[4]);
        if (n > 5) strncpy(entry->mode_hint, fields[5], sizeof(entry->mode_hint) - 1);
        if (n > 6) entry->max_abs_eta = strtod(fields[6], NULL);
        if (n > 7) entry->dm41_eV2 = strtod(fields[7], NULL);
        if (n > 8) entry->dm51_eV2 = strtod(fields[8], NULL);
        fclose(in);
        return DUNE_STATUS_OK;
    }

    fclose(in);
    return DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_theory_index_find_point(const char *index_csv, int point_id, char *path, int path_size) {
    if (!index_csv || !path || path_size <= 0) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    DuneTheoryIndexEntry entry;
    DuneStatus status = dune_theory_index_find_entry(index_csv, point_id, &entry);
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    build_relative_to_index(index_csv, entry.point_file, path, path_size);
    return path[0] != '\0' ? DUNE_STATUS_OK : DUNE_STATUS_MISSING_INPUT;
}

DuneStatus dune_theory_point_load(DuneRunContext *ctx) {
    if (!ctx) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }
    if (!ctx->cfg || ctx->cfg->dune_theory_index_csv[0] == '\0' || ctx->cfg->dune_point_id <= 0) {
        return DUNE_STATUS_MISSING_INPUT;
    }

    DuneTheoryIndexEntry entry;
    DuneStatus status = dune_theory_index_find_entry(
        ctx->cfg->dune_theory_index_csv,
        ctx->cfg->dune_point_id,
        &entry);
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    char point_path[512];
    status = dune_theory_index_find_point(
        ctx->cfg->dune_theory_index_csv,
        ctx->cfg->dune_point_id,
        point_path,
        (int)sizeof(point_path));
    if (status != DUNE_STATUS_OK) {
        return status;
    }

    const char *model = ctx->cfg->dune_theory_model[0] != '\0' ? ctx->cfg->dune_theory_model : entry.model;
    if (strcmp(model, "iss23") == 0) {
        status = dune_iss23_read_point(point_path, &ctx->theory);
    } else {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    if (status == DUNE_STATUS_OK && ctx->theory.point_id == 0) {
        ctx->theory.point_id = ctx->cfg->dune_point_id;
    }
    return status;
}
