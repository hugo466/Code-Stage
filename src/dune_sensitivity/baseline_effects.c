#include "dune_sensitivity/baseline_effects.h"

#include "dune/theory.h"

#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define DS_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define DS_MKDIR(path) mkdir(path, 0777)
#endif

#define DS_MAX_INDEX_ENTRIES 200000
#define DS_MAX_COMPONENT_ROWS DUNE_SENSITIVITY_MAX_BINS
#define DS_N_GROUPS_MAX 2048
#define DS_MAX_SHAPE_PULLS 512
#define DS_SHAPE_SIGMA 0.05

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int point_id;
    char model[32];
    char point_file[512];
    double dm41_eV2;
} SensitivityIndexEntry;

typedef struct {
    char detector[8];
    char panel[16];
    double e_rec_GeV;
    double asimov_events;
    double base_test_events;
    double pull_coeff[DUNE_SENSITIVITY_N_NORM_PULLS];
    int shape_id;
    double shape_coeff;
} SensitivityGroup;

typedef struct {
    const DuneSensitivitySpectrumBin *rows;
    int n_rows;
    double epsilon;
    int shape_enabled;
} PullObjectiveData;

typedef struct {
    SensitivityGroup groups[DS_N_GROUPS_MAX];
    int n_groups;
    char shape_panel[DS_MAX_SHAPE_PULLS][16];
    double shape_e_rec_GeV[DS_MAX_SHAPE_PULLS];
    int n_shape_pulls;
    double epsilon;
    int shape_enabled;
} PreparedPullObjectiveData;

typedef struct {
    double chi2_total;
    double chi2_stat;
    double chi2_norm_pulls;
    double chi2_shape_pulls;
    double chi2_prior;
    double chi2_detector_nd;
    double chi2_detector_fd;
    double chi2_rule[4];
} Chi2Breakdown;

static const double PULL_SIGMA[DUNE_SENSITIVITY_N_NORM_PULLS] = {
    0.01, 0.01,
    0.08, 0.08, 0.15, 0.15,
    0.004, 0.004, 0.02, 0.02,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.25, 0.25, 0.02, 0.02
};

static const char *PULL_NAME[DUNE_SENSITIVITY_N_NORM_PULLS] = {
    "ND_fiducial_volume",
    "FD_fiducial_volume",
    "flux_FHC_signal",
    "flux_RHC_signal",
    "flux_FHC_background",
    "flux_RHC_background",
    "ND_flux_FHC_signal",
    "ND_flux_RHC_signal",
    "ND_flux_FHC_background",
    "ND_flux_RHC_background",
    "xsec_nue_CC",
    "xsec_numu_CC",
    "xsec_nutau_CC",
    "xsec_nuebar_CC",
    "xsec_numubar_CC",
    "xsec_nutaubar_CC",
    "ND_xsec_nue_CC",
    "ND_xsec_numu_CC",
    "ND_xsec_nutau_CC",
    "ND_xsec_nuebar_CC",
    "ND_xsec_numubar_CC",
    "ND_xsec_nutaubar_CC",
    "xsec_nu_NC",
    "xsec_nubar_NC",
    "ND_xsec_nu_NC",
    "ND_xsec_nubar_NC"
};

static void make_parent_dirs(const char *path) {
    char buffer[1024];
    if (!path) return;
    strncpy(buffer, path, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    for (char *p = buffer; *p; ++p) {
        if (*p == '/' || *p == '\\') {
            const char saved = *p;
            *p = '\0';
            if (buffer[0] != '\0') {
                (void)DS_MKDIR(buffer);
            }
            *p = saved;
        }
    }
}

static void trim(char *s) {
    size_t len;
    if (!s) return;
    len = strlen(s);
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

static void build_relative_path(const char *index_csv, const char *point_file, char *out, int out_size) {
    if (!out || out_size <= 0) return;
    out[0] = '\0';
    if (!point_file) return;
    if (is_absolute_path_local(point_file)) {
        strncpy(out, point_file, (size_t)out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }
    const char *slash1 = strrchr(index_csv, '/');
    const char *slash2 = strrchr(index_csv, '\\');
    const char *slash = slash1;
    if (!slash || (slash2 && slash2 > slash)) slash = slash2;
    if (!slash) {
        strncpy(out, point_file, (size_t)out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }
    const size_t prefix_len = (size_t)(slash - index_csv + 1);
    if (prefix_len >= (size_t)out_size) return;
    memcpy(out, index_csv, prefix_len);
    out[prefix_len] = '\0';
    strncat(out, point_file, (size_t)out_size - strlen(out) - 1);
}

static int load_index_entries(const char *index_csv, SensitivityIndexEntry *entries, int max_entries, int *out_count) {
    FILE *in = fopen(index_csv, "r");
    if (!in) {
        fprintf(stderr, "DUNE sensitivity: impossible d'ouvrir l'index %s\n", index_csv);
        return 1;
    }
    char line[65536];
    char header[65536];
    char *header_fields[512] = {0};
    if (!fgets(header, sizeof(header), in)) {
        fclose(in);
        return 1;
    }
    const int n_header = split_csv_simple(header, header_fields, 512);
    int idx_point_id = -1;
    int idx_model = -1;
    int idx_point_file = -1;
    int idx_pmns_pass = -1;
    int idx_eta_pass = -1;
    int idx_dm41 = -1;
    for (int i = 0; i < n_header; ++i) {
        if (strcmp(header_fields[i], "point_id") == 0) idx_point_id = i;
        else if (strcmp(header_fields[i], "model") == 0) idx_model = i;
        else if (strcmp(header_fields[i], "point_file") == 0) idx_point_file = i;
        else if (strcmp(header_fields[i], "pmns_pass") == 0) idx_pmns_pass = i;
        else if (strcmp(header_fields[i], "eta_pass") == 0) idx_eta_pass = i;
        else if (strcmp(header_fields[i], "dm41_calc_eV2") == 0) idx_dm41 = i;
        else if (idx_dm41 < 0 && strcmp(header_fields[i], "dm41_eV2") == 0) idx_dm41 = i;
        else if (idx_dm41 < 0 && strcmp(header_fields[i], "dm41_target_eV2") == 0) idx_dm41 = i;
    }
    if (idx_point_id < 0) {
        fclose(in);
        fprintf(stderr, "DUNE sensitivity: colonne point_id absente dans %s\n", index_csv);
        return 1;
    }
    int count = 0;
    while (fgets(line, sizeof(line), in)) {
        char work[65536];
        char *fields[512] = {0};
        strncpy(work, line, sizeof(work) - 1);
        work[sizeof(work) - 1] = '\0';
        const int n = split_csv_simple(work, fields, 512);
        if (n <= idx_point_id) continue;
        if (idx_pmns_pass >= 0 && idx_pmns_pass < n && atoi(fields[idx_pmns_pass]) != 1) continue;
        if (idx_eta_pass >= 0 && idx_eta_pass < n && atoi(fields[idx_eta_pass]) != 1) continue;
        if (count >= max_entries) {
            fclose(in);
            return 2;
        }
        SensitivityIndexEntry *e = &entries[count++];
        memset(e, 0, sizeof(*e));
        e->point_id = atoi(fields[idx_point_id]);
        if (idx_model >= 0 && idx_model < n && fields[idx_model][0] != '\0') {
            strncpy(e->model, fields[idx_model], sizeof(e->model) - 1);
        } else {
            strncpy(e->model, "iss23", sizeof(e->model) - 1);
        }
        if (idx_point_file >= 0 && idx_point_file < n && fields[idx_point_file][0] != '\0') {
            strncpy(e->point_file, fields[idx_point_file], sizeof(e->point_file) - 1);
        } else {
            snprintf(e->point_file, sizeof(e->point_file), "%d.txt", e->point_id);
        }
        if (idx_dm41 >= 0 && idx_dm41 < n) e->dm41_eV2 = strtod(fields[idx_dm41], NULL);
    }
    fclose(in);
    *out_count = count;
    return 0;
}

static double cfg_or_default(double value, double fallback) {
    return value > 0.0 ? value : fallback;
}

static void build_standard_3nu_point(const SimulationConfig *cfg, DuneTheoryPoint *point) {
    const double deg = M_PI / 180.0;
    const double th12 = cfg_or_default(cfg->theta12_deg, 33.44) * deg;
    const double th13 = cfg_or_default(cfg->theta13_deg, 8.57) * deg;
    const double th23 = cfg_or_default(cfg->theta23_deg, 49.2) * deg;
    const double delta = cfg->delta_cp_deg * deg;
    const double c12 = cos(th12), s12 = sin(th12);
    const double c13 = cos(th13), s13 = sin(th13);
    const double c23 = cos(th23), s23 = sin(th23);
    const double complex eid = cos(delta) + I * sin(delta);
    const double complex emid = cos(delta) - I * sin(delta);

    memset(point, 0, sizeof(*point));
    point->point_id = 0;
    strncpy(point->model, "standard3nu", sizeof(point->model) - 1);
    point->n_light = 3;
    point->n_active = 3;
    point->dm21_eV2 = cfg_or_default(cfg->dm21_eV2, 7.42e-5);
    point->dm31_eV2 = cfg_or_default(cfg->dm31_eV2, 2.517e-3);
    point->light_masses_eV[0] = 0.0;
    point->light_masses_eV[1] = sqrt(point->dm21_eV2);
    point->light_masses_eV[2] = sqrt(point->dm31_eV2);

    point->mixing[0][0] = c12 * c13;
    point->mixing[0][1] = s12 * c13;
    point->mixing[0][2] = s13 * emid;
    point->mixing[1][0] = -s12 * c23 - c12 * s23 * s13 * eid;
    point->mixing[1][1] = c12 * c23 - s12 * s23 * s13 * eid;
    point->mixing[1][2] = s23 * c13;
    point->mixing[2][0] = s12 * s23 - c12 * c23 * s13 * eid;
    point->mixing[2][1] = -c12 * s23 - s12 * c23 * s13 * eid;
    point->mixing[2][2] = c23 * c13;
}

static void build_active_subblock_point(const DuneTheoryPoint *source, DuneTheoryPoint *point) {
    *point = *source;
    strncpy(point->model, "active3nu_subblock", sizeof(point->model) - 1);
    point->model[sizeof(point->model) - 1] = '\0';
    point->n_light = 3;
    point->n_active = 3;
    point->dm41_eV2 = 0.0;
    for (int i = 3; i < 8; ++i) point->light_masses_eV[i] = 0.0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            point->mixing[r][c] = (r < 3 && c < 3) ? source->mixing[r][c] : 0.0 + 0.0 * I;
        }
    }
}

static void build_analytic_3p1_point(
    const SimulationConfig *cfg,
    double dm41_eV2,
    double theta14_deg,
    double theta24_deg,
    DuneTheoryPoint *point) {
    const double deg = M_PI / 180.0;
    const double th12 = cfg_or_default(cfg->theta12_deg, 33.44) * deg;
    const double th13 = cfg_or_default(cfg->theta13_deg, 8.57) * deg;
    const double th23 = cfg_or_default(cfg->theta23_deg, 49.2) * deg;
    const double d13 = cfg->delta_cp_deg * deg;
    const double th14 = theta14_deg * deg;
    const double th24 = theta24_deg * deg;
    const double th34 = cfg->sensitivity_theta34_deg * deg;
    const double d24 = cfg->sensitivity_delta24_deg * deg;
    const double d34 = cfg->sensitivity_delta34_deg * deg;

    memset(point, 0, sizeof(*point));
    point->point_id = 0;
    strncpy(point->model, "analytic_3p1", sizeof(point->model) - 1);
    point->n_light = 4;
    point->n_active = 3;
    point->dm21_eV2 = cfg_or_default(cfg->dm21_eV2, 7.42e-5);
    point->dm31_eV2 = cfg_or_default(cfg->dm31_eV2, 2.517e-3);
    point->dm41_eV2 = dm41_eV2;
    point->light_masses_eV[0] = 0.0;
    point->light_masses_eV[1] = sqrt(point->dm21_eV2);
    point->light_masses_eV[2] = sqrt(point->dm31_eV2);
    point->light_masses_eV[3] = sqrt(dm41_eV2);

    const double c12 = cos(th12), s12 = sin(th12);
    const double c13 = cos(th13), s13 = sin(th13);
    const double c23 = cos(th23), s23 = sin(th23);
    const double c14 = cos(th14), s14 = sin(th14);
    const double c24 = cos(th24), s24 = sin(th24);
    const double c34 = cos(th34), s34 = sin(th34);
    const double complex eid13 = cexp(I * d13);
    const double complex emid13 = cexp(-I * d13);
    const double complex emid24 = cexp(-I * d24);
    const double complex emid34 = cexp(-I * d34);
    const double complex e_p_d24 = cexp(I * d24);
    const double complex e_p_d34 = cexp(I * d34);
    const double complex e_p_d24_m_d34 = cexp(I * (d24 - d34));
    const double complex e_m_d13_p_d24 = cexp(-I * (d13 + d24));
    const double complex e_m_d13_p_d34 = cexp(-I * (d13 + d34));

    point->mixing[0][0] = c12 * c13 * c14;
    point->mixing[0][1] = c13 * c14 * s12;
    point->mixing[0][2] = c14 * s13 * emid13;
    point->mixing[0][3] = s14;

    point->mixing[1][0] = -c23 * c24 * s12 - c12 * (c24 * s13 * s23 * eid13 + c13 * s14 * s24 * emid24);
    point->mixing[1][1] = c12 * c23 * c24 - s12 * (c24 * s13 * s23 * eid13 + c13 * s14 * s24 * emid24);
    point->mixing[1][2] = c13 * c24 * s23 - s13 * s14 * s24 * e_m_d13_p_d24;
    point->mixing[1][3] = c14 * s24 * emid24;

    point->mixing[2][0] =
        s12 * (c34 * s23 + c23 * s24 * s34 * e_p_d24_m_d34) +
        c12 * (s13 * (s23 * s24 * s34 * e_p_d24_m_d34 - c23 * c34) * eid13 -
               c13 * c24 * s14 * s34 * emid34);
    point->mixing[2][1] =
        -c12 * (c34 * s23 + c23 * s24 * s34 * e_p_d24_m_d34) +
        s12 * (s13 * (s23 * s24 * s34 * e_p_d24_m_d34 - c23 * c34) * eid13 -
               c13 * c24 * s14 * s34 * emid34);
    point->mixing[2][2] =
        -c24 * s13 * s14 * s34 * e_m_d13_p_d34 +
        c13 * (c23 * c34 - s23 * s24 * s34 * e_p_d24_m_d34);
    point->mixing[2][3] = c14 * c24 * s34 * emid34;

    point->mixing[3][0] =
        s12 * (c23 * c34 * s24 * e_p_d24 - s23 * s34 * e_p_d34) +
        c12 * (s13 * (c34 * s23 * s24 * e_p_d24 + c23 * s34 * e_p_d34) * eid13 -
               c13 * c24 * c34 * s14);
    point->mixing[3][1] =
        -c12 * (c23 * c34 * s24 * e_p_d24 - s23 * s34 * e_p_d34) +
        s12 * (s13 * (c34 * s23 * s24 * e_p_d24 + c23 * s34 * e_p_d34) * eid13 -
               c13 * c24 * c34 * s14);
    point->mixing[3][2] =
        -c24 * c34 * s13 * s14 * emid13 -
        c13 * (c34 * s23 * s24 * e_p_d24 + c23 * s34 * e_p_d34);
    point->mixing[3][3] = c14 * c24 * c34;
}

static int panel_rule(const char *panel) {
    if (strcmp(panel, "FHC_app") == 0) return 0;
    if (strcmp(panel, "RHC_app") == 0) return 1;
    if (strcmp(panel, "FHC_dis") == 0) return 2;
    if (strcmp(panel, "RHC_dis") == 0) return 3;
    return -1;
}

static int is_fhc_panel(const char *panel) {
    return strncmp(panel, "FHC", 3) == 0;
}

static int is_nd_detector(const char *detector) {
    return strcmp(detector, "ND") == 0;
}

static int is_signal_component(const char *component) {
    return strcmp(component, "signal") == 0;
}

static int is_background_component(const char *component) {
    return !is_signal_component(component);
}

static int is_nc_component(const char *component) {
    return strcmp(component, "nc") == 0;
}

static int component_uses_pull(const DuneSensitivitySpectrumBin *row, int pull) {
    const int nd = is_nd_detector(row->detector);
    const int fhc = is_fhc_panel(row->panel);
    const int signal = is_signal_component(row->component);
    const int background = is_background_component(row->component);
    const int nc = is_nc_component(row->component);
    const int app = strstr(row->panel, "_app") != NULL;
    const int rhc = !fhc;

    if (pull == 0) return nd;
    if (pull == 1) return !nd;
    if (pull == 2) return fhc && signal;
    if (pull == 3) return rhc && signal;
    if (pull == 4) return fhc && background;
    if (pull == 5) return rhc && background;
    if (pull == 6) return nd && fhc && signal;
    if (pull == 7) return nd && rhc && signal;
    if (pull == 8) return nd && fhc && background;
    if (pull == 9) return nd && rhc && background;

    if (pull == 22) return nc;
    if (pull == 23) return nc;
    if (pull == 24) return nd && nc;
    if (pull == 25) return nd && nc;
    if (nc) return 0;

    const int nd_xsec = pull >= 16 && pull <= 21;
    const int base_pull = nd_xsec ? pull - 6 : pull;
    if (nd_xsec && !nd) return 0;

    if (app && (strcmp(row->component, "signal") == 0 || strcmp(row->component, "beam") == 0)) {
        return base_pull == 10 || base_pull == 13;
    }
    if (app && strcmp(row->component, "numu") == 0) {
        return base_pull == 11 || base_pull == 14;
    }
    if (!app && strcmp(row->component, "signal") == 0) {
        return fhc ? base_pull == 11 : base_pull == 14;
    }
    if (!app && strcmp(row->component, "wrong_mu") == 0) {
        return fhc ? base_pull == 14 : base_pull == 11;
    }
    if (!app && strcmp(row->component, "tau") == 0) {
        return base_pull == 12 || base_pull == 15;
    }
    return 0;
}

static int component_uses_shape_pull(const DuneSensitivitySpectrumBin *row) {
    const int app = strstr(row->panel, "_app") != NULL;
    if (app) {
        return is_background_component(row->component);
    }
    return is_signal_component(row->component);
}

static int find_or_add_shape_pull(PreparedPullObjectiveData *prepared, const DuneSensitivitySpectrumBin *row) {
    for (int i = 0; i < prepared->n_shape_pulls; ++i) {
        if (strcmp(prepared->shape_panel[i], row->panel) == 0 &&
            fabs(prepared->shape_e_rec_GeV[i] - row->e_rec_GeV) < 1e-9) {
            return i;
        }
    }
    if (prepared->n_shape_pulls >= DS_MAX_SHAPE_PULLS) {
        return -1;
    }
    const int id = prepared->n_shape_pulls++;
    strncpy(prepared->shape_panel[id], row->panel, sizeof(prepared->shape_panel[id]) - 1);
    prepared->shape_e_rec_GeV[id] = row->e_rec_GeV;
    return id;
}

static int find_or_add_group(SensitivityGroup *groups, int *n_groups, const DuneSensitivitySpectrumBin *row) {
    for (int i = 0; i < *n_groups; ++i) {
        if (strcmp(groups[i].detector, row->detector) == 0 &&
            strcmp(groups[i].panel, row->panel) == 0 &&
            fabs(groups[i].e_rec_GeV - row->e_rec_GeV) < 1e-9) {
            return i;
        }
    }
    if (*n_groups >= DS_N_GROUPS_MAX) return -1;
    SensitivityGroup *g = &groups[(*n_groups)++];
    memset(g, 0, sizeof(*g));
    strncpy(g->detector, row->detector, sizeof(g->detector) - 1);
    strncpy(g->panel, row->panel, sizeof(g->panel) - 1);
    g->e_rec_GeV = row->e_rec_GeV;
    g->shape_id = -1;
    return *n_groups - 1;
}

static double poisson_term(double observed, double expected, double eps) {
    expected = expected > eps ? expected : eps;
    observed = observed > 0.0 ? observed : 0.0;
    if (observed <= 0.0) {
        return 2.0 * expected;
    }
    return 2.0 * (expected - observed + observed * log(observed / expected));
}

static int prepare_pull_objective(const PullObjectiveData *data, PreparedPullObjectiveData *prepared) {
    memset(prepared, 0, sizeof(*prepared));
    prepared->epsilon = data->epsilon;
    prepared->shape_enabled = data->shape_enabled;
    for (int i = 0; i < data->n_rows; ++i) {
        const DuneSensitivitySpectrumBin *row = &data->rows[i];
        const int group_id = find_or_add_group(prepared->groups, &prepared->n_groups, row);
        if (group_id < 0) {
            return 1;
        }
        prepared->groups[group_id].asimov_events += row->asimov_events;
        prepared->groups[group_id].base_test_events += row->test_events;
        for (int k = 0; k < DUNE_SENSITIVITY_N_NORM_PULLS; ++k) {
            if (component_uses_pull(row, k)) {
                prepared->groups[group_id].pull_coeff[k] += row->test_events;
            }
        }
        if (prepared->shape_enabled && component_uses_shape_pull(row)) {
            const int shape_id = find_or_add_shape_pull(prepared, row);
            if (shape_id < 0) {
                return 1;
            }
            prepared->groups[group_id].shape_id = shape_id;
            prepared->groups[group_id].shape_coeff += row->test_events;
        }
    }
    return 0;
}

static double group_expected(const SensitivityGroup *group, const PreparedPullObjectiveData *data, const double *zeta) {
    double expected = group->base_test_events;
    for (int k = 0; k < DUNE_SENSITIVITY_N_NORM_PULLS; ++k) {
        expected += zeta[k] * group->pull_coeff[k];
    }
    if (data->shape_enabled && group->shape_id >= 0) {
        expected += zeta[DUNE_SENSITIVITY_N_NORM_PULLS + group->shape_id] * group->shape_coeff;
    }
    return expected;
}

static double evaluate_prepared_objective(const double *zeta, const PreparedPullObjectiveData *data, Chi2Breakdown *breakdown) {
    Chi2Breakdown bd;
    memset(&bd, 0, sizeof(bd));
    for (int i = 0; i < data->n_groups; ++i) {
        const double expected = group_expected(&data->groups[i], data, zeta);
        if (!(expected > data->epsilon) || !isfinite(expected)) {
            return 1.0e300;
        }
        const double chi = poisson_term(data->groups[i].asimov_events, expected, data->epsilon);
        bd.chi2_stat += chi;
        if (strcmp(data->groups[i].detector, "ND") == 0) bd.chi2_detector_nd += chi;
        if (strcmp(data->groups[i].detector, "FD") == 0) bd.chi2_detector_fd += chi;
        {
            const int rule = panel_rule(data->groups[i].panel);
            if (rule >= 0 && rule < 4) bd.chi2_rule[rule] += chi;
        }
    }
    for (int k = 0; k < DUNE_SENSITIVITY_N_NORM_PULLS; ++k) {
        bd.chi2_norm_pulls += (zeta[k] * zeta[k]) / (PULL_SIGMA[k] * PULL_SIGMA[k]);
    }
    if (data->shape_enabled) {
        for (int s = 0; s < data->n_shape_pulls; ++s) {
            const double pull = zeta[DUNE_SENSITIVITY_N_NORM_PULLS + s];
            bd.chi2_shape_pulls += (pull * pull) / (DS_SHAPE_SIGMA * DS_SHAPE_SIGMA);
        }
    }
    bd.chi2_total = bd.chi2_stat + bd.chi2_norm_pulls + bd.chi2_shape_pulls + bd.chi2_prior;
    if (breakdown) *breakdown = bd;
    return bd.chi2_total;
}

static int solve_linear_system(double *a, double *b, double *x, int n) {
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double pivot_abs = fabs(a[k * n + k]);
        for (int r = k + 1; r < n; ++r) {
            const double v = fabs(a[r * n + k]);
            if (v > pivot_abs) {
                pivot = r;
                pivot_abs = v;
            }
        }
        if (pivot_abs < 1.0e-24) {
            return 1;
        }
        if (pivot != k) {
            for (int c = k; c < n; ++c) {
                const double tmp = a[k * n + c];
                a[k * n + c] = a[pivot * n + c];
                a[pivot * n + c] = tmp;
            }
            const double tmp_b = b[k];
            b[k] = b[pivot];
            b[pivot] = tmp_b;
        }
        const double diag = a[k * n + k];
        for (int c = k; c < n; ++c) {
            a[k * n + c] /= diag;
        }
        b[k] /= diag;
        for (int r = 0; r < n; ++r) {
            if (r == k) continue;
            const double factor = a[r * n + k];
            if (factor == 0.0) continue;
            for (int c = k; c < n; ++c) {
                a[r * n + c] -= factor * a[k * n + c];
            }
            b[r] -= factor * b[k];
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = b[i];
    }
    return 0;
}

static void prepared_gradient_hessian(
    const PreparedPullObjectiveData *data,
    const double *zeta,
    double *grad,
    double *hess,
    int n) {
    for (int k = 0; k < n; ++k) {
        const double sigma = k < DUNE_SENSITIVITY_N_NORM_PULLS ? PULL_SIGMA[k] : DS_SHAPE_SIGMA;
        grad[k] = 2.0 * zeta[k] / (sigma * sigma);
        for (int l = 0; l < n; ++l) hess[k * n + l] = 0.0;
        hess[k * n + k] = 2.0 / (sigma * sigma);
    }

    for (int i = 0; i < data->n_groups; ++i) {
        const SensitivityGroup *g = &data->groups[i];
        const double expected = fmax(data->epsilon, group_expected(g, data, zeta));
        const double observed = fmax(0.0, g->asimov_events);
        const double common_grad = 2.0 * (1.0 - observed / expected);
        const double common_hess = 2.0 * observed / (expected * expected);
        for (int k = 0; k < n; ++k) {
            double ak = 0.0;
            if (k < DUNE_SENSITIVITY_N_NORM_PULLS) {
                ak = g->pull_coeff[k];
            } else if (data->shape_enabled && g->shape_id == k - DUNE_SENSITIVITY_N_NORM_PULLS) {
                ak = g->shape_coeff;
            }
            if (ak == 0.0) continue;
            grad[k] += common_grad * ak;
            for (int l = 0; l < n; ++l) {
                double al = 0.0;
                if (l < DUNE_SENSITIVITY_N_NORM_PULLS) {
                    al = g->pull_coeff[l];
                } else if (data->shape_enabled && g->shape_id == l - DUNE_SENSITIVITY_N_NORM_PULLS) {
                    al = g->shape_coeff;
                }
                if (al == 0.0) continue;
                hess[k * n + l] += common_hess * ak * al;
            }
        }
    }
}

static int minimize_pulls_newton(
    const PullObjectiveData *raw_data,
    int max_iter,
    double tolerance,
    const double *initial_zeta,
    double *best_zeta,
    Chi2Breakdown *best_breakdown) {
    PreparedPullObjectiveData data;
    if (prepare_pull_objective(raw_data, &data) != 0) {
        return 1;
    }
    const int n = DUNE_SENSITIVITY_N_NORM_PULLS + (data.shape_enabled ? data.n_shape_pulls : 0);
    double *grad = (double *)calloc((size_t)n, sizeof(double));
    double *hess = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    double *rhs = (double *)calloc((size_t)n, sizeof(double));
    double *step = (double *)calloc((size_t)n, sizeof(double));
    double *trial_zeta = (double *)calloc((size_t)n, sizeof(double));
    if (!grad || !hess || !rhs || !step || !trial_zeta) {
        free(grad); free(hess); free(rhs); free(step); free(trial_zeta);
        return 1;
    }
    for (int k = 0; k < n; ++k) {
        best_zeta[k] = initial_zeta ? initial_zeta[k] : 0.0;
    }
    double current = evaluate_prepared_objective(best_zeta, &data, NULL);
    int status = 2;

    for (int iter = 0; iter < max_iter; ++iter) {
        prepared_gradient_hessian(&data, best_zeta, grad, hess, n);
        double grad_norm = 0.0;
        for (int k = 0; k < n; ++k) {
            grad_norm += grad[k] * grad[k];
            rhs[k] = -grad[k];
            step[k] = 0.0;
        }
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tolerance) {
            status = 0;
            break;
        }
        if (solve_linear_system(hess, rhs, step, n) != 0) {
            status = 1;
            break;
        }
        double step_norm = 0.0;
        for (int k = 0; k < n; ++k) step_norm += step[k] * step[k];
        step_norm = sqrt(step_norm);
        if (step_norm < tolerance) {
            status = 0;
            break;
        }

        int accepted = 0;
        double alpha = 1.0;
        for (int ls = 0; ls < 20; ++ls) {
            for (int k = 0; k < n; ++k) {
                trial_zeta[k] = best_zeta[k] + alpha * step[k];
            }
            const double trial = evaluate_prepared_objective(trial_zeta, &data, NULL);
            if (isfinite(trial) && trial < current) {
                memcpy(best_zeta, trial_zeta, (size_t)n * sizeof(double));
                current = trial;
                accepted = 1;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            status = 0;
            break;
        }
        if (fabs(alpha * step_norm) < tolerance * (1.0 + sqrt(current))) {
            status = 0;
            break;
        }
    }

    evaluate_prepared_objective(best_zeta, &data, best_breakdown);
    free(grad); free(hess); free(rhs); free(step); free(trial_zeta);
    return status;
}

static int build_spectrum_rows(
    const SimulationConfig *cfg,
    const DuneTheoryPoint *asimov,
    const DuneTheoryPoint *test,
    DuneSensitivitySpectrumBin *rows,
    int max_rows,
    int *out_count) {
    int n = 0;
    const int use_nd = strstr(cfg->sensitivity_detector_mode, "ND") != NULL || strcmp(cfg->sensitivity_detector_mode, "ND+FD") == 0;
    const int use_fd = strstr(cfg->sensitivity_detector_mode, "FD") != NULL || strcmp(cfg->sensitivity_detector_mode, "ND+FD") == 0;
    if (use_nd) {
        int added = 0;
        if (dune_nd_fig4_build_sensitivity_rows(cfg, asimov, test, rows + n, max_rows - n, &added) != 0) {
            return 1;
        }
        n += added;
    }
    if (use_fd) {
        int added = 0;
        if (dune_fd_fig4_build_sensitivity_rows(cfg, asimov, test, rows + n, max_rows - n, &added) != 0) {
            return 1;
        }
        n += added;
    }
    *out_count = n;
    return n > 0 ? 0 : 1;
}

static double safe_abs2(double complex z) {
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

static void write_output_header(FILE *out) {
    fprintf(out,
            "point_id,backend,asimov_mode,detector_mode,source_model,dm41_eV2,"
            "sin2_theta14_eff,sin2_theta24_eff,theta14_rad_eff,theta24_rad_eff,"
            "chi2_min,chi2_stat,chi2_prior,chi2_pulls_norm,chi2_pulls_shape,"
            "chi2_ND,chi2_FD,chi2_FHC_app,chi2_RHC_app,chi2_FHC_dis,chi2_RHC_dis,"
            "n_component_rows,minimizer_status");
    for (int k = 0; k < DUNE_SENSITIVITY_N_NORM_PULLS; ++k) {
        fprintf(out, ",pull_%02d_%s", k + 1, PULL_NAME[k]);
    }
    fprintf(out, "\n");
}

int run_dune_baseline_effects_sensitivity(const SimulationConfig *cfg) {
    if (!cfg) {
        return 1;
    }
    const int backend_iss23 = strcmp(cfg->sensitivity_test_backend, "iss23_points") == 0;
    const int backend_analytic = strcmp(cfg->sensitivity_test_backend, "analytic_3p1") == 0;
    if (!backend_iss23 && !backend_analytic) {
        fprintf(stderr,
                "DUNE sensitivity: backend '%s' non implemente "
                "(utilise sensitivity.test_backend = iss23_points ou analytic_3p1).\n",
                cfg->sensitivity_test_backend);
        return 1;
    }
    SensitivityIndexEntry *entries = NULL;
    int n_entries = 0;
    if (backend_iss23) {
        entries = (SensitivityIndexEntry *)calloc(DS_MAX_INDEX_ENTRIES, sizeof(*entries));
        if (!entries) return 1;
        if (load_index_entries(cfg->sensitivity_points_index_csv, entries, DS_MAX_INDEX_ENTRIES, &n_entries) != 0 || n_entries <= 0) {
            free(entries);
            return 1;
        }
    }
    const int analytic_theta_points =
        strcmp(cfg->sensitivity_scan_plane, "theta14_dm41") == 0
            ? cfg->sensitivity_theta14_points
            : cfg->sensitivity_theta24_points;
    const int analytic_total = cfg->sensitivity_dm41_points * analytic_theta_points;
    const int n_available = backend_analytic ? analytic_total : n_entries;
    int point_offset = cfg->sensitivity_point_offset;
    if (point_offset < 0) point_offset = 0;
    if (point_offset > n_available) point_offset = n_available;
    const int n_remaining = n_available - point_offset;
    const int n_to_run = (cfg->sensitivity_max_points > 0 && cfg->sensitivity_max_points < n_remaining)
                             ? cfg->sensitivity_max_points
                             : n_remaining;

    make_parent_dirs(cfg->sensitivity_output_csv);
    FILE *out = fopen(cfg->sensitivity_output_csv, "w");
    if (!out) {
        fprintf(stderr, "DUNE sensitivity: impossible d'ouvrir %s\n", cfg->sensitivity_output_csv);
        free(entries);
        return 1;
    }
    write_output_header(out);

    DuneSensitivitySpectrumBin *rows = (DuneSensitivitySpectrumBin *)calloc(DS_MAX_COMPONENT_ROWS, sizeof(*rows));
    if (!rows) {
        fclose(out);
        free(entries);
        return 1;
    }

    fprintf(stderr,
            "DUNE sensitivity: %d points, mode=%s, Asimov=%s, source=%s, norm pulls=26, shape=%s\n",
            n_to_run,
            cfg->sensitivity_detector_mode,
            cfg->sensitivity_asimov_mode,
            cfg->sensitivity_source_model,
            cfg->sensitivity_shape_systematics_enabled ? "on" : "off");
    if (cfg->sensitivity_priors_enabled) {
        fprintf(stderr, "DUNE sensitivity: attention, chi2_prior structurellement present mais les priors oscillation/densite ne sont pas encore marginalises.\n");
    }
    const time_t run_start_time = time(NULL);
    double warm_start_zeta[DUNE_SENSITIVITY_N_NORM_PULLS + DS_MAX_SHAPE_PULLS] = {0.0};
    int warm_start_valid = 0;

    for (int ip = 0; ip < n_to_run; ++ip) {
        const int global_ip = point_offset + ip;
        char point_path[1024];
        DuneTheoryPoint test_point;
        DuneTheoryPoint asimov_point;
        memset(&test_point, 0, sizeof(test_point));
        if (backend_iss23) {
            build_relative_path(cfg->sensitivity_points_index_csv, entries[global_ip].point_file, point_path, (int)sizeof(point_path));
            if (dune_iss23_read_point(point_path, &test_point) != DUNE_STATUS_OK) {
                fprintf(stderr, "DUNE sensitivity: point %d illisible: %s\n", entries[global_ip].point_id, point_path);
                continue;
            }
            if (test_point.point_id == 0) test_point.point_id = entries[global_ip].point_id;
        } else {
            const int idm = global_ip / analytic_theta_points;
            const int ith = global_ip % analytic_theta_points;
            double dm41 = cfg->sensitivity_dm41_min_eV2;
            if (cfg->sensitivity_dm41_points > 1) {
                const double t = (double)idm / (double)(cfg->sensitivity_dm41_points - 1);
                if (cfg->sensitivity_dm41_logspace) {
                    dm41 = exp(log(cfg->sensitivity_dm41_min_eV2) +
                               t * log(cfg->sensitivity_dm41_max_eV2 / cfg->sensitivity_dm41_min_eV2));
                } else {
                    dm41 = cfg->sensitivity_dm41_min_eV2 +
                           t * (cfg->sensitivity_dm41_max_eV2 - cfg->sensitivity_dm41_min_eV2);
                }
            }
            double theta14 = cfg->sensitivity_theta14_min_deg;
            double theta24 = cfg->sensitivity_theta24_min_deg;
            const double t = analytic_theta_points > 1 ? (double)ith / (double)(analytic_theta_points - 1) : 0.0;
            if (strcmp(cfg->sensitivity_scan_plane, "theta14_dm41") == 0) {
                if (cfg->sensitivity_theta14_logspace) {
                    const double sin2 = exp(log(cfg->sensitivity_sin2_theta14_min) +
                                            t * log(cfg->sensitivity_sin2_theta14_max / cfg->sensitivity_sin2_theta14_min));
                    theta14 = asin(sqrt(fmin(1.0, fmax(0.0, sin2)))) * 180.0 / M_PI;
                } else {
                    theta14 = cfg->sensitivity_theta14_min_deg +
                              t * (cfg->sensitivity_theta14_max_deg - cfg->sensitivity_theta14_min_deg);
                }
            } else {
                if (cfg->sensitivity_theta24_logspace) {
                    const double sin2 = exp(log(cfg->sensitivity_sin2_theta24_min) +
                                            t * log(cfg->sensitivity_sin2_theta24_max / cfg->sensitivity_sin2_theta24_min));
                    theta24 = asin(sqrt(fmin(1.0, fmax(0.0, sin2)))) * 180.0 / M_PI;
                } else {
                    theta24 = cfg->sensitivity_theta24_min_deg +
                              t * (cfg->sensitivity_theta24_max_deg - cfg->sensitivity_theta24_min_deg);
                }
            }
            build_analytic_3p1_point(cfg, dm41, theta14, theta24, &test_point);
            test_point.point_id = global_ip + 1;
        }
        if (strcmp(cfg->sensitivity_asimov_mode, "active_subblock") == 0) {
            build_active_subblock_point(&test_point, &asimov_point);
        } else if (strcmp(cfg->sensitivity_asimov_mode, "standard3nu") == 0) {
            build_standard_3nu_point(cfg, &asimov_point);
        } else {
            fprintf(stderr, "DUNE sensitivity: asimov_mode '%s' inconnu\n", cfg->sensitivity_asimov_mode);
            fclose(out);
            free(rows);
            free(entries);
            return 1;
        }

        int n_rows = 0;
        if (build_spectrum_rows(cfg, &asimov_point, &test_point, rows, DS_MAX_COMPONENT_ROWS, &n_rows) != 0) {
            fprintf(stderr, "DUNE sensitivity: spectres impossibles pour point %d\n", test_point.point_id);
            continue;
        }

        PullObjectiveData data;
        data.rows = rows;
        data.n_rows = n_rows;
        data.epsilon = cfg->sensitivity_poisson_epsilon;
        data.shape_enabled = cfg->sensitivity_shape_systematics_enabled != 0;
        double best_zeta[DUNE_SENSITIVITY_N_NORM_PULLS + DS_MAX_SHAPE_PULLS] = {0.0};
        Chi2Breakdown bd;
        memset(&bd, 0, sizeof(bd));
        const int min_status = minimize_pulls_newton(
            &data,
            cfg->sensitivity_minimizer_max_iter,
            cfg->sensitivity_minimizer_tolerance,
            warm_start_valid ? warm_start_zeta : NULL,
            best_zeta,
            &bd);
        memcpy(warm_start_zeta, best_zeta, sizeof(warm_start_zeta));
        warm_start_valid = 1;

        const double ue4_abs2 = safe_abs2(test_point.mixing[0][3]);
        const double umu4_abs2 = safe_abs2(test_point.mixing[1][3]);
        const double sin2_t14 = ue4_abs2;
        const double denom = fmax(1.0e-15, 1.0 - ue4_abs2);
        double sin2_t24 = umu4_abs2 / denom;
        if (sin2_t24 < 0.0) sin2_t24 = 0.0;
        if (sin2_t24 > 1.0) sin2_t24 = 1.0;
        const double theta14 = asin(sqrt(fmin(1.0, fmax(0.0, sin2_t14))));
        const double theta24 = asin(sqrt(sin2_t24));

        fprintf(out,
                "%d,%s,%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,"
                "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d",
                test_point.point_id,
                cfg->sensitivity_test_backend,
                cfg->sensitivity_asimov_mode,
                cfg->sensitivity_detector_mode,
                cfg->sensitivity_source_model,
                test_point.dm41_eV2,
                sin2_t14,
                sin2_t24,
                theta14,
                theta24,
                bd.chi2_total,
                bd.chi2_stat,
                bd.chi2_prior,
                bd.chi2_norm_pulls,
                bd.chi2_shape_pulls,
                bd.chi2_detector_nd,
                bd.chi2_detector_fd,
                bd.chi2_rule[0],
                bd.chi2_rule[1],
                bd.chi2_rule[2],
                bd.chi2_rule[3],
                n_rows,
                min_status);
        for (int k = 0; k < DUNE_SENSITIVITY_N_NORM_PULLS; ++k) {
            fprintf(out, ",%.12g", best_zeta[k]);
        }
        fprintf(out, "\n");

        if ((ip + 1) % 10 == 0 || ip + 1 == n_to_run) {
            const double elapsed = fmax(1.0, difftime(time(NULL), run_start_time));
            const double points_per_sec = (double)(ip + 1) / elapsed;
            const double eta_sec = points_per_sec > 0.0 ? (double)(n_to_run - (ip + 1)) / points_per_sec : 0.0;
            fprintf(stderr,
                    "  DUNE sensitivity: %d/%d points (%.1f%%), elapsed=%.0fs, ETA=%.0fs\n",
                    ip + 1,
                    n_to_run,
                    100.0 * (double)(ip + 1) / (double)n_to_run,
                    elapsed,
                    eta_sec);
        }
    }

    fclose(out);
    free(rows);
    free(entries);
    fprintf(stderr, "DUNE sensitivity: resultats -> %s\n", cfg->sensitivity_output_csv);
    return 0;
}
