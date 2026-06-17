#include "scan.h"

#include "dune/dk2nu_flux_z.h"
#include "dune/flux.h"
#include "dune/dune.h"
#include "inverse_seesaw/oscillation.h"
#include "dune/xsec.h"
#include "dune/theory.h"

#include <ctype.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIG4_INPUT_CSV "data/dune/validation/fig4_reconstructed_energy_spectra.csv"
#define GLOBES_BASE "data/dune/2103.04797v2/dune_globes"
#define DUNE_FD_MAX_ROWS 4096
#define FD_MAX_BINS 512

#define AVOGADRO 6.02214076e23
#define M2_TO_CM2 1.0e4
#define XSEC_SCALE_CM2 1.0e-38
#define FD_BASELINE_KM 1284.9
#define POT_PER_YEAR 11.0e20
#define NUTIME_YEARS 6.5
#define NUBARTIME_YEARS 6.5
#define TARGET_MASS_KT 40.0
#define TRUE_ENERGY_MAX_GEV 20.0
#define GLOBES_EVENT_NORM 1.017718
#define FD_MATTER_DENSITY_G_CM3 2.848
#define FD_MATTER_ELECTRON_FRACTION 0.5
#define FD_MATTER_NEUTRON_FRACTION 0.5
#define FD_MATTER_INCLUDE_NC_STERILE 1
#define FD_MATTER_A_CC_COEFF 1.52e-4

typedef struct {
    char panel[32];
    char component[32];
    double energy_GeV;
    double globes_events;
} Fig4Row;

typedef struct {
    const char *flux_mode;
    int initial;
    int final;
    int interaction_cc;
    const char *smear;
    const char *eff;
    int anti;
} FdChannel;

typedef struct {
    double matrix[FD_MAX_BINS][FD_MAX_BINS];
    int n_rows;
    int n_cols;
} SmearingMatrix;

typedef struct {
    double values[FD_MAX_BINS];
    int n;
} Vector;

static const FdChannel CHANNEL_FHC_APP_SIGNAL[] = {
    {"FHC", 1, 0, 1, "app_nue_sig", "post_app_FHC_nue_sig", 0},
    {"FHC", 1, 0, 1, "app_nuebar_sig", "post_app_FHC_nuebar_sig", 1},
};
static const FdChannel CHANNEL_FHC_APP_BEAM[] = {
    {"FHC", 0, 0, 1, "app_nue_bkg", "post_app_FHC_nue_bkg", 0},
    {"FHC", 0, 0, 1, "app_nuebar_bkg", "post_app_FHC_nuebar_bkg", 1},
};
static const FdChannel CHANNEL_FHC_APP_NUMU[] = {
    {"FHC", 1, 1, 1, "app_numu_bkg", "post_app_FHC_numu_bkg", 0},
    {"FHC", 1, 1, 1, "app_numubar_bkg", "post_app_FHC_numubar_bkg", 1},
};
static const FdChannel CHANNEL_FHC_APP_NC[] = {
    {"FHC", 1, 1, 0, "app_NC_bkg", "post_app_FHC_NC_bkg", 0},
    {"FHC", 1, 1, 0, "app_NC_bkg", "post_app_FHC_aNC_bkg", 1},
};

static const FdChannel CHANNEL_RHC_APP_SIGNAL[] = {
    {"RHC", 1, 0, 1, "app_nue_sig", "post_app_RHC_nue_sig", 0},
    {"RHC", 1, 0, 1, "app_nuebar_sig", "post_app_RHC_nuebar_sig", 1},
};
static const FdChannel CHANNEL_RHC_APP_BEAM[] = {
    {"RHC", 0, 0, 1, "app_nue_bkg", "post_app_RHC_nue_bkg", 0},
    {"RHC", 0, 0, 1, "app_nuebar_bkg", "post_app_RHC_nuebar_bkg", 1},
};
static const FdChannel CHANNEL_RHC_APP_NUMU[] = {
    {"RHC", 1, 1, 1, "app_numu_bkg", "post_app_RHC_numu_bkg", 0},
    {"RHC", 1, 1, 1, "app_numubar_bkg", "post_app_RHC_numubar_bkg", 1},
};
static const FdChannel CHANNEL_RHC_APP_NC[] = {
    {"RHC", 1, 1, 0, "app_NC_bkg", "post_app_RHC_NC_bkg", 0},
    {"RHC", 1, 1, 0, "app_aNC_bkg", "post_app_RHC_aNC_bkg", 1},
};

static const FdChannel CHANNEL_FHC_DIS_SIGNAL[] = {
    {"FHC", 1, 1, 1, "dis_numu_sig", "post_dis_FHC_numu_sig", 0},
};
static const FdChannel CHANNEL_FHC_DIS_WRONG[] = {
    {"FHC", 1, 1, 1, "dis_numubar_sig", "post_dis_FHC_numubar_sig", 1},
};
static const FdChannel CHANNEL_FHC_DIS_TAU[] = {
    {"FHC", 1, 2, 1, "dis_nutau_bkg", "post_dis_FHC_nutau_bkg", 0},
    {"FHC", 1, 2, 1, "dis_nutaubar_bkg", "post_dis_FHC_nutaubar_bkg", 1},
};
static const FdChannel CHANNEL_FHC_DIS_NC[] = {
    {"FHC", 1, 1, 0, "dis_NC_bkg", "post_dis_FHC_NC_bkg", 0},
    {"FHC", 1, 1, 0, "dis_aNC_bkg", "post_dis_FHC_NC_bkg", 1},
};

static const FdChannel CHANNEL_RHC_DIS_SIGNAL[] = {
    {"RHC", 1, 1, 1, "dis_numubar_sig", "post_dis_RHC_numubar_sig", 1},
};
static const FdChannel CHANNEL_RHC_DIS_WRONG[] = {
    {"RHC", 1, 1, 1, "dis_numu_sig", "post_dis_RHC_numu_sig", 0},
};
static const FdChannel CHANNEL_RHC_DIS_TAU[] = {
    {"RHC", 1, 2, 1, "dis_nutau_bkg", "post_dis_RHC_nutau_bkg", 0},
    {"RHC", 1, 2, 1, "dis_nutaubar_bkg", "post_dis_RHC_nutaubar_bkg", 1},
};
static const FdChannel CHANNEL_RHC_DIS_NC[] = {
    {"RHC", 1, 1, 0, "dis_NC_bkg", "post_dis_RHC_NC_bkg", 0},
    {"RHC", 1, 1, 0, "dis_aNC_bkg", "post_dis_RHC_NC_bkg", 1},
};

static int load_fig4_rows(Fig4Row *rows, int max_rows, int *out_count) {
    FILE *in = fopen(FIG4_INPUT_CSV, "r");
    if (!in) {
        return 1;
    }

    char line[512];
    int count = 0;
    if (!fgets(line, sizeof(line), in)) {
        fclose(in);
        return 1;
    }

    while (fgets(line, sizeof(line), in)) {
        if (count >= max_rows) {
            fclose(in);
            return 1;
        }
        Fig4Row row;
        memset(&row, 0, sizeof(row));
        if (sscanf(line, "%31[^,],%31[^,],%lf,%lf",
                   row.panel,
                   row.component,
                   &row.energy_GeV,
                   &row.globes_events) == 4) {
            rows[count++] = row;
        }
    }

    fclose(in);
    *out_count = count;
    return count > 0 ? 0 : 1;
}

static int parse_numbers_until_brace(FILE *in, double *values, int max_values) {
    int n = 0;
    char token[128];
    int token_len = 0;
    int c;
    while ((c = fgetc(in)) != EOF) {
        if (c == '}') {
            if (token_len > 0 && n < max_values) {
                token[token_len] = '\0';
                values[n++] = strtod(token, NULL);
            }
            return n;
        }
        if (isdigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
            if (token_len < (int)sizeof(token) - 1) {
                token[token_len++] = (char)c;
            }
        } else if (token_len > 0) {
            if (n < max_values) {
                token[token_len] = '\0';
                values[n++] = strtod(token, NULL);
            }
            token_len = 0;
        }
    }
    return -1;
}

static int read_glb_vector(const char *path, const char *name, Vector *out) {
    FILE *in = fopen(path, "r");
    if (!in) return 1;

    const size_t name_len = strlen(name);
    char window[128] = {0};
    size_t window_len = 0;
    int c;
    int found = 0;
    while ((c = fgetc(in)) != EOF) {
        if (window_len < sizeof(window) - 1) {
            window[window_len++] = (char)c;
            window[window_len] = '\0';
        } else {
            memmove(window, window + 1, sizeof(window) - 2);
            window[sizeof(window) - 2] = (char)c;
            window[sizeof(window) - 1] = '\0';
        }
        if (window_len >= name_len && strstr(window, name)) {
            found = 1;
            break;
        }
    }
    if (!found) {
        fclose(in);
        return 1;
    }
    while ((c = fgetc(in)) != EOF && c != '{') {
    }
    if (c != '{') {
        fclose(in);
        return 1;
    }
    double values[FD_MAX_BINS];
    int n = parse_numbers_until_brace(in, values, FD_MAX_BINS);
    fclose(in);
    if (n <= 0) return 1;
    out->n = n;
    for (int i = 0; i < n; ++i) out->values[i] = values[i];
    return 0;
}

static int read_first_brace_vector(const char *path, Vector *out) {
    FILE *in = fopen(path, "r");
    if (!in) return 1;
    int c;
    while ((c = fgetc(in)) != EOF && c != '{') {
    }
    if (c != '{') {
        fclose(in);
        return 1;
    }
    double values[FD_MAX_BINS];
    int n = parse_numbers_until_brace(in, values, FD_MAX_BINS);
    fclose(in);
    if (n <= 0) return 1;
    out->n = n;
    for (int i = 0; i < n; ++i) out->values[i] = values[i];
    return 0;
}

static int read_smearing(const char *path, SmearingMatrix *out) {
    FILE *in = fopen(path, "r");
    if (!in) return 1;
    memset(out, 0, sizeof(*out));

    int c;
    while ((c = fgetc(in)) != EOF) {
        if (c != '{') continue;
        double values[FD_MAX_BINS + 2];
        int n = parse_numbers_until_brace(in, values, FD_MAX_BINS + 2);
        if (n < 3) continue;
        if (out->n_rows >= FD_MAX_BINS) {
            fclose(in);
            return 1;
        }
        int lo = (int)lround(values[0]);
        int hi = (int)lround(values[1]);
        if (lo < 0) lo = 0;
        if (hi >= FD_MAX_BINS) hi = FD_MAX_BINS - 1;
        const int row = out->n_rows++;
        for (int col = lo; col <= hi && 2 + col - lo < n; ++col) {
            out->matrix[row][col] = values[2 + col - lo];
            if (col + 1 > out->n_cols) out->n_cols = col + 1;
        }
    }

    fclose(in);
    return out->n_rows > 0 ? 0 : 1;
}

static void path_join(char *out, size_t out_size, const char *a, const char *b, const char *c) {
    snprintf(out, out_size, "%s/%s/%s", a, b, c);
}

static DuneFluxFlavor flavor_for(int flavor, int anti) {
    if (flavor == 0) return anti ? DUNE_FLUX_NUEBAR : DUNE_FLUX_NUE;
    if (flavor == 1) return anti ? DUNE_FLUX_NUMUBAR : DUNE_FLUX_NUMU;
    return anti ? DUNE_FLUX_NUTAUBAR : DUNE_FLUX_NUTAU;
}

static double clamp_probability(double p) {
    if (!isfinite(p)) return 0.0;
    if (p < 0.0) return 0.0;
    if (p > 1.0) return 1.0;
    return p;
}

static double probability_iss_vacuum(
    const DuneTheoryPoint *point,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    int anti) {
    if (!point || energy_GeV <= 0.0 || baseline_km < 0.0 ||
        alpha < 0 || beta < 0 || alpha >= point->n_light || beta >= point->n_light) {
        return alpha == beta ? 1.0 : 0.0;
    }

    double p = (alpha == beta) ? 1.0 : 0.0;
    for (int i = 0; i < point->n_light; ++i) {
        const double mi2 = point->light_masses_eV[i] * point->light_masses_eV[i];
        for (int j = i + 1; j < point->n_light; ++j) {
            const double mj2 = point->light_masses_eV[j] * point->light_masses_eV[j];
            const double phase = 1.267 * (mj2 - mi2) * baseline_km / energy_GeV;
            const double complex a =
                point->mixing[alpha][i] * conj(point->mixing[beta][i]) *
                conj(point->mixing[alpha][j]) * point->mixing[beta][j];
            p -= 4.0 * creal(a) * sin(phase) * sin(phase);
            p += (anti ? -2.0 : 2.0) * cimag(a) * sin(2.0 * phase);
        }
    }
    return clamp_probability(p);
}

static double probability_iss_matter(
    const DuneTheoryPoint *point,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    int anti) {
    if (!point || energy_GeV <= 0.0 || baseline_km < 0.0 ||
        alpha < 0 || beta < 0 || alpha >= point->n_light || beta >= point->n_light) {
        return alpha == beta ? 1.0 : 0.0;
    }

    const int n = point->n_light;
    double mass_sq[n];
    double complex u[n][n];
    for (int i = 0; i < n; ++i) {
        mass_sq[i] = point->light_masses_eV[i] * point->light_masses_eV[i];
        for (int j = 0; j < n; ++j) {
            u[i][j] = point->mixing[i][j];
        }
    }

    const double p = probability_diagonalization_constant_density_n(
        n,
        alpha,
        beta,
        energy_GeV,
        baseline_km,
        mass_sq,
        u,
        anti,
        FD_MATTER_DENSITY_G_CM3,
        FD_MATTER_ELECTRON_FRACTION,
        FD_MATTER_NEUTRON_FRACTION,
        FD_MATTER_INCLUDE_NC_STERILE,
        FD_MATTER_A_CC_COEFF);
    return clamp_probability(p);
}

static double active_probability_iss(
    const DuneTheoryPoint *point,
    int alpha,
    double energy_GeV,
    double baseline_km,
    int anti,
    int use_matter) {
    double p = 0.0;
    for (int beta = 0; beta < 3 && beta < point->n_light; ++beta) {
        p += use_matter
                 ? probability_iss_matter(point, alpha, beta, energy_GeV, baseline_km, anti)
                 : probability_iss_vacuum(point, alpha, beta, energy_GeV, baseline_km, anti);
    }
    return clamp_probability(p);
}

static double channel_probability_at_baseline(
    const FdChannel *channel,
    const DuneTheoryPoint *point,
    double energy_GeV,
    double baseline_km,
    int use_matter) {
    return channel->interaction_cc
               ? (use_matter
                      ? probability_iss_matter(point, channel->initial, channel->final, energy_GeV, baseline_km, channel->anti)
                      : probability_iss_vacuum(point, channel->initial, channel->final, energy_GeV, baseline_km, channel->anti))
               : active_probability_iss(point, channel->initial, energy_GeV, baseline_km, channel->anti, use_matter);
}

static double channel_probability_with_source_profile(
    const FdChannel *channel,
    const DuneTheoryPoint *point,
    DuneFluxFlavor flux_flavor,
    double energy_GeV,
    int use_matter,
    const DuneDk2nuFluxZTable *source_profile) {
    if (!source_profile || !source_profile->rows) {
        return channel_probability_at_baseline(channel, point, energy_GeV, FD_BASELINE_KM, use_matter);
    }

    double weighted = 0.0;
    double denom = 0.0;
    for (int i = 0; i < source_profile->n_rows; ++i) {
        const DuneDk2nuFluxZRow *row = &source_profile->rows[i];
        if (row->flavor != flux_flavor ||
            energy_GeV < row->e_low_GeV ||
            energy_GeV >= row->e_high_GeV) {
            continue;
        }
        const double z_m = 0.5 * (row->z_low_m + row->z_high_m);
        double baseline_km = FD_BASELINE_KM - z_m * 1.0e-3;
        if (baseline_km < 0.0) baseline_km = 0.0;
        weighted += row->weight *
                    channel_probability_at_baseline(channel, point, energy_GeV, baseline_km, use_matter);
        denom += row->weight;
    }

    if (denom <= 0.0) {
        return channel_probability_at_baseline(channel, point, energy_GeV, FD_BASELINE_KM, use_matter);
    }
    return clamp_probability(weighted / denom);
}

static void build_active_3nu_benchmark_point(const DuneTheoryPoint *source, DuneTheoryPoint *benchmark) {
    if (!source || !benchmark) {
        return;
    }

    *benchmark = *source;
    strncpy(benchmark->model, "iss23_active3nu", sizeof(benchmark->model) - 1);
    benchmark->model[sizeof(benchmark->model) - 1] = '\0';
    benchmark->n_light = 3;
    benchmark->n_active = 3;
    benchmark->dm41_eV2 = 0.0;

    for (int i = 3; i < 8; ++i) {
        benchmark->light_masses_eV[i] = 0.0;
    }
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            benchmark->mixing[r][c] = (r < 3 && c < 3) ? source->mixing[r][c] : 0.0 + 0.0 * I;
        }
    }
}

static double eval_flux_value(const DuneFluxTable *table, DuneFluxFlavor flavor, double energy_GeV) {
    double value = 0.0;
    if (dune_flux_table_eval(table, flavor, energy_GeV, &value) != DUNE_STATUS_OK) return 0.0;
    return value;
}

static double eval_xsec_value(const DuneXsecTable *table, DuneFluxFlavor flavor, double energy_GeV) {
    double sigma_over_e = 0.0;
    if (dune_xsec_table_eval(table, flavor, energy_GeV, &sigma_over_e) != DUNE_STATUS_OK) return 0.0;
    return sigma_over_e * energy_GeV * XSEC_SCALE_CM2;
}

static double compute_channel_reco(
    const FdChannel *channel,
    const DuneTheoryPoint *point,
    const DuneFluxTable *fhc_flux,
    const DuneFluxTable *rhc_flux,
    const DuneXsecTable *cc_xsec,
    const DuneXsecTable *nc_xsec,
    const double *sampling_centers,
    const double *sampling_widths,
    int n_sampling,
    const double *rec_edges,
    int n_rec_edges,
    double *out_025,
    int *out_n_025,
    int use_matter,
    const DuneDk2nuFluxZTable *source_profile) {
    char path[512];
    char filename[128];
    snprintf(filename, sizeof(filename), "%s.txt", channel->smear);
    path_join(path, sizeof(path), GLOBES_BASE, "smr", filename);
    SmearingMatrix *smear = (SmearingMatrix *)calloc(1, sizeof(*smear));
    if (!smear) return 1;
    if (read_smearing(path, smear) != 0) {
        free(smear);
        return 1;
    }

    snprintf(filename, sizeof(filename), "%s.txt", channel->eff);
    path_join(path, sizeof(path), GLOBES_BASE, "eff", filename);
    Vector eff;
    if (read_first_brace_vector(path, &eff) != 0) {
        free(smear);
        return 1;
    }

    const DuneFluxTable *flux_table = strcmp(channel->flux_mode, "RHC") == 0 ? rhc_flux : fhc_flux;
    const DuneXsecTable *xsec_table = channel->interaction_cc ? cc_xsec : nc_xsec;
    const DuneFluxFlavor flux_flavor = flavor_for(channel->initial, channel->anti);
    const DuneFluxFlavor xsec_flavor = channel->interaction_cc
                                           ? flavor_for(channel->final, channel->anti)
                                           : flux_flavor;
    const double pot = POT_PER_YEAR * (strcmp(channel->flux_mode, "RHC") == 0 ? NUBARTIME_YEARS : NUTIME_YEARS);
    const double target_nucleons = TARGET_MASS_KT * 1.0e9 * AVOGADRO;

    double true_counts[FD_MAX_BINS] = {0.0};
    for (int i = 0; i < n_sampling && i < FD_MAX_BINS; ++i) {
        const double energy = sampling_centers[i];
        if (energy > TRUE_ENERGY_MAX_GEV) {
            true_counts[i] = 0.0;
            continue;
        }
        const double flux = eval_flux_value(flux_table, flux_flavor, energy) / M2_TO_CM2;
        const double xsec = eval_xsec_value(xsec_table, xsec_flavor, energy);
        const double prob = channel_probability_with_source_profile(
            channel,
            point,
            flux_flavor,
            energy,
            use_matter,
            source_profile);
        true_counts[i] = GLOBES_EVENT_NORM * flux * xsec * prob * sampling_widths[i] * pot * target_nucleons;
    }

    double reco[FD_MAX_BINS] = {0.0};
    const int n_reco = smear->n_rows < FD_MAX_BINS ? smear->n_rows : FD_MAX_BINS;
    for (int r = 0; r < n_reco; ++r) {
        double sum = 0.0;
        for (int t = 0; t < smear->n_cols && t < n_sampling; ++t) {
            sum += smear->matrix[r][t] * true_counts[t];
        }
        reco[r] = sum * (r < eff.n ? eff.values[r] : 0.0);
    }

    int n = 0;
    for (int r = 0; r + 1 < n_reco && r + 2 < n_rec_edges && rec_edges[r] < 8.0; r += 2) {
        out_025[n++] = reco[r] + reco[r + 1];
    }
    *out_n_025 = n;
    free(smear);
    return 0;
}

static int channel_group_for(const char *panel, const char *component, const FdChannel **channels, int *n_channels) {
    *channels = NULL;
    *n_channels = 0;
#define MATCH(P, C, ARR) do { if (strcmp(panel, P) == 0 && strcmp(component, C) == 0) { *channels = ARR; *n_channels = (int)(sizeof(ARR) / sizeof((ARR)[0])); return 0; } } while (0)
    MATCH("FHC_app", "signal", CHANNEL_FHC_APP_SIGNAL);
    MATCH("FHC_app", "beam", CHANNEL_FHC_APP_BEAM);
    MATCH("FHC_app", "numu", CHANNEL_FHC_APP_NUMU);
    MATCH("FHC_app", "nc", CHANNEL_FHC_APP_NC);
    MATCH("RHC_app", "signal", CHANNEL_RHC_APP_SIGNAL);
    MATCH("RHC_app", "beam", CHANNEL_RHC_APP_BEAM);
    MATCH("RHC_app", "numu", CHANNEL_RHC_APP_NUMU);
    MATCH("RHC_app", "nc", CHANNEL_RHC_APP_NC);
    MATCH("FHC_dis", "signal", CHANNEL_FHC_DIS_SIGNAL);
    MATCH("FHC_dis", "wrong_mu", CHANNEL_FHC_DIS_WRONG);
    MATCH("FHC_dis", "tau", CHANNEL_FHC_DIS_TAU);
    MATCH("FHC_dis", "nc", CHANNEL_FHC_DIS_NC);
    MATCH("RHC_dis", "signal", CHANNEL_RHC_DIS_SIGNAL);
    MATCH("RHC_dis", "wrong_mu", CHANNEL_RHC_DIS_WRONG);
    MATCH("RHC_dis", "tau", CHANNEL_RHC_DIS_TAU);
    MATCH("RHC_dis", "nc", CHANNEL_RHC_DIS_NC);
#undef MATCH
    return 1;
}

static int build_centers(const Vector *widths, double *centers, double *edges, int max_bins) {
    if (!widths || widths->n <= 0 || widths->n + 1 > max_bins) return 0;
    edges[0] = 0.0;
    for (int i = 0; i < widths->n; ++i) {
        edges[i + 1] = edges[i] + widths->values[i];
        centers[i] = 0.5 * (edges[i] + edges[i + 1]);
    }
    return widths->n;
}

static void matter_output_path(const char *vacuum_path, char *out, size_t out_size) {
    const char *dot = strrchr(vacuum_path, '.');
    if (dot && strcmp(dot, ".csv") == 0) {
        const size_t prefix_len = (size_t)(dot - vacuum_path);
        if (prefix_len + strlen("_matter.csv") + 1 < out_size) {
            memcpy(out, vacuum_path, prefix_len);
            out[prefix_len] = '\0';
            strncat(out, "_matter.csv", out_size - strlen(out) - 1);
            return;
        }
    }
    snprintf(out, out_size, "%s_matter.csv", vacuum_path);
}

static int write_fd_fig4_output(
    const char *output_csv,
    const Fig4Row *rows,
    int n_rows,
    const DuneTheoryPoint *point,
    const DuneFluxTable *fhc_flux,
    const DuneFluxTable *rhc_flux,
    const DuneXsecTable *cc_xsec,
    const DuneXsecTable *nc_xsec,
    const double *sampling_centers,
    const double *sampling_widths,
    int n_sampling,
    const double *rec_edges,
    int n_rec_edges,
    int use_matter,
    const DuneDk2nuFluxZTable *fhc_source_profile,
    const DuneDk2nuFluxZTable *rhc_source_profile,
    double *out_max_abs_rel) {

    FILE *out = fopen(output_csv, "w");
    if (!out) {
        fprintf(stderr, "DUNE FD Fig.4: impossible d'ouvrir %s\n", output_csv);
        return 1;
    }

    DuneTheoryPoint benchmark3nu;
    build_active_3nu_benchmark_point(point, &benchmark3nu);

    fprintf(out, "point_id,panel,component,Erec_GeV,globes_events,benchmark3nu_events,iss23_events,rel_diff,rel_diff_vs_globes\n");
    double max_abs_rel = 0.0;
    typedef struct {
        char panel[32];
        char component[32];
        double benchmark_values[FD_MAX_BINS];
        double iss_values[FD_MAX_BINS];
        int n_benchmark;
        int n_iss;
    } ComponentCache;
    ComponentCache cache[64];
    int n_cache = 0;
    for (int i = 0; i < n_rows; ++i) {
        ComponentCache *item = NULL;
        for (int ic = 0; ic < n_cache; ++ic) {
            if (strcmp(cache[ic].panel, rows[i].panel) == 0 &&
                strcmp(cache[ic].component, rows[i].component) == 0) {
                item = &cache[ic];
                break;
            }
        }
        if (!item && n_cache < (int)(sizeof(cache) / sizeof(cache[0]))) {
            item = &cache[n_cache++];
            memset(item, 0, sizeof(*item));
            strncpy(item->panel, rows[i].panel, sizeof(item->panel) - 1);
            strncpy(item->component, rows[i].component, sizeof(item->component) - 1);

            const FdChannel *channels = NULL;
            int n_channels = 0;
            if (channel_group_for(rows[i].panel, rows[i].component, &channels, &n_channels) == 0) {
                for (int c = 0; c < n_channels; ++c) {
                    const DuneDk2nuFluxZTable *source_profile =
                        strcmp(channels[c].flux_mode, "RHC") == 0 ? rhc_source_profile : fhc_source_profile;
                    double tmp_benchmark[FD_MAX_BINS] = {0.0};
                    double tmp[FD_MAX_BINS] = {0.0};
                    int n_tmp_benchmark = 0;
                    int n_tmp = 0;
                    if (compute_channel_reco(&channels[c], &benchmark3nu, fhc_flux, rhc_flux, cc_xsec, nc_xsec,
                                             sampling_centers, sampling_widths, n_sampling,
                                             rec_edges, n_rec_edges, tmp_benchmark, &n_tmp_benchmark,
                                             use_matter, source_profile) == 0) {
                        if (n_tmp_benchmark > item->n_benchmark) item->n_benchmark = n_tmp_benchmark;
                        for (int k = 0; k < n_tmp_benchmark; ++k) item->benchmark_values[k] += tmp_benchmark[k];
                    }
                    if (compute_channel_reco(&channels[c], point, fhc_flux, rhc_flux, cc_xsec, nc_xsec,
                                             sampling_centers, sampling_widths, n_sampling,
                                             rec_edges, n_rec_edges, tmp, &n_tmp,
                                             use_matter, source_profile) == 0) {
                        if (n_tmp > item->n_iss) item->n_iss = n_tmp;
                        for (int k = 0; k < n_tmp; ++k) item->iss_values[k] += tmp[k];
                    }
                }
            }
        }
        int bin_index = -1;
        const int n_benchmark = item ? item->n_benchmark : 0;
        const int n_iss = item ? item->n_iss : 0;
        const int n_bins = n_benchmark > n_iss ? n_benchmark : n_iss;
        for (int k = 0; k < n_bins; ++k) {
            const double e = 0.125 + 0.25 * (double)k;
            if (fabs(e - rows[i].energy_GeV) < 1e-9) {
                bin_index = k;
                break;
            }
        }
        const double globes = rows[i].globes_events;
        const double benchmark = (item && bin_index >= 0) ? item->benchmark_values[bin_index] : globes;
        const double iss = (item && bin_index >= 0) ? item->iss_values[bin_index] : globes;
        const double rel = benchmark > 0.0
                               ? (iss - benchmark) / benchmark
                               : 0.0;
        const double rel_vs_globes = globes > 0.0
                                         ? (iss - globes) / globes
                                         : 0.0;
        if (fabs(rel) > max_abs_rel) max_abs_rel = fabs(rel);
        fprintf(out, "%d,%s,%s,%.10g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                point->point_id,
                rows[i].panel,
                rows[i].component,
                rows[i].energy_GeV,
                globes,
                benchmark,
                iss,
                rel,
                rel_vs_globes);
    }

    fclose(out);
    if (out_max_abs_rel) {
        *out_max_abs_rel = max_abs_rel;
    }
    return 0;
}

int run_dune_fd_fig4_validation(const SimulationConfig *cfg) {
    if (!cfg || cfg->dune_spectrum_pred_csv[0] == '\0') {
        fprintf(stderr, "DUNE FD Fig.4 validation: chemin output.spectrum_pred_csv manquant\n");
        return 1;
    }

    DuneRunContext ctx;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE FD Fig.4 init failed: %s\n", dune_status_message(status));
        return 1;
    }
    status = dune_theory_point_load(&ctx);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE FD Fig.4 point load failed: %s\n", dune_status_message(status));
        return 1;
    }

    static Fig4Row rows[DUNE_FD_MAX_ROWS];
    int n_rows = 0;
    if (load_fig4_rows(rows, DUNE_FD_MAX_ROWS, &n_rows) != 0) {
        fprintf(stderr, "DUNE FD Fig.4: impossible de lire %s. Lance d'abord le script GLoBES Fig.4.\n", FIG4_INPUT_CSV);
        return 1;
    }

    char path[512];
    static DuneFluxTable fhc_flux;
    static DuneFluxTable rhc_flux;
    static DuneXsecTable cc_xsec;
    static DuneXsecTable nc_xsec;
    path_join(path, sizeof(path), GLOBES_BASE, "flux", "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_neutrino_LBNEFD_globes_flux.txt");
    if (dune_flux_table_load_globes(path, &fhc_flux) != DUNE_STATUS_OK) return 1;
    path_join(path, sizeof(path), GLOBES_BASE, "flux", "histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_antineutrino_LBNEFD_globes_flux.txt");
    if (dune_flux_table_load_globes(path, &rhc_flux) != DUNE_STATUS_OK) return 1;
    path_join(path, sizeof(path), GLOBES_BASE, "xsec", "xsec_cc.dat");
    if (dune_xsec_table_load_globes(path, &cc_xsec) != DUNE_STATUS_OK) return 1;
    path_join(path, sizeof(path), GLOBES_BASE, "xsec", "xsec_nc.dat");
    if (dune_xsec_table_load_globes(path, &nc_xsec) != DUNE_STATUS_OK) return 1;

    path_join(path, sizeof(path), GLOBES_BASE, "", "DUNE_GLoBES.glb");
    Vector binsize;
    Vector sampling;
    if (read_glb_vector(path, "$binsize", &binsize) != 0 ||
        read_glb_vector(path, "$sampling_stepsize", &sampling) != 0) {
        fprintf(stderr, "DUNE FD Fig.4: impossible de lire binsize/sampling dans %s\n", path);
        return 1;
    }

    double rec_centers[FD_MAX_BINS], rec_edges[FD_MAX_BINS];
    double sampling_centers[FD_MAX_BINS], sampling_edges[FD_MAX_BINS];
    (void)rec_centers;
    const int n_rec = build_centers(&binsize, rec_centers, rec_edges, FD_MAX_BINS);
    const int n_sampling = build_centers(&sampling, sampling_centers, sampling_edges, FD_MAX_BINS);
    if (n_rec <= 0 || n_sampling <= 0) return 1;

    static DuneDk2nuFluxZTable fhc_source_profile;
    static DuneDk2nuFluxZTable rhc_source_profile;
    DuneDk2nuFluxZTable *fhc_source_profile_ptr = NULL;
    DuneDk2nuFluxZTable *rhc_source_profile_ptr = NULL;
    if (cfg->dune_dk2nu_flux_z_fhc_file[0] && cfg->dune_dk2nu_flux_z_rhc_file[0]) {
        const DuneStatus fhc_profile_status =
            dune_dk2nu_flux_z_load_csv(cfg->dune_dk2nu_flux_z_fhc_file, &fhc_source_profile);
        const DuneStatus rhc_profile_status =
            dune_dk2nu_flux_z_load_csv(cfg->dune_dk2nu_flux_z_rhc_file, &rhc_source_profile);
        if (fhc_profile_status == DUNE_STATUS_OK && rhc_profile_status == DUNE_STATUS_OK) {
            fhc_source_profile_ptr = &fhc_source_profile;
            rhc_source_profile_ptr = &rhc_source_profile;
            fprintf(stderr,
                    "DUNE FD Fig.4: profils de source dk2nu actifs "
                    "(FHC rows=%d z=[%.4g, %.4g] m, RHC rows=%d z=[%.4g, %.4g] m)\n",
                    fhc_source_profile.n_rows,
                    fhc_source_profile.z_min_m,
                    fhc_source_profile.z_max_m,
                    rhc_source_profile.n_rows,
                    rhc_source_profile.z_min_m,
                    rhc_source_profile.z_max_m);
        } else {
            fprintf(stderr,
                    "DUNE FD Fig.4: profils de source dk2nu indisponibles, fallback L fixe "
                    "(FHC=%s, RHC=%s)\n",
                    cfg->dune_dk2nu_flux_z_fhc_file,
                    cfg->dune_dk2nu_flux_z_rhc_file);
            dune_dk2nu_flux_z_free(&fhc_source_profile);
            dune_dk2nu_flux_z_free(&rhc_source_profile);
        }
    }

    double max_abs_rel = 0.0;
    if (write_fd_fig4_output(
            cfg->dune_spectrum_pred_csv,
            rows,
            n_rows,
            &ctx.theory,
            &fhc_flux,
            &rhc_flux,
            &cc_xsec,
            &nc_xsec,
            sampling_centers,
            sampling.values,
            n_sampling,
            rec_edges,
            n_rec + 1,
            0,
            fhc_source_profile_ptr,
            rhc_source_profile_ptr,
            &max_abs_rel) != 0) {
        return 1;
    }

    char matter_csv[512];
    double matter_max_abs_rel = 0.0;
    matter_output_path(cfg->dune_spectrum_pred_csv, matter_csv, sizeof(matter_csv));
    if (write_fd_fig4_output(
            matter_csv,
            rows,
            n_rows,
            &ctx.theory,
            &fhc_flux,
            &rhc_flux,
            &cc_xsec,
            &nc_xsec,
            sampling_centers,
            sampling.values,
            n_sampling,
            rec_edges,
            n_rec + 1,
            1,
            fhc_source_profile_ptr,
            rhc_source_profile_ptr,
            &matter_max_abs_rel) != 0) {
        return 1;
    }

    fprintf(stderr,
            "DUNE FD Fig.4 validation: ISS spectra recomputed from FD unoscillated fluxes "
            "(point=%d, rows=%d, max_abs_rel=%.6g, L=%.6g km, csv=%s)\n",
            ctx.theory.point_id,
            n_rows,
            max_abs_rel,
            FD_BASELINE_KM,
            cfg->dune_spectrum_pred_csv);
    fprintf(stderr,
            "DUNE FD Fig.4 validation: ISS matter spectra recomputed with rho=%.6g g/cm3 "
            "(point=%d, rows=%d, max_abs_rel=%.6g, L=%.6g km, csv=%s)\n",
            FD_MATTER_DENSITY_G_CM3,
            ctx.theory.point_id,
            n_rows,
            matter_max_abs_rel,
            FD_BASELINE_KM,
            matter_csv);
    dune_dk2nu_flux_z_free(&fhc_source_profile);
    dune_dk2nu_flux_z_free(&rhc_source_profile);
    return 0;
}
