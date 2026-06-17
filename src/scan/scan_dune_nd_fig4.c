#include "scan.h"

#include "beam/dk2nu_flux_z.h"
#include "beam/flux_table.h"
#include "dune_nd/dune_nd.h"
#include "interactions/xsec_table.h"
#include "oscillation_dune/exact_light_engine.h"
#include "theory/theory_point_loader.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define ND_FIG4_MAX_ROWS 1024
#define ND_AVOGADRO 6.02214076e23
#define ND_M2_TO_CM2 1.0e4
#define ND_XSEC_SCALE_CM2 1.0e-38
#define ND_POT_PER_YEAR 11.0e20
#define ND_EXPOSURE_YEARS 5.0
#define ND_TARGET_MASS_KT 0.067

typedef struct {
    char panel[32];
    char component[32];
    double energy_GeV;
    double globes_events;
    double iss23_events;
} NdFig4Row;

typedef enum {
    ND_SOURCE_POINT = 0,
    ND_SOURCE_UNIFORM,
    ND_SOURCE_DK2NU
} NdSourceModel;

typedef struct {
    NdSourceModel model;
    double detector_baseline_km;
    double source_z_start_m;
    double decay_pipe_length_m;
    int source_z_bins;
    const DuneDk2nuFluxZTable *dk2nu;
} NdSourceAverage;

static NdSourceModel parse_source_model(const char *value) {
    if (value && strcmp(value, "point") == 0) {
        return ND_SOURCE_POINT;
    }
    if (value && strcmp(value, "dk2nu") == 0) {
        return ND_SOURCE_DK2NU;
    }
    return ND_SOURCE_UNIFORM;
}

static const char *source_model_name(NdSourceModel model) {
    switch (model) {
        case ND_SOURCE_POINT: return "point";
        case ND_SOURCE_DK2NU: return "dk2nu";
        case ND_SOURCE_UNIFORM:
        default: return "uniform";
    }
}

static const char *flux_flavor_name(DuneFluxFlavor flavor) {
    switch (flavor) {
        case DUNE_FLUX_NUE: return "nue";
        case DUNE_FLUX_NUMU: return "numu";
        case DUNE_FLUX_NUEBAR: return "nuebar";
        case DUNE_FLUX_NUMUBAR: return "numubar";
        default: return "unknown";
    }
}

static int probability_csv_path(const char *spectrum_path, char *out_path, size_t out_size) {
    if (!spectrum_path || !out_path || out_size == 0) {
        return 1;
    }
    const char *dot = strrchr(spectrum_path, '.');
    if (dot && strcmp(dot, ".csv") == 0) {
        const size_t prefix_len = (size_t)(dot - spectrum_path);
        const char *suffix = "_raw_probabilities.csv";
        if (prefix_len + strlen(suffix) + 1 > out_size) {
            return 1;
        }
        memcpy(out_path, spectrum_path, prefix_len);
        out_path[prefix_len] = '\0';
        strcat(out_path, suffix);
        return 0;
    }
    const char *suffix = "_raw_probabilities.csv";
    if (strlen(spectrum_path) + strlen(suffix) + 1 > out_size) {
        return 1;
    }
    strcpy(out_path, spectrum_path);
    strcat(out_path, suffix);
    return 0;
}

static double clamp_probability(double p) {
    if (!isfinite(p)) return 0.0;
    if (p < 0.0) return 0.0;
    if (p > 1.0) return 1.0;
    return p;
}

static double probability_no_sterile(int alpha, int beta) {
    return alpha == beta ? 1.0 : 0.0;
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

static double probability_iss23(
    const DuneTheoryPoint *point,
    DuneFluxFlavor initial,
    int alpha,
    int beta,
    double energy_GeV,
    const NdSourceAverage *source) {
    if (!source || source->model == ND_SOURCE_POINT || source->source_z_bins <= 1 || source->decay_pipe_length_m <= 0.0) {
        double p = 0.0;
        if (dune_exact_light_probability(point, alpha, beta, energy_GeV, source ? source->detector_baseline_km : 0.0, &p) != DUNE_STATUS_OK) {
            return probability_no_sterile(alpha, beta);
        }
        return clamp_probability(p);
    }

    if (source->model == ND_SOURCE_DK2NU && source->dk2nu && source->dk2nu->rows) {
        double weighted = 0.0;
        double denom = 0.0;
        for (int i = 0; i < source->dk2nu->n_rows; ++i) {
            const DuneDk2nuFluxZRow *row = &source->dk2nu->rows[i];
            if (row->flavor != initial ||
                energy_GeV < row->e_low_GeV ||
                energy_GeV >= row->e_high_GeV) {
                continue;
            }
            const double z_m = 0.5 * (row->z_low_m + row->z_high_m);
            double baseline_km = source->detector_baseline_km - z_m * 1.0e-3;
            if (baseline_km < 0.0) {
                baseline_km = 0.0;
            }
            double p = 0.0;
            if (dune_exact_light_probability(point, alpha, beta, energy_GeV, baseline_km, &p) != DUNE_STATUS_OK) {
                p = probability_no_sterile(alpha, beta);
            }
            weighted += row->weight * clamp_probability(p);
            denom += row->weight;
        }
        if (denom > 0.0) {
            return weighted / denom;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < source->source_z_bins; ++i) {
        const double s_m = source->source_z_start_m +
                           (((double)i + 0.5) / (double)source->source_z_bins) * source->decay_pipe_length_m;
        double baseline_km = source->detector_baseline_km - s_m * 1.0e-3;
        if (baseline_km < 0.0) {
            baseline_km = 0.0;
        }

        double p = 0.0;
        if (dune_exact_light_probability(point, alpha, beta, energy_GeV, baseline_km, &p) != DUNE_STATUS_OK) {
            p = probability_no_sterile(alpha, beta);
        }
        sum += clamp_probability(p);
    }

    return sum / (double)source->source_z_bins;
}

static double active_probability_iss23(
    const DuneTheoryPoint *point,
    DuneFluxFlavor initial,
    int alpha,
    double energy_GeV,
    const NdSourceAverage *source) {
    double p = 0.0;
    for (int beta = 0; beta < 3; ++beta) {
        p += probability_iss23(point, initial, alpha, beta, energy_GeV, source);
    }
    return clamp_probability(p);
}

static DuneFluxFlavor electron_flavor(int anti) {
    return anti ? DUNE_FLUX_NUEBAR : DUNE_FLUX_NUE;
}

static DuneFluxFlavor muon_flavor(int anti) {
    return anti ? DUNE_FLUX_NUMUBAR : DUNE_FLUX_NUMU;
}

static int eval_flux(const DuneFluxTable *table, DuneFluxFlavor flavor, double energy_GeV, double *value) {
    return dune_flux_table_eval(table, flavor, energy_GeV, value) == DUNE_STATUS_OK ? 0 : 1;
}

static int eval_xsec(const DuneXsecTable *table, DuneFluxFlavor flavor, double energy_GeV, double *value) {
    double sigma_over_e = 0.0;
    if (dune_xsec_table_eval(table, flavor, energy_GeV, &sigma_over_e) != DUNE_STATUS_OK) {
        return 1;
    }
    *value = sigma_over_e * energy_GeV * ND_XSEC_SCALE_CM2;
    return 0;
}

static double cc_rate(
    const DuneFluxTable *flux_table,
    const DuneXsecTable *cc_table,
    const DuneTheoryPoint *point,
    DuneFluxFlavor initial,
    int alpha,
    int beta,
    DuneFluxFlavor final_flavor,
    double energy_GeV,
    const NdSourceAverage *source,
    double width_GeV,
    int use_oscillation) {
    double flux = 0.0;
    double xsec = 0.0;
    if (eval_flux(flux_table, initial, energy_GeV, &flux) != 0 ||
        eval_xsec(cc_table, final_flavor, energy_GeV, &xsec) != 0) {
        return 0.0;
    }
    const double p = use_oscillation
                         ? probability_iss23(point, initial, alpha, beta, energy_GeV, source)
                         : probability_no_sterile(alpha, beta);
    return (flux / ND_M2_TO_CM2) * xsec * p * width_GeV;
}

static double nc_rate_for_initial(
    const DuneFluxTable *flux_table,
    const DuneXsecTable *nc_table,
    const DuneTheoryPoint *point,
    DuneFluxFlavor initial,
    int alpha,
    double energy_GeV,
    const NdSourceAverage *source,
    double width_GeV,
    int use_oscillation) {
    double flux = 0.0;
    double xsec = 0.0;
    if (eval_flux(flux_table, initial, energy_GeV, &flux) != 0 ||
        eval_xsec(nc_table, initial, energy_GeV, &xsec) != 0) {
        return 0.0;
    }
    const double p_active = use_oscillation ? active_probability_iss23(point, initial, alpha, energy_GeV, source) : 1.0;
    return (flux / ND_M2_TO_CM2) * xsec * p_active * width_GeV;
}

static double nc_rate_all(
    const DuneFluxTable *flux_table,
    const DuneXsecTable *nc_table,
    const DuneTheoryPoint *point,
    double energy_GeV,
    const NdSourceAverage *source,
    double width_GeV,
    int use_oscillation) {
    return nc_rate_for_initial(flux_table, nc_table, point, DUNE_FLUX_NUE, 0, energy_GeV, source, width_GeV, use_oscillation) +
           nc_rate_for_initial(flux_table, nc_table, point, DUNE_FLUX_NUMU, 1, energy_GeV, source, width_GeV, use_oscillation) +
           nc_rate_for_initial(flux_table, nc_table, point, DUNE_FLUX_NUEBAR, 0, energy_GeV, source, width_GeV, use_oscillation) +
           nc_rate_for_initial(flux_table, nc_table, point, DUNE_FLUX_NUMUBAR, 1, energy_GeV, source, width_GeV, use_oscillation);
}

static double component_rate(
    const DuneFluxTable *flux_table,
    const DuneXsecTable *cc_table,
    const DuneXsecTable *nc_table,
    const DuneTheoryPoint *point,
    const char *panel,
    const char *component,
    double energy_GeV,
    const NdSourceAverage *source,
    double width_GeV,
    int use_oscillation) {
    const int is_rhc = strncmp(panel, "RHC", 3) == 0;
    const int is_app = strstr(panel, "_app") != NULL;
    const int right_anti = is_rhc ? 1 : 0;
    const int wrong_anti = right_anti ? 0 : 1;
    const DuneFluxFlavor right_mu = muon_flavor(right_anti);
    const DuneFluxFlavor wrong_mu = muon_flavor(wrong_anti);
    const DuneFluxFlavor right_e = electron_flavor(right_anti);
    const DuneFluxFlavor wrong_e = electron_flavor(wrong_anti);

    if (strcmp(component, "nc") == 0) {
        const double nc_selection = is_app ? 0.025 : 0.015;
        return nc_selection * nc_rate_all(flux_table, nc_table, point, energy_GeV, source, width_GeV, use_oscillation);
    }

    if (is_app && strcmp(component, "numu") == 0) {
        const double mis_id = 0.015;
        return mis_id * (
            cc_rate(flux_table, cc_table, point, right_mu, 1, 1, right_mu, energy_GeV, source, width_GeV, use_oscillation) +
            cc_rate(flux_table, cc_table, point, wrong_mu, 1, 1, wrong_mu, energy_GeV, source, width_GeV, use_oscillation));
    }

    if (is_app && strcmp(component, "beam") == 0) {
        return cc_rate(flux_table, cc_table, point, right_e, 0, 0, right_e, energy_GeV, source, width_GeV, use_oscillation) +
               cc_rate(flux_table, cc_table, point, wrong_e, 0, 0, wrong_e, energy_GeV, source, width_GeV, use_oscillation);
    }

    if (is_app && strcmp(component, "signal") == 0) {
        return cc_rate(flux_table, cc_table, point, right_mu, 1, 0, right_e, energy_GeV, source, width_GeV, use_oscillation) +
               cc_rate(flux_table, cc_table, point, wrong_mu, 1, 0, wrong_e, energy_GeV, source, width_GeV, use_oscillation);
    }

    if (!is_app && strcmp(component, "wrong_mu") == 0) {
        return cc_rate(flux_table, cc_table, point, wrong_mu, 1, 1, wrong_mu, energy_GeV, source, width_GeV, use_oscillation);
    }

    if (!is_app && strcmp(component, "tau") == 0) {
        return 0.0;
    }

    if (!is_app && strcmp(component, "signal") == 0) {
        return cc_rate(flux_table, cc_table, point, right_mu, 1, 1, right_mu, energy_GeV, source, width_GeV, use_oscillation);
    }

    return 0.0;
}

static int append_row(
    NdFig4Row *rows,
    int *n_rows,
    const char *panel,
    const char *component,
    double energy_GeV,
    double globes_events,
    double iss23_events) {
    if (*n_rows >= ND_FIG4_MAX_ROWS) {
        return 1;
    }
    NdFig4Row *row = &rows[(*n_rows)++];
    memset(row, 0, sizeof(*row));
    strncpy(row->panel, panel, sizeof(row->panel) - 1);
    strncpy(row->component, component, sizeof(row->component) - 1);
    row->energy_GeV = energy_GeV;
    row->globes_events = globes_events;
    row->iss23_events = iss23_events;
    return 0;
}

static double stacked_total_for_row_set(const NdFig4Row *rows, int n_rows, const char *panel, double energy_GeV, int iss) {
    double total = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        if (strcmp(rows[i].panel, panel) == 0 && fabs(rows[i].energy_GeV - energy_GeV) < 1e-12) {
            total += iss ? rows[i].iss23_events : rows[i].globes_events;
        }
    }
    return total;
}

static void write_probability_row(
    FILE *out,
    int point_id,
    const char *source_model,
    double detector_baseline_km,
    double source_z_start_m,
    double decay_pipe_length_m,
    int source_z_bins,
    const char *panel,
    const char *channel,
    DuneFluxFlavor initial,
    int alpha,
    int beta,
    double energy_GeV,
    const DuneTheoryPoint *benchmark3nu,
    const DuneTheoryPoint *iss23,
    const NdSourceAverage *source) {
    const double p3 = beta >= 0
                          ? probability_iss23(benchmark3nu, initial, alpha, beta, energy_GeV, source)
                          : active_probability_iss23(benchmark3nu, initial, alpha, energy_GeV, source);
    const double p4 = beta >= 0
                          ? probability_iss23(iss23, initial, alpha, beta, energy_GeV, source)
                          : active_probability_iss23(iss23, initial, alpha, energy_GeV, source);
    fprintf(out,
            "%d,ND,%s,%.10g,%.10g,%.10g,%d,%s,%s,%s,%d,%d,%.10g,%.12g,%.12g,%.12g\n",
            point_id,
            source_model,
            detector_baseline_km,
            source_z_start_m,
            decay_pipe_length_m,
            source_z_bins,
            panel,
            channel,
            flux_flavor_name(initial),
            alpha,
            beta,
            energy_GeV,
            p3,
            p4,
            p4 - p3);
}

static void debug_compare_flux_sums(
    const char *label,
    const DuneFluxTable *flux_table,
    const DuneDk2nuFluxZTable *dk2nu_table) {
    if (!flux_table || !dk2nu_table || !dk2nu_table->rows) {
        return;
    }
    const double energies[] = {1.125, 2.375, 4.875};
    const DuneFluxFlavor flavors[] = {DUNE_FLUX_NUMU, DUNE_FLUX_NUE, DUNE_FLUX_NUMUBAR, DUNE_FLUX_NUEBAR};
    const char *names[] = {"numu", "nue", "numubar", "nuebar"};
    fprintf(stderr,
            "DUNE ND dk2nu %s: rows=%d, E=[%.4g, %.4g] GeV, z=[%.4g, %.4g] m\n",
            label,
            dk2nu_table->n_rows,
            dk2nu_table->e_min_GeV,
            dk2nu_table->e_max_GeV,
            dk2nu_table->z_min_m,
            dk2nu_table->z_max_m);
    for (int ie = 0; ie < (int)(sizeof(energies) / sizeof(energies[0])); ++ie) {
        for (int jf = 0; jf < (int)(sizeof(flavors) / sizeof(flavors[0])); ++jf) {
            double tabulated = 0.0;
            if (dune_flux_table_eval(flux_table, flavors[jf], energies[ie], &tabulated) != DUNE_STATUS_OK || tabulated <= 0.0) {
                continue;
            }
            const double weighted = dune_dk2nu_flux_z_weight_sum(dk2nu_table, flavors[jf], energies[ie]);
            if (weighted > 0.0) {
                fprintf(stderr,
                        "  %s E=%.2f GeV: sum_z w=%.6g, flux_table=%.6g, ratio=%.6g\n",
                        names[jf],
                        energies[ie],
                        weighted,
                        tabulated,
                        weighted / tabulated);
            }
        }
    }
}

int run_dune_nd_fig4_source_line(const SimulationConfig *cfg) {
    if (!cfg || cfg->dune_spectrum_pred_csv[0] == '\0') {
        fprintf(stderr, "DUNE ND Fig.4-like: chemin output.spectrum_pred_csv manquant\n");
        return 1;
    }

    DuneRunContext ctx;
    DuneStatus status = dune_run_context_init(&ctx, cfg);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND Fig.4-like init failed: %s\n", dune_status_message(status));
        return 1;
    }
    status = dune_theory_point_load(&ctx);
    if (status != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND Fig.4-like point load failed: %s\n", dune_status_message(status));
        return 1;
    }

    static DuneFluxTable fhc_flux;
    static DuneFluxTable rhc_flux;
    static DuneXsecTable cc_table;
    static DuneXsecTable nc_table;
    if (dune_flux_table_load_globes(cfg->dune_flux_fhc_file, &fhc_flux) != DUNE_STATUS_OK ||
        dune_flux_table_load_globes(cfg->dune_flux_rhc_file, &rhc_flux) != DUNE_STATUS_OK ||
        dune_xsec_table_load_globes(cfg->dune_xsec_cc_file, &cc_table) != DUNE_STATUS_OK ||
        dune_xsec_table_load_globes(cfg->dune_xsec_nc_file, &nc_table) != DUNE_STATUS_OK) {
        fprintf(stderr, "DUNE ND Fig.4-like: impossible de charger flux/xsec depuis le preset\n");
        return 1;
    }

    const int n_bins = cfg->dune_Erec_bins > 0 ? cfg->dune_Erec_bins : 30;
    const double e_min = cfg->dune_Erec_min_GeV > 0.0 ? cfg->dune_Erec_min_GeV : 0.5;
    const double e_max = cfg->dune_Erec_max_GeV > e_min ? cfg->dune_Erec_max_GeV : 8.0;
    const double width = (e_max - e_min) / (double)n_bins;
    const double detector_baseline_km = cfg->dune_detector_distance_m > 0.0 ? cfg->dune_detector_distance_m * 1.0e-3 : cfg->baseline_km;
    const double source_z_start_m = cfg->dune_source_z_start_m > 0.0 ? cfg->dune_source_z_start_m : 0.0;
    const double decay_pipe_length_m = cfg->dune_decay_pipe_length_m > 0.0 ? cfg->dune_decay_pipe_length_m : 194.0;
    const int source_z_bins = cfg->dune_source_z_bins > 1 ? cfg->dune_source_z_bins : 80;
    NdSourceModel source_model = parse_source_model(cfg->dune_source_model);
    if (n_bins <= 0 || n_bins * 16 > ND_FIG4_MAX_ROWS || width <= 0.0 || detector_baseline_km <= 0.0 || decay_pipe_length_m <= 0.0) {
        fprintf(stderr, "DUNE ND Fig.4-like: binning ou baseline invalide\n");
        return 1;
    }

    static DuneDk2nuFluxZTable fhc_dk2nu;
    static DuneDk2nuFluxZTable rhc_dk2nu;
    int have_dk2nu = 0;
    if (source_model == ND_SOURCE_DK2NU) {
        const DuneStatus fhc_status = dune_dk2nu_flux_z_load_csv(cfg->dune_dk2nu_flux_z_fhc_file, &fhc_dk2nu);
        const DuneStatus rhc_status = dune_dk2nu_flux_z_load_csv(cfg->dune_dk2nu_flux_z_rhc_file, &rhc_dk2nu);
        if (fhc_status == DUNE_STATUS_OK && rhc_status == DUNE_STATUS_OK) {
            have_dk2nu = 1;
            fprintf(stderr, "DUNE ND Fig.4-like: source_model=dk2nu actif\n");
            debug_compare_flux_sums("FHC", &fhc_flux, &fhc_dk2nu);
            debug_compare_flux_sums("RHC", &rhc_flux, &rhc_dk2nu);
        } else {
            fprintf(stderr,
                    "DUNE ND Fig.4-like: source_model=dk2nu demande mais CSV flux_z indisponibles "
                    "(FHC='%s', RHC='%s'); fallback source_model=uniform.\n",
                    cfg->dune_dk2nu_flux_z_fhc_file,
                    cfg->dune_dk2nu_flux_z_rhc_file);
            dune_dk2nu_flux_z_free(&fhc_dk2nu);
            dune_dk2nu_flux_z_free(&rhc_dk2nu);
            source_model = ND_SOURCE_UNIFORM;
        }
    }

    const char *panels[] = {"FHC_app", "RHC_app", "FHC_dis", "RHC_dis"};
    const char *app_components[] = {"nc", "numu", "beam", "signal"};
    const char *dis_components[] = {"nc", "wrong_mu", "tau", "signal"};

    NdFig4Row rows[ND_FIG4_MAX_ROWS];
    int n_rows = 0;
    DuneTheoryPoint benchmark3nu;
    build_active_3nu_benchmark_point(&ctx.theory, &benchmark3nu);
    for (int p = 0; p < 4; ++p) {
        const int is_rhc = strncmp(panels[p], "RHC", 3) == 0;
        const int is_app = strstr(panels[p], "_app") != NULL;
        const DuneFluxTable *flux = is_rhc ? &rhc_flux : &fhc_flux;
        NdSourceAverage source;
        memset(&source, 0, sizeof(source));
        source.model = source_model;
        source.detector_baseline_km = detector_baseline_km;
        source.source_z_start_m = source_z_start_m;
        source.decay_pipe_length_m = decay_pipe_length_m;
        source.source_z_bins = source_z_bins;
        source.dk2nu = (source_model == ND_SOURCE_DK2NU && have_dk2nu) ? (is_rhc ? &rhc_dk2nu : &fhc_dk2nu) : NULL;
        const char **components = is_app ? app_components : dis_components;
        for (int b = 0; b < n_bins; ++b) {
            const double energy = e_min + ((double)b + 0.5) * width;
            for (int c = 0; c < 4; ++c) {
                const double globes = component_rate(flux, &cc_table, &nc_table, &benchmark3nu, panels[p], components[c], energy, &source, width, 1);
                const double iss = component_rate(flux, &cc_table, &nc_table, &ctx.theory, panels[p], components[c], energy, &source, width, 1);
                if (append_row(rows, &n_rows, panels[p], components[c], energy, globes, iss) != 0) {
                    fprintf(stderr, "DUNE ND Fig.4-like: trop de lignes\n");
                    return 1;
                }
            }
        }
    }

    const double exposure_pot = ND_POT_PER_YEAR * ND_EXPOSURE_YEARS;
    const double target_nucleons = ND_TARGET_MASS_KT * 1.0e9 * ND_AVOGADRO;
    const double event_scale = exposure_pot * target_nucleons;

    FILE *out = fopen(cfg->dune_spectrum_pred_csv, "w");
    if (!out) {
        fprintf(stderr, "DUNE ND Fig.4-like: impossible d'ouvrir %s\n", cfg->dune_spectrum_pred_csv);
        return 1;
    }
    fprintf(out, "point_id,detector,source_model,baseline_km,source_z_start_m,decay_pipe_length_m,source_z_bins,panel,component,Erec_GeV,globes_events,iss23_events,ratio_iss_over_3nu,rel_diff,delta_events\n");
    double max_abs_rel = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        const double g = rows[i].globes_events * event_scale;
        const double y = rows[i].iss23_events * event_scale;
        const double rel = g > 0.0 ? (y - g) / g : 0.0;
        const double ratio = g > 0.0 ? y / g : 0.0;
        if (fabs(rel) > max_abs_rel) max_abs_rel = fabs(rel);
        fprintf(out, "%d,ND,%s,%.10g,%.10g,%.10g,%d,%s,%s,%.10g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                ctx.theory.point_id,
                source_model_name(source_model),
                detector_baseline_km,
                source_z_start_m,
                decay_pipe_length_m,
                source_z_bins,
                rows[i].panel,
                rows[i].component,
                rows[i].energy_GeV,
                g,
                y,
                ratio,
                rel,
                y - g);
    }
    fclose(out);

    char prob_csv[256];
    if (probability_csv_path(cfg->dune_spectrum_pred_csv, prob_csv, sizeof(prob_csv)) == 0) {
        FILE *pout = fopen(prob_csv, "w");
        if (!pout) {
            fprintf(stderr, "DUNE ND Fig.4-like: impossible d'ouvrir %s\n", prob_csv);
            return 1;
        }
        fprintf(pout, "point_id,detector,source_model,baseline_km,source_z_start_m,decay_pipe_length_m,source_z_bins,panel,channel,initial_flavor,alpha,beta,E_GeV,benchmark3nu_probability,iss23_probability,delta_probability\n");
        for (int p = 0; p < 4; ++p) {
            const int is_rhc = strncmp(panels[p], "RHC", 3) == 0;
            const int right_anti = is_rhc ? 1 : 0;
            const int wrong_anti = right_anti ? 0 : 1;
            const DuneFluxFlavor right_mu = muon_flavor(right_anti);
            const DuneFluxFlavor wrong_mu = muon_flavor(wrong_anti);
            const DuneFluxFlavor right_e = electron_flavor(right_anti);
            const DuneFluxFlavor wrong_e = electron_flavor(wrong_anti);
            NdSourceAverage source;
            memset(&source, 0, sizeof(source));
            source.model = source_model;
            source.detector_baseline_km = detector_baseline_km;
            source.source_z_start_m = source_z_start_m;
            source.decay_pipe_length_m = decay_pipe_length_m;
            source.source_z_bins = source_z_bins;
            source.dk2nu = (source_model == ND_SOURCE_DK2NU && have_dk2nu) ? (is_rhc ? &rhc_dk2nu : &fhc_dk2nu) : NULL;
            for (int b = 0; b < n_bins; ++b) {
                const double energy = e_min + ((double)b + 0.5) * width;
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "right_mu_to_e", right_mu, 1, 0, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "right_mu_to_mu", right_mu, 1, 1, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "right_mu_active", right_mu, 1, -1, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "right_e_to_e", right_e, 0, 0, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "right_e_active", right_e, 0, -1, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "wrong_mu_to_e", wrong_mu, 1, 0, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "wrong_mu_to_mu", wrong_mu, 1, 1, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "wrong_mu_active", wrong_mu, 1, -1, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "wrong_e_to_e", wrong_e, 0, 0, energy, &benchmark3nu, &ctx.theory, &source);
                write_probability_row(pout, ctx.theory.point_id, source_model_name(source_model), detector_baseline_km, source_z_start_m, decay_pipe_length_m, source_z_bins, panels[p], "wrong_e_active", wrong_e, 0, -1, energy, &benchmark3nu, &ctx.theory, &source);
            }
        }
        fclose(pout);
        fprintf(stderr, "DUNE ND Fig.4-like raw probabilities: csv=%s\n", prob_csv);
    }

    fprintf(stderr,
            "DUNE ND Fig.4-like source-line: ok (point=%d, L_ND=%.6g km, z_start=%.6g m, l_dec=%.6g m, bins=%d, rows=%d, scale=%.6g, max_abs_rel=%.6g, csv=%s)\n",
            ctx.theory.point_id,
            detector_baseline_km,
            source_z_start_m,
            decay_pipe_length_m,
            source_z_bins,
            n_rows,
            event_scale,
            max_abs_rel,
            cfg->dune_spectrum_pred_csv);
    dune_dk2nu_flux_z_free(&fhc_dk2nu);
    dune_dk2nu_flux_z_free(&rhc_dk2nu);
    return 0;
}
