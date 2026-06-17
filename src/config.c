#include "config.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONFIG_INCLUDE_MAX_DEPTH 8

static const char *recommended_preset_for_operation(SimulationOperation operation) {
    switch (operation) {
        case OPERATION_ENERGY_3P1:
            return "config/presets/oscillations/3p1/energy.txt";
        case OPERATION_ENERGY_3P2:
            return "config/presets/oscillations/3p2/energy.txt";
        case OPERATION_DISTANCE_3P1:
            return "config/presets/oscillations/3p1/distance.txt";
        case OPERATION_HEATMAP_DELTA_PMUMU_3P2:
            return "config/presets/oscillations/3p2/heatmap_delta_pmumu.txt";
        case OPERATION_CP_HEATMAP_3P1:
            return "config/presets/oscillations/3p1/heatmap_cp.txt";
        case OPERATION_INVERSE_PMNS_FILTER_3P1:
            return "config/presets/inverse_seesaw/3p1/pmns_filter.txt";
        case OPERATION_INVERSE_PMNS_FILTER_3P2:
            return "config/presets/inverse_seesaw/3p2/pmns_filter.txt";
        case OPERATION_INVERSE_CONSTRUCT_23_3P1:
            return "config/presets/inverse_seesaw/3p1/construct_23.txt";
        case OPERATION_DUNE_ND_PREDICT_SPECTRUM:
        case OPERATION_DUNE_FD_FIG4_VALIDATION:
        case OPERATION_DUNE_ND_FIG4_SOURCE_LINE:
            return "config/presets/dune/nd/minimal_onaxis.txt";
        default:
            return "config/presets/oscillations/3p1/energy.txt";
    }
}

static int is_removed_legacy_key(const char *key) {
    return (strcmp(key, "sin22theta14_values") == 0 ||
            strcmp(key, "sin22theta14_range_enabled") == 0 ||
            strcmp(key, "sin22theta14_range_min") == 0 ||
            strcmp(key, "sin22theta14_range_max") == 0 ||
            strcmp(key, "sin22theta14_range_step") == 0 ||
            strcmp(key, "sin22theta14_range_points") == 0);
}

static void trim_in_place(char *text) {
    size_t len = strlen(text);
    while (len > 0 && isspace((unsigned char)text[len - 1])) {
        text[len - 1] = '\0';
        --len;
    }

    size_t start = 0;
    while (text[start] != '\0' && isspace((unsigned char)text[start])) {
        ++start;
    }

    if (start > 0) {
        memmove(text, text + start, strlen(text + start) + 1);
    }
}

static void strip_optional_quotes(char *text) {
    size_t len = strlen(text);
    if (len >= 2) {
        const char first = text[0];
        const char last = text[len - 1];
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            memmove(text, text + 1, len - 2);
            text[len - 2] = '\0';
        }
    }
}

static int is_absolute_path(const char *path) {
    if (!path || path[0] == '\0') {
        return 0;
    }

    return ((isalpha((unsigned char)path[0]) && path[1] == ':') || path[0] == '\\' || path[0] == '/');
}

static void build_include_path(const char *current_file, const char *include_value, char *out, size_t out_size) {
    const char *slash1 = NULL;
    const char *slash2 = NULL;
    const char *slash = NULL;

    if (!out || out_size == 0) {
        return;
    }

    if (is_absolute_path(include_value)) {
        strncpy(out, include_value, out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    slash1 = strrchr(current_file, '/');
    slash2 = strrchr(current_file, '\\');
    slash = slash1;
    if (!slash || (slash2 && slash2 > slash)) {
        slash = slash2;
    }

    if (!slash) {
        strncpy(out, include_value, out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    {
        const size_t prefix_len = (size_t)(slash - current_file + 1);
        if (prefix_len >= out_size) {
            out[0] = '\0';
            return;
        }

        memcpy(out, current_file, prefix_len);
        out[prefix_len] = '\0';
        strncat(out, include_value, out_size - strlen(out) - 1);
    }
}

static int parse_double(const char *value_str, double *out) {
    char *end_ptr = NULL;
    const double value = strtod(value_str, &end_ptr);
    if (end_ptr == value_str) {
        return 1;
    }

    while (*end_ptr != '\0') {
        if (!isspace((unsigned char)*end_ptr)) {
            return 1;
        }
        ++end_ptr;
    }

    *out = value;
    return 0;
}

static int parse_int(const char *value_str, int *out) {
    char *end_ptr = NULL;
    const long value = strtol(value_str, &end_ptr, 10);
    if (end_ptr == value_str) {
        return 1;
    }

    while (*end_ptr != '\0') {
        if (!isspace((unsigned char)*end_ptr)) {
            return 1;
        }
        ++end_ptr;
    }

    *out = (int)value;
    return 0;
}

static int parse_operation(const char *value_str, SimulationOperation *out) {
    if (strcmp(value_str, "energy_3p1") == 0) {
        *out = OPERATION_ENERGY_3P1;
        return 0;
    }
    if (strcmp(value_str, "energy_3p2") == 0) {
        *out = OPERATION_ENERGY_3P2;
        return 0;
    }
    if (strcmp(value_str, "distance_3p1") == 0) {
        *out = OPERATION_DISTANCE_3P1;
        return 0;
    }
    if (strcmp(value_str, "heatmap_delta_pmu_e") == 0) {
        *out = OPERATION_HEATMAP_DELTA_PMUE;
        return 0;
    }
    if (strcmp(value_str, "heatmap_delta_pmu_e_3p2") == 0) {
        *out = OPERATION_HEATMAP_DELTA_PMUE_3P2;
        return 0;
    }
    if (strcmp(value_str, "heatmap_delta_pmu_mu_3p2") == 0) {
        *out = OPERATION_HEATMAP_DELTA_PMUMU_3P2;
        return 0;
    }
    if (strcmp(value_str, "cp_heatmap_3p1") == 0) {
        *out = OPERATION_CP_HEATMAP_3P1;
        return 0;
    }
    if (strcmp(value_str, "inverse_pmns_filter_3p1") == 0) {
        *out = OPERATION_INVERSE_PMNS_FILTER_3P1;
        return 0;
    }
    if (strcmp(value_str, "inverse_pmns_filter_3p2") == 0) {
        *out = OPERATION_INVERSE_PMNS_FILTER_3P2;
        return 0;
    }
    if (strcmp(value_str, "inverse_construct_23_3p1") == 0) {
        *out = OPERATION_INVERSE_CONSTRUCT_23_3P1;
        return 0;
    }
    if (strcmp(value_str, "dune_nd_predict_spectrum") == 0) {
        *out = OPERATION_DUNE_ND_PREDICT_SPECTRUM;
        return 0;
    }
    if (strcmp(value_str, "dune_fd_fig4_validation") == 0) {
        *out = OPERATION_DUNE_FD_FIG4_VALIDATION;
        return 0;
    }
    if (strcmp(value_str, "dune_nd_fig4_source_line") == 0 ||
        strcmp(value_str, "dune_nd_fig4_point_source") == 0) {
        *out = OPERATION_DUNE_ND_FIG4_SOURCE_LINE;
        return 0;
    }
    return 1;
}

static int parse_double_list(const char *value_str, double *out, int max_count, int *out_count) {
    char buffer[1024];
    strncpy(buffer, value_str, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    int count = 0;
    char *token = strtok(buffer, ",");
    while (token != NULL) {
        if (count >= max_count) {
            return 1;
        }

        trim_in_place(token);
        if (parse_double(token, &out[count]) != 0) {
            return 1;
        }

        ++count;
        token = strtok(NULL, ",");
    }

    if (count == 0) {
        return 1;
    }

    *out_count = count;
    return 0;
}

static int infer_sterile_count_from_pairs(int pair_count) {
    for (int n = 1; n <= MAX_STERILE_NEUTRINOS; ++n) {
        if ((n * (n - 1)) / 2 == pair_count) {
            return n;
        }
    }
    return 0;
}

static int set_active_sterile_parameter(
    SimulationConfig *cfg,
    int active_index,
    int sterile_index,
    double value,
    int is_phase) {

    if (active_index < 0 || active_index >= 3) {
        return 1;
    }
    if (sterile_index < 0 || sterile_index >= MAX_STERILE_NEUTRINOS) {
        return 1;
    }

    if (is_phase) {
        cfg->delta_active_sterile_deg[active_index][sterile_index] = value;
    } else {
        cfg->theta_active_sterile_deg[active_index][sterile_index] = value;
    }

    if (cfg->n_sterile < sterile_index + 1) {
        cfg->n_sterile = sterile_index + 1;
    }

    return 0;
}

static int set_sterile_sterile_parameter(
    SimulationConfig *cfg,
    int sterile_i,
    int sterile_j,
    double value,
    int is_phase) {

    if (sterile_i < 0 || sterile_j < 0 || sterile_i >= MAX_STERILE_NEUTRINOS || sterile_j >= MAX_STERILE_NEUTRINOS || sterile_i == sterile_j) {
        return 1;
    }

    int i = sterile_i;
    int j = sterile_j;
    if (i > j) {
        const int tmp = i;
        i = j;
        j = tmp;
    }

    if (is_phase) {
        cfg->delta_sterile_sterile_deg[i][j] = value;
    } else {
        cfg->theta_sterile_sterile_deg[i][j] = value;
    }

    if (cfg->n_sterile < j + 1) {
        cfg->n_sterile = j + 1;
    }

    return 0;
}

static int parse_dm41_list(const char *value_str, SimulationConfig *cfg) {
    char buffer[512];
    strncpy(buffer, value_str, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    int count = 0;
    char *token = strtok(buffer, ",");
    while (token != NULL) {
        if (count >= MAX_DM41_VALUES) {
            return 1;
        }

        trim_in_place(token);
        if (parse_double(token, &cfg->dm41_values_eV2[count]) != 0) {
            return 1;
        }

        ++count;
        token = strtok(NULL, ",");
    }

    if (count == 0) {
        return 1;
    }

    cfg->dm41_count = count;
    return 0;
}

static int parse_dm54_list(const char *value_str, SimulationConfig *cfg) {
    return parse_double_list(value_str, cfg->dm54_values_eV2, MAX_DM54_VALUES, &cfg->dm54_count);
}

static int parse_dm41_3p2_value(const char *value_str, SimulationConfig *cfg) {
    if (parse_double(value_str, &cfg->dm41_3p2_eV2) != 0) {
        return 1;
    }
    cfg->dm41_3p2_is_set = 1;
    return 0;
}

static int parse_dm41_heatmap_3p2_list(const char *value_str, SimulationConfig *cfg) {
    return parse_double_list(value_str, cfg->dm41_heatmap_3p2_values_eV2, 16, &cfg->dm41_heatmap_3p2_count);
}

static int parse_inverse_pmns_abs_min_3x3(const char *value_str, SimulationConfig *cfg) {
    double values[9];
    int count = 0;
    if (parse_double_list(value_str, values, 9, &count) != 0 || count != 9) {
        return 1;
    }

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cfg->inverse_pmns_abs_min_3x3[i][j] = values[idx++];
        }
    }
    return 0;
}

static int parse_inverse_pmns_abs_max_3x3(const char *value_str, SimulationConfig *cfg) {
    double values[9];
    int count = 0;
    if (parse_double_list(value_str, values, 9, &count) != 0 || count != 9) {
        return 1;
    }

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cfg->inverse_pmns_abs_max_3x3[i][j] = values[idx++];
        }
    }
    return 0;
}

static int parse_inverse_eta_matrix_3x3(const char *value_str, double out[3][3]) {
    double values[9];
    int count = 0;
    if (parse_double_list(value_str, values, 9, &count) != 0 || count != 9) {
        return 1;
    }

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = values[idx++];
        }
    }
    return 0;
}

static int parse_inverse_eta_abs_max_3x3(const char *value_str, SimulationConfig *cfg) {
    if (parse_inverse_eta_matrix_3x3(value_str, cfg->inverse_eta_abs_max_3x3) != 0) {
        return 1;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double value = cfg->inverse_eta_abs_max_3x3[i][j];
            cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j] = value;
            cfg->inverse_eta_abs_max_light_highdm_3x3[i][j] = value;
            cfg->inverse_eta_abs_max_light_lowdm_3x3[i][j] = value;
        }
    }

    return 0;
}

static int parse_inverse_eta_abs_max_nonunitarity_3x3(const char *value_str, SimulationConfig *cfg) {
    return parse_inverse_eta_matrix_3x3(value_str, cfg->inverse_eta_abs_max_nonunitarity_3x3);
}

static int parse_inverse_eta_abs_max_light_highdm_3x3(const char *value_str, SimulationConfig *cfg) {
    return parse_inverse_eta_matrix_3x3(value_str, cfg->inverse_eta_abs_max_light_highdm_3x3);
}

static int parse_inverse_eta_abs_max_light_lowdm_3x3(const char *value_str, SimulationConfig *cfg) {
    return parse_inverse_eta_matrix_3x3(value_str, cfg->inverse_eta_abs_max_light_lowdm_3x3);
}

static int build_dm41_values_from_range(SimulationConfig *cfg) {
    if (!cfg || cfg->dm41_range_max_eV2 < cfg->dm41_range_min_eV2) {
        return 1;
    }

    int count = 0;
    if (cfg->dm41_range_logspace) {
        if (cfg->dm41_range_min_eV2 <= 0.0 || cfg->dm41_range_max_eV2 <= 0.0 || cfg->dm41_range_points < 2 || cfg->dm41_range_points > MAX_DM41_VALUES) {
            return 1;
        }

        const double log_min = log10(cfg->dm41_range_min_eV2);
        const double log_max = log10(cfg->dm41_range_max_eV2);
        const double dlog = (log_max - log_min) / (cfg->dm41_range_points - 1);

        for (int i = 0; i < cfg->dm41_range_points; ++i) {
            cfg->dm41_values_eV2[count++] = pow(10.0, log_min + i * dlog);
        }
    } else {
        if (cfg->dm41_range_points >= 2) {
            if (cfg->dm41_range_points > MAX_DM41_VALUES) {
                return 1;
            }

            const double step = (cfg->dm41_range_max_eV2 - cfg->dm41_range_min_eV2) / (cfg->dm41_range_points - 1);
            for (int i = 0; i < cfg->dm41_range_points; ++i) {
                cfg->dm41_values_eV2[count++] = cfg->dm41_range_min_eV2 + i * step;
            }
        } else {
            if (cfg->dm41_range_step_eV2 <= 0.0) {
                return 1;
            }

            for (double value = cfg->dm41_range_min_eV2;
                 value <= cfg->dm41_range_max_eV2 + 1e-12;
                 value += cfg->dm41_range_step_eV2) {

                if (count >= MAX_DM41_VALUES) {
                    return 1;
                }
                cfg->dm41_values_eV2[count] = value;
                ++count;
            }
        }
    }

    if (count <= 0) {
        return 1;
    }

    cfg->dm41_count = count;
    return 0;
}

static int build_dm54_values_from_range(SimulationConfig *cfg) {
    if (!cfg || cfg->dm54_range_max_eV2 < cfg->dm54_range_min_eV2) {
        return 1;
    }

    int count = 0;
    if (cfg->dm54_range_logspace) {
        if (cfg->dm54_range_min_eV2 <= 0.0 || cfg->dm54_range_max_eV2 <= 0.0 || cfg->dm54_range_points < 2 || cfg->dm54_range_points > MAX_DM54_VALUES) {
            return 1;
        }

        const double log_min = log10(cfg->dm54_range_min_eV2);
        const double log_max = log10(cfg->dm54_range_max_eV2);
        const double dlog = (log_max - log_min) / (cfg->dm54_range_points - 1);

        for (int i = 0; i < cfg->dm54_range_points; ++i) {
            cfg->dm54_values_eV2[count++] = pow(10.0, log_min + i * dlog);
        }
    } else {
        if (cfg->dm54_range_points >= 2) {
            if (cfg->dm54_range_points > MAX_DM54_VALUES) {
                return 1;
            }

            const double step = (cfg->dm54_range_max_eV2 - cfg->dm54_range_min_eV2) / (cfg->dm54_range_points - 1);
            for (int i = 0; i < cfg->dm54_range_points; ++i) {
                cfg->dm54_values_eV2[count++] = cfg->dm54_range_min_eV2 + i * step;
            }
        } else {
            if (cfg->dm54_range_step_eV2 <= 0.0) {
                return 1;
            }

            for (double value = cfg->dm54_range_min_eV2;
                 value <= cfg->dm54_range_max_eV2 + 1e-12;
                 value += cfg->dm54_range_step_eV2) {

                if (count >= MAX_DM54_VALUES) {
                    return 1;
                }
                cfg->dm54_values_eV2[count] = value;
                ++count;
            }
        }
    }

    if (count <= 0) {
        return 1;
    }

    cfg->dm54_count = count;
    return 0;
}

static int build_delta41_values_from_range(SimulationConfig *cfg) {
    if (!cfg || cfg->delta41_range_max_deg < cfg->delta41_range_min_deg) {
        return 1;
    }

    int count = 0;
    if (cfg->delta41_range_points >= 2) {
        if (cfg->delta41_range_points > 3600) {
            return 1;
        }
        const double step = (cfg->delta41_range_max_deg - cfg->delta41_range_min_deg) / (cfg->delta41_range_points - 1);
        for (int i = 0; i < cfg->delta41_range_points; ++i) {
            cfg->delta41_values_deg[count++] = cfg->delta41_range_min_deg + i * step;
        }
    } else {
        return 1;
    }

    if (count <= 0) {
        return 1;
    }

    cfg->delta41_count = count;
    return 0;
}

static int parse_active_sterile_list(SimulationConfig *cfg, int active_index, const char *value_str, int is_phase) {
    double values[MAX_STERILE_NEUTRINOS];
    int count = 0;
    if (parse_double_list(value_str, values, MAX_STERILE_NEUTRINOS, &count) != 0) {
        return 1;
    }

    if (cfg->n_sterile == 0) {
        cfg->n_sterile = count;
    } else if (count != cfg->n_sterile) {
        return 1;
    }

    for (int sterile_index = 0; sterile_index < count; ++sterile_index) {
        if (set_active_sterile_parameter(cfg, active_index, sterile_index, values[sterile_index], is_phase) != 0) {
            return 1;
        }
    }

    return 0;
}

static int parse_sterile_pair_list(SimulationConfig *cfg, const char *value_str, int is_phase) {
    double values[(MAX_STERILE_NEUTRINOS * (MAX_STERILE_NEUTRINOS - 1)) / 2];
    int count = 0;
    if (parse_double_list(value_str, values, (int)(sizeof(values) / sizeof(values[0])), &count) != 0) {
        return 1;
    }

    int n_sterile_from_pairs = 0;
    if (cfg->n_sterile == 0) {
        n_sterile_from_pairs = infer_sterile_count_from_pairs(count);
        if (n_sterile_from_pairs == 0) {
            return 1;
        }
        cfg->n_sterile = n_sterile_from_pairs;
    }

    const int expected_count = (cfg->n_sterile * (cfg->n_sterile - 1)) / 2;
    if (count != expected_count) {
        return 1;
    }

    int idx = 0;
    for (int i = 0; i < cfg->n_sterile; ++i) {
        for (int j = i + 1; j < cfg->n_sterile; ++j) {
            if (set_sterile_sterile_parameter(cfg, i, j, values[idx], is_phase) != 0) {
                return 1;
            }
            ++idx;
        }
    }

    return 0;
}

static int parse_legacy_active_sterile(SimulationConfig *cfg, const char *value, int active_index, int sterile_index, int is_phase) {
    double parsed_value = 0.0;
    if (parse_double(value, &parsed_value) != 0) {
        return 1;
    }
    return set_active_sterile_parameter(cfg, active_index, sterile_index, parsed_value, is_phase);
}

static int parse_legacy_sterile_sterile(SimulationConfig *cfg, const char *value, int sterile_i, int sterile_j, int is_phase) {
    double parsed_value = 0.0;
    if (parse_double(value, &parsed_value) != 0) {
        return 1;
    }
    return set_sterile_sterile_parameter(cfg, sterile_i, sterile_j, parsed_value, is_phase);
}

static int set_key_value(SimulationConfig *cfg, const char *key, const char *value) {
    if (strcmp(key, "operation") == 0) return parse_operation(value, &cfg->operation);

    if (strcmp(key, "baseline_km") == 0) return parse_double(value, &cfg->baseline_km);
    if (strcmp(key, "baseline_values_km") == 0) return parse_double_list(value, cfg->baseline_values_km, 16, &cfg->baseline_count);
    if (strcmp(key, "energy_min_GeV") == 0) return parse_double(value, &cfg->energy_min_GeV);
    if (strcmp(key, "energy_max_GeV") == 0) return parse_double(value, &cfg->energy_max_GeV);
    if (strcmp(key, "energy_step_GeV") == 0) return parse_double(value, &cfg->energy_step_GeV);
    if (strcmp(key, "energy_logspace") == 0) return parse_int(value, &cfg->energy_logspace);
    if (strcmp(key, "energy_points") == 0)   return parse_int(value, &cfg->energy_points);

    if (strcmp(key, "theta12_deg") == 0) return parse_double(value, &cfg->theta12_deg);
    if (strcmp(key, "theta13_deg") == 0) return parse_double(value, &cfg->theta13_deg);
    if (strcmp(key, "theta23_deg") == 0) return parse_double(value, &cfg->theta23_deg);
    if (strcmp(key, "delta_cp_deg") == 0) return parse_double(value, &cfg->delta_cp_deg);

    if (strcmp(key, "n_sterile") == 0) return parse_int(value, &cfg->n_sterile);
    if (strcmp(key, "theta1s_deg") == 0) return parse_active_sterile_list(cfg, 0, value, 0);
    if (strcmp(key, "theta2s_deg") == 0) return parse_active_sterile_list(cfg, 1, value, 0);
    if (strcmp(key, "theta3s_deg") == 0) return parse_active_sterile_list(cfg, 2, value, 0);
    if (strcmp(key, "delta1s_deg") == 0) return parse_active_sterile_list(cfg, 0, value, 1);
    if (strcmp(key, "delta2s_deg") == 0) return parse_active_sterile_list(cfg, 1, value, 1);
    if (strcmp(key, "delta3s_deg") == 0) return parse_active_sterile_list(cfg, 2, value, 1);
    if (strcmp(key, "theta_ss_deg") == 0) return parse_sterile_pair_list(cfg, value, 0);
    if (strcmp(key, "delta_ss_deg") == 0) return parse_sterile_pair_list(cfg, value, 1);

    if (strcmp(key, "theta14_deg") == 0) return parse_legacy_active_sterile(cfg, value, 0, 0, 0);
    if (strcmp(key, "theta24_deg") == 0) return parse_legacy_active_sterile(cfg, value, 1, 0, 0);
    if (strcmp(key, "theta34_deg") == 0) return parse_legacy_active_sterile(cfg, value, 2, 0, 0);
    if (strcmp(key, "delta14_deg") == 0) return parse_legacy_active_sterile(cfg, value, 0, 0, 1);
    if (strcmp(key, "delta24_deg") == 0) return parse_legacy_active_sterile(cfg, value, 1, 0, 1);
    if (strcmp(key, "delta34_deg") == 0) return parse_legacy_active_sterile(cfg, value, 2, 0, 1);
    if (strcmp(key, "theta15_deg") == 0) return parse_legacy_active_sterile(cfg, value, 0, 1, 0);
    if (strcmp(key, "theta25_deg") == 0) return parse_legacy_active_sterile(cfg, value, 1, 1, 0);
    if (strcmp(key, "theta35_deg") == 0) return parse_legacy_active_sterile(cfg, value, 2, 1, 0);
    if (strcmp(key, "delta15_deg") == 0) return parse_legacy_active_sterile(cfg, value, 0, 1, 1);
    if (strcmp(key, "delta25_deg") == 0) return parse_legacy_active_sterile(cfg, value, 1, 1, 1);
    if (strcmp(key, "delta35_deg") == 0) return parse_legacy_active_sterile(cfg, value, 2, 1, 1);
    if (strcmp(key, "theta45_deg") == 0) return parse_legacy_sterile_sterile(cfg, value, 0, 1, 0);

    if (strcmp(key, "dm21_eV2") == 0) return parse_double(value, &cfg->dm21_eV2);
    if (strcmp(key, "dm31_eV2") == 0) return parse_double(value, &cfg->dm31_eV2);

    if (strcmp(key, "gaussian_filter_enabled") == 0) return parse_int(value, &cfg->gaussian_filter_enabled);
    if (strcmp(key, "sigmaE_over_E") == 0) return parse_double(value, &cfg->sigmaE_over_E);
    if (strcmp(key, "matter_effects_enabled") == 0) return parse_int(value, &cfg->matter_effects_enabled);
    if (strcmp(key, "matter_density_g_cm3") == 0) return parse_double(value, &cfg->matter_density_g_cm3);
    if (strcmp(key, "matter_electron_fraction") == 0) return parse_double(value, &cfg->matter_electron_fraction);
    if (strcmp(key, "matter_neutron_fraction") == 0) return parse_double(value, &cfg->matter_neutron_fraction);
    if (strcmp(key, "matter_include_neutral_current_sterile") == 0) return parse_int(value, &cfg->matter_include_neutral_current_sterile);
    if (strcmp(key, "matter_evolution_steps") == 0) return parse_int(value, &cfg->matter_evolution_steps);
    if (strcmp(key, "matter_a_cc_coeff_eV2_per_GeV_per_gcm3") == 0) return parse_double(value, &cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3);

    if (strcmp(key, "dm41_range_enabled") == 0) return parse_int(value, &cfg->dm41_range_enabled);
    if (strcmp(key, "dm41_range_min_eV2") == 0) return parse_double(value, &cfg->dm41_range_min_eV2);
    if (strcmp(key, "dm41_range_max_eV2") == 0) return parse_double(value, &cfg->dm41_range_max_eV2);
    if (strcmp(key, "dm41_range_step_eV2") == 0) return parse_double(value, &cfg->dm41_range_step_eV2);
    if (strcmp(key, "dm41_range_logspace") == 0) return parse_int(value, &cfg->dm41_range_logspace);
    if (strcmp(key, "dm41_range_points") == 0) return parse_int(value, &cfg->dm41_range_points);

    if (strcmp(key, "dm41_values_eV2") == 0) return parse_dm41_list(value, cfg);
    if (strcmp(key, "dm54_range_enabled") == 0) return parse_int(value, &cfg->dm54_range_enabled);
    if (strcmp(key, "dm54_range_min_eV2") == 0) return parse_double(value, &cfg->dm54_range_min_eV2);
    if (strcmp(key, "dm54_range_max_eV2") == 0) return parse_double(value, &cfg->dm54_range_max_eV2);
    if (strcmp(key, "dm54_range_step_eV2") == 0) return parse_double(value, &cfg->dm54_range_step_eV2);
    if (strcmp(key, "dm54_range_logspace") == 0) return parse_int(value, &cfg->dm54_range_logspace);
    if (strcmp(key, "dm54_range_points") == 0) return parse_int(value, &cfg->dm54_range_points);
    if (strcmp(key, "dm54_values_eV2") == 0) return parse_dm54_list(value, cfg);
    if (strcmp(key, "dm41_3p2_eV2") == 0) return parse_dm41_3p2_value(value, cfg);
    if (strcmp(key, "dm41_heatmap_3p2_values_eV2") == 0) return parse_dm41_heatmap_3p2_list(value, cfg);

    if (strcmp(key, "inverse_pmns_abs_min_3x3") == 0) return parse_inverse_pmns_abs_min_3x3(value, cfg);
    if (strcmp(key, "inverse_pmns_abs_max_3x3") == 0) return parse_inverse_pmns_abs_max_3x3(value, cfg);
    if (strcmp(key, "inverse_eta_abs_max_3x3") == 0) return parse_inverse_eta_abs_max_3x3(value, cfg);
    if (strcmp(key, "inverse_eta_abs_max_nonunitarity_3x3") == 0) return parse_inverse_eta_abs_max_nonunitarity_3x3(value, cfg);
    if (strcmp(key, "inverse_eta_abs_max_light_highdm_3x3") == 0) return parse_inverse_eta_abs_max_light_highdm_3x3(value, cfg);
    if (strcmp(key, "inverse_eta_abs_max_light_lowdm_3x3") == 0) return parse_inverse_eta_abs_max_light_lowdm_3x3(value, cfg);
    if (strcmp(key, "inverse_eta_dm41_low_min_eV2") == 0) return parse_double(value, &cfg->inverse_eta_dm41_low_min_eV2);
    if (strcmp(key, "inverse_eta_dm41_low_max_eV2") == 0) return parse_double(value, &cfg->inverse_eta_dm41_low_max_eV2);
    if (strcmp(key, "inverse_eta_dm41_high_min_eV2") == 0) return parse_double(value, &cfg->inverse_eta_dm41_high_min_eV2);
    if (strcmp(key, "inverse_br_muegamma_max") == 0) return parse_double(value, &cfg->inverse_br_muegamma_max);
    if (strcmp(key, "dune_theory_index_csv") == 0 ||
        strcmp(key, "theory.index_csv") == 0 ||
        strcmp(key, "point_index_csv") == 0 ||
        strcmp(key, "index_csv") == 0) {
        strncpy(cfg->dune_theory_index_csv, value, sizeof(cfg->dune_theory_index_csv) - 1);
        cfg->dune_theory_index_csv[sizeof(cfg->dune_theory_index_csv) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "dune_theory_model") == 0 ||
        strcmp(key, "theory.model") == 0 ||
        strcmp(key, "model") == 0) {
        strncpy(cfg->dune_theory_model, value, sizeof(cfg->dune_theory_model) - 1);
        cfg->dune_theory_model[sizeof(cfg->dune_theory_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "dune_point_source") == 0 ||
        strcmp(key, "theory.point_source") == 0 ||
        strcmp(key, "point_source") == 0) {
        strncpy(cfg->dune_point_source, value, sizeof(cfg->dune_point_source) - 1);
        cfg->dune_point_source[sizeof(cfg->dune_point_source) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "dune_point_id") == 0 ||
        strcmp(key, "theory.point_id") == 0 ||
        strcmp(key, "point_id") == 0) {
        return parse_int(value, &cfg->dune_point_id);
    }
    if (strcmp(key, "beam.mode") == 0 || strcmp(key, "dune_beam_mode") == 0 || strcmp(key, "beam_mode") == 0) {
        strncpy(cfg->dune_beam_mode, value, sizeof(cfg->dune_beam_mode) - 1);
        cfg->dune_beam_mode[sizeof(cfg->dune_beam_mode) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.flux_model") == 0 || strcmp(key, "dune_flux_model") == 0 || strcmp(key, "flux_model") == 0) {
        strncpy(cfg->dune_flux_model, value, sizeof(cfg->dune_flux_model) - 1);
        cfg->dune_flux_model[sizeof(cfg->dune_flux_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.flux_format") == 0 || strcmp(key, "dune_flux_format") == 0 || strcmp(key, "flux_format") == 0) {
        strncpy(cfg->dune_flux_format, value, sizeof(cfg->dune_flux_format) - 1);
        cfg->dune_flux_format[sizeof(cfg->dune_flux_format) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.flux_fhc_file") == 0 || strcmp(key, "dune_flux_fhc_file") == 0 || strcmp(key, "flux_fhc_file") == 0) {
        strncpy(cfg->dune_flux_fhc_file, value, sizeof(cfg->dune_flux_fhc_file) - 1);
        cfg->dune_flux_fhc_file[sizeof(cfg->dune_flux_fhc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.flux_rhc_file") == 0 || strcmp(key, "dune_flux_rhc_file") == 0 || strcmp(key, "flux_rhc_file") == 0) {
        strncpy(cfg->dune_flux_rhc_file, value, sizeof(cfg->dune_flux_rhc_file) - 1);
        cfg->dune_flux_rhc_file[sizeof(cfg->dune_flux_rhc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.baseline_model") == 0 || strcmp(key, "baseline_model") == 0) {
        strncpy(cfg->dune_baseline_model, value, sizeof(cfg->dune_baseline_model) - 1);
        cfg->dune_baseline_model[sizeof(cfg->dune_baseline_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.source_model") == 0 || strcmp(key, "source_model") == 0) {
        strncpy(cfg->dune_source_model, value, sizeof(cfg->dune_source_model) - 1);
        cfg->dune_source_model[sizeof(cfg->dune_source_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.dk2nu_flux_z_fhc_file") == 0 ||
        strcmp(key, "beam.source_profile_z_fhc_file") == 0 ||
        strcmp(key, "dk2nu_flux_z_fhc_file") == 0 ||
        strcmp(key, "source_profile_z_fhc_file") == 0) {
        strncpy(cfg->dune_dk2nu_flux_z_fhc_file, value, sizeof(cfg->dune_dk2nu_flux_z_fhc_file) - 1);
        cfg->dune_dk2nu_flux_z_fhc_file[sizeof(cfg->dune_dk2nu_flux_z_fhc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.dk2nu_flux_z_rhc_file") == 0 ||
        strcmp(key, "beam.source_profile_z_rhc_file") == 0 ||
        strcmp(key, "dk2nu_flux_z_rhc_file") == 0 ||
        strcmp(key, "source_profile_z_rhc_file") == 0) {
        strncpy(cfg->dune_dk2nu_flux_z_rhc_file, value, sizeof(cfg->dune_dk2nu_flux_z_rhc_file) - 1);
        cfg->dune_dk2nu_flux_z_rhc_file[sizeof(cfg->dune_dk2nu_flux_z_rhc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "beam.detector_distance_m") == 0 || strcmp(key, "detector_distance_m") == 0) return parse_double(value, &cfg->dune_detector_distance_m);
    if (strcmp(key, "beam.source_z_start_m") == 0 || strcmp(key, "source_z_start_m") == 0) return parse_double(value, &cfg->dune_source_z_start_m);
    if (strcmp(key, "beam.decay_pipe_length_m") == 0 || strcmp(key, "decay_pipe_length_m") == 0) return parse_double(value, &cfg->dune_decay_pipe_length_m);
    if (strcmp(key, "beam.source_z_bins") == 0 || strcmp(key, "source_z_bins") == 0) return parse_int(value, &cfg->dune_source_z_bins);
    if (strcmp(key, "beam.source_debug") == 0 || strcmp(key, "source_debug") == 0) return parse_int(value, &cfg->dune_source_debug);
    if (strcmp(key, "oscillation.engine") == 0 || strcmp(key, "osc_engine") == 0) {
        strncpy(cfg->dune_osc_engine, value, sizeof(cfg->dune_osc_engine) - 1);
        cfg->dune_osc_engine[sizeof(cfg->dune_osc_engine) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "oscillation.matter_enabled") == 0 || strcmp(key, "matter_enabled") == 0) return parse_int(value, &cfg->dune_matter_enabled);
    if (strcmp(key, "interactions.xsec_model") == 0 || strcmp(key, "xsec_model") == 0) {
        strncpy(cfg->dune_xsec_model, value, sizeof(cfg->dune_xsec_model) - 1);
        cfg->dune_xsec_model[sizeof(cfg->dune_xsec_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "interactions.xsec_format") == 0 || strcmp(key, "xsec_format") == 0) {
        strncpy(cfg->dune_xsec_format, value, sizeof(cfg->dune_xsec_format) - 1);
        cfg->dune_xsec_format[sizeof(cfg->dune_xsec_format) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "interactions.xsec_cc_file") == 0 || strcmp(key, "xsec_cc_file") == 0) {
        strncpy(cfg->dune_xsec_cc_file, value, sizeof(cfg->dune_xsec_cc_file) - 1);
        cfg->dune_xsec_cc_file[sizeof(cfg->dune_xsec_cc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "interactions.xsec_nc_file") == 0 || strcmp(key, "xsec_nc_file") == 0) {
        strncpy(cfg->dune_xsec_nc_file, value, sizeof(cfg->dune_xsec_nc_file) - 1);
        cfg->dune_xsec_nc_file[sizeof(cfg->dune_xsec_nc_file) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "detector.detectors") == 0 || strcmp(key, "detectors") == 0) {
        strncpy(cfg->dune_detectors, value, sizeof(cfg->dune_detectors) - 1);
        cfg->dune_detectors[sizeof(cfg->dune_detectors) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "detector.ndlar.response_model") == 0 || strcmp(key, "response_model") == 0) {
        strncpy(cfg->dune_ndlar_response_model, value, sizeof(cfg->dune_ndlar_response_model) - 1);
        cfg->dune_ndlar_response_model[sizeof(cfg->dune_ndlar_response_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "detector.ndlar.migration_model") == 0 || strcmp(key, "migration_model") == 0) {
        strncpy(cfg->dune_ndlar_migration_model, value, sizeof(cfg->dune_ndlar_migration_model) - 1);
        cfg->dune_ndlar_migration_model[sizeof(cfg->dune_ndlar_migration_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "detector.ndlar.category_model") == 0 || strcmp(key, "category_model") == 0) {
        strncpy(cfg->dune_ndlar_category_model, value, sizeof(cfg->dune_ndlar_category_model) - 1);
        cfg->dune_ndlar_category_model[sizeof(cfg->dune_ndlar_category_model) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "samples.enabled") == 0 || strcmp(key, "samples_enabled") == 0) {
        strncpy(cfg->dune_samples_enabled, value, sizeof(cfg->dune_samples_enabled) - 1);
        cfg->dune_samples_enabled[sizeof(cfg->dune_samples_enabled) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "samples.axis") == 0 || strcmp(key, "samples_axis") == 0) {
        strncpy(cfg->dune_samples_axis, value, sizeof(cfg->dune_samples_axis) - 1);
        cfg->dune_samples_axis[sizeof(cfg->dune_samples_axis) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "samples.Erec_min_GeV") == 0 || strcmp(key, "Erec_min_GeV") == 0) return parse_double(value, &cfg->dune_Erec_min_GeV);
    if (strcmp(key, "samples.Erec_max_GeV") == 0 || strcmp(key, "Erec_max_GeV") == 0) return parse_double(value, &cfg->dune_Erec_max_GeV);
    if (strcmp(key, "samples.Erec_bins") == 0 || strcmp(key, "Erec_bins") == 0) return parse_int(value, &cfg->dune_Erec_bins);
    if (strcmp(key, "output.spectrum_pred_csv") == 0 || strcmp(key, "spectrum_pred_csv") == 0) {
        strncpy(cfg->dune_spectrum_pred_csv, value, sizeof(cfg->dune_spectrum_pred_csv) - 1);
        cfg->dune_spectrum_pred_csv[sizeof(cfg->dune_spectrum_pred_csv) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "output.spectrum_null_csv") == 0 || strcmp(key, "spectrum_null_csv") == 0) {
        strncpy(cfg->dune_spectrum_null_csv, value, sizeof(cfg->dune_spectrum_null_csv) - 1);
        cfg->dune_spectrum_null_csv[sizeof(cfg->dune_spectrum_null_csv) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "output.residuals_csv") == 0 || strcmp(key, "residuals_csv") == 0) {
        strncpy(cfg->dune_residuals_csv, value, sizeof(cfg->dune_residuals_csv) - 1);
        cfg->dune_residuals_csv[sizeof(cfg->dune_residuals_csv) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "output.point_observables_csv") == 0 || strcmp(key, "point_observables_csv") == 0) {
        strncpy(cfg->dune_point_observables_csv, value, sizeof(cfg->dune_point_observables_csv) - 1);
        cfg->dune_point_observables_csv[sizeof(cfg->dune_point_observables_csv) - 1] = '\0';
        return 0;
    }
    if (strcmp(key, "inverse_random_samples") == 0) return parse_int(value, &cfg->inverse_random_samples);
    if (strcmp(key, "inverse_random_seed") == 0) return parse_int(value, &cfg->inverse_random_seed);
    if (strcmp(key, "inverse_random_mu_min_eV") == 0) return parse_double(value, &cfg->inverse_random_mu_min_eV);
    if (strcmp(key, "inverse_random_mu_max_eV") == 0) return parse_double(value, &cfg->inverse_random_mu_max_eV);
    if (strcmp(key, "inverse_random_MR_min_eV") == 0) return parse_double(value, &cfg->inverse_random_MR_min_eV);
    if (strcmp(key, "inverse_random_MR_max_eV") == 0) return parse_double(value, &cfg->inverse_random_MR_max_eV);
    if (strcmp(key, "inverse_random_mD_min_eV") == 0) return parse_double(value, &cfg->inverse_random_mD_min_eV);
    if (strcmp(key, "inverse_random_mD_max_eV") == 0) return parse_double(value, &cfg->inverse_random_mD_max_eV);

    if (strcmp(key, "inverse_construct_23_samples") == 0) return parse_int(value, &cfg->inverse_construct_23_samples);
    if (strcmp(key, "inverse_construct_23_seed") == 0) return parse_int(value, &cfg->inverse_construct_23_seed);
    if (strcmp(key, "inverse_construct_23_dm41_min_eV2") == 0) return parse_double(value, &cfg->inverse_construct_23_dm41_min_eV2);
    if (strcmp(key, "inverse_construct_23_dm41_max_eV2") == 0) return parse_double(value, &cfg->inverse_construct_23_dm41_max_eV2);
    if (strcmp(key, "inverse_construct_23_zeta_norm_min") == 0) return parse_double(value, &cfg->inverse_construct_23_zeta_norm_min);
    if (strcmp(key, "inverse_construct_23_zeta_norm_max") == 0) return parse_double(value, &cfg->inverse_construct_23_zeta_norm_max);
    if (strcmp(key, "inverse_construct_23_zeta_direction_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_zeta_direction_min_deg);
    if (strcmp(key, "inverse_construct_23_zeta_direction_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_zeta_direction_max_deg);
    if (strcmp(key, "inverse_construct_23_alpha21_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_alpha21_min_deg);
    if (strcmp(key, "inverse_construct_23_alpha21_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_alpha21_max_deg);
    if (strcmp(key, "inverse_construct_23_alpha31_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_alpha31_min_deg);
    if (strcmp(key, "inverse_construct_23_alpha31_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_alpha31_max_deg);
    if (strcmp(key, "inverse_construct_23_f11_min") == 0) return parse_double(value, &cfg->inverse_construct_23_f11_min);
    if (strcmp(key, "inverse_construct_23_f11_max") == 0) return parse_double(value, &cfg->inverse_construct_23_f11_max);
    if (strcmp(key, "inverse_construct_23_f11_phase_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f11_phase_min_deg);
    if (strcmp(key, "inverse_construct_23_f11_phase_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f11_phase_max_deg);
    if (strcmp(key, "inverse_construct_23_f12_min") == 0) return parse_double(value, &cfg->inverse_construct_23_f12_min);
    if (strcmp(key, "inverse_construct_23_f12_max") == 0) return parse_double(value, &cfg->inverse_construct_23_f12_max);
    if (strcmp(key, "inverse_construct_23_f12_phase_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f12_phase_min_deg);
    if (strcmp(key, "inverse_construct_23_f12_phase_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f12_phase_max_deg);
    if (strcmp(key, "inverse_construct_23_f21_min") == 0) return parse_double(value, &cfg->inverse_construct_23_f21_min);
    if (strcmp(key, "inverse_construct_23_f21_max") == 0) return parse_double(value, &cfg->inverse_construct_23_f21_max);
    if (strcmp(key, "inverse_construct_23_f21_phase_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f21_phase_min_deg);
    if (strcmp(key, "inverse_construct_23_f21_phase_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f21_phase_max_deg);
    if (strcmp(key, "inverse_construct_23_f22_min") == 0) return parse_double(value, &cfg->inverse_construct_23_f22_min);
    if (strcmp(key, "inverse_construct_23_f22_max") == 0) return parse_double(value, &cfg->inverse_construct_23_f22_max);
    if (strcmp(key, "inverse_construct_23_f22_phase_min_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f22_phase_min_deg);
    if (strcmp(key, "inverse_construct_23_f22_phase_max_deg") == 0) return parse_double(value, &cfg->inverse_construct_23_f22_phase_max_deg);
    if (strcmp(key, "inverse_construct_23_f_det_min_abs") == 0) return parse_double(value, &cfg->inverse_construct_23_f_det_min_abs);
    if (strcmp(key, "inverse_construct_23_f_det_max_abs") == 0) return parse_double(value, &cfg->inverse_construct_23_f_det_max_abs);
    if (strcmp(key, "inverse_construct_23_f_sigma_min_min") == 0) return parse_double(value, &cfg->inverse_construct_23_f_sigma_min_min);
    if (strcmp(key, "inverse_construct_23_kappa_f_max") == 0) return parse_double(value, &cfg->inverse_construct_23_kappa_f_max);
    if (strcmp(key, "inverse_construct_23_M1_min_GeV") == 0) return parse_double(value, &cfg->inverse_construct_23_M1_min_GeV);
    if (strcmp(key, "inverse_construct_23_M1_max_GeV") == 0) return parse_double(value, &cfg->inverse_construct_23_M1_max_GeV);
    if (strcmp(key, "inverse_construct_23_M2_min_GeV") == 0) return parse_double(value, &cfg->inverse_construct_23_M2_min_GeV);
    if (strcmp(key, "inverse_construct_23_M2_max_GeV") == 0) return parse_double(value, &cfg->inverse_construct_23_M2_max_GeV);


    if (strcmp(key, "inverse_nufit_theta12_deg") == 0) return parse_double(value, &cfg->inverse_nufit_theta12_deg);
    if (strcmp(key, "inverse_nufit_theta23_deg") == 0) return parse_double(value, &cfg->inverse_nufit_theta23_deg);
    if (strcmp(key, "inverse_nufit_theta13_deg") == 0) return parse_double(value, &cfg->inverse_nufit_theta13_deg);
    if (strcmp(key, "inverse_nufit_deltacp_deg") == 0) return parse_double(value, &cfg->inverse_nufit_deltacp_deg);

    if (strcmp(key, "inverse_scan_mu00_min_eV") == 0) return parse_double(value, &cfg->inverse_scan_mu00_min_eV);
    if (strcmp(key, "inverse_scan_mu00_max_eV") == 0) return parse_double(value, &cfg->inverse_scan_mu00_max_eV);
    if (strcmp(key, "inverse_scan_mu00_step_eV") == 0) return parse_double(value, &cfg->inverse_scan_mu00_step_eV);
    if (strcmp(key, "inverse_scan_M1_min_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M1_min_GeV);
    if (strcmp(key, "inverse_scan_M1_max_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M1_max_GeV);
    if (strcmp(key, "inverse_scan_M1_step_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M1_step_GeV);
    if (strcmp(key, "inverse_scan_M2_min_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M2_min_GeV);
    if (strcmp(key, "inverse_scan_M2_max_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M2_max_GeV);
    if (strcmp(key, "inverse_scan_M2_step_GeV") == 0) return parse_double(value, &cfg->inverse_scan_M2_step_GeV);
    if (strcmp(key, "inverse_scan_z_real_min") == 0) return parse_double(value, &cfg->inverse_scan_z_real_min);
    if (strcmp(key, "inverse_scan_z_real_max") == 0) return parse_double(value, &cfg->inverse_scan_z_real_max);
    if (strcmp(key, "inverse_scan_z_real_step") == 0) return parse_double(value, &cfg->inverse_scan_z_real_step);
    if (strcmp(key, "inverse_scan_z_imag_min") == 0) return parse_double(value, &cfg->inverse_scan_z_imag_min);
    if (strcmp(key, "inverse_scan_z_imag_max") == 0) return parse_double(value, &cfg->inverse_scan_z_imag_max);
    if (strcmp(key, "inverse_scan_z_imag_step") == 0) return parse_double(value, &cfg->inverse_scan_z_imag_step);
    if (strcmp(key, "inverse_scan_muH11_min_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH11_min_eV);
    if (strcmp(key, "inverse_scan_muH11_max_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH11_max_eV);
    if (strcmp(key, "inverse_scan_muH11_step_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH11_step_eV);
    if (strcmp(key, "inverse_scan_muH22_min_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH22_min_eV);
    if (strcmp(key, "inverse_scan_muH22_max_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH22_max_eV);
    if (strcmp(key, "inverse_scan_muH22_step_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH22_step_eV);
    if (strcmp(key, "inverse_scan_muH01_min_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH01_min_eV);
    if (strcmp(key, "inverse_scan_muH01_max_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH01_max_eV);
    if (strcmp(key, "inverse_scan_muH01_step_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH01_step_eV);
    if (strcmp(key, "inverse_scan_muH02_min_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH02_min_eV);
    if (strcmp(key, "inverse_scan_muH02_max_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH02_max_eV);
    if (strcmp(key, "inverse_scan_muH02_step_eV") == 0) return parse_double(value, &cfg->inverse_scan_muH02_step_eV);

    if (strcmp(key, "output_csv_path") == 0) {
        strncpy(cfg->output_csv_path, value, sizeof(cfg->output_csv_path) - 1);
        cfg->output_csv_path[sizeof(cfg->output_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_csv_3p2_path") == 0) {
        strncpy(cfg->output_csv_3p2_path, value, sizeof(cfg->output_csv_3p2_path) - 1);
        cfg->output_csv_3p2_path[sizeof(cfg->output_csv_3p2_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_heatmap_csv_path") == 0) {
        strncpy(cfg->output_heatmap_csv_path, value, sizeof(cfg->output_heatmap_csv_path) - 1);
        cfg->output_heatmap_csv_path[sizeof(cfg->output_heatmap_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_heatmap_3p2_csv_path") == 0) {
        strncpy(cfg->output_heatmap_3p2_csv_path, value, sizeof(cfg->output_heatmap_3p2_csv_path) - 1);
        cfg->output_heatmap_3p2_csv_path[sizeof(cfg->output_heatmap_3p2_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_heatmap_pmumu_3p2_csv_path") == 0) {
        strncpy(cfg->output_heatmap_pmumu_3p2_csv_path, value, sizeof(cfg->output_heatmap_pmumu_3p2_csv_path) - 1);
        cfg->output_heatmap_pmumu_3p2_csv_path[sizeof(cfg->output_heatmap_pmumu_3p2_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_cp_heatmap_csv_path") == 0) {
        strncpy(cfg->output_cp_heatmap_csv_path, value, sizeof(cfg->output_cp_heatmap_csv_path) - 1);
        cfg->output_cp_heatmap_csv_path[sizeof(cfg->output_cp_heatmap_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_inverse_pmns_filter_csv_path") == 0) {
        strncpy(cfg->output_inverse_pmns_filter_csv_path, value, sizeof(cfg->output_inverse_pmns_filter_csv_path) - 1);
        cfg->output_inverse_pmns_filter_csv_path[sizeof(cfg->output_inverse_pmns_filter_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "output_inverse_construct_23_csv_path") == 0) {
        strncpy(cfg->output_inverse_construct_23_csv_path, value, sizeof(cfg->output_inverse_construct_23_csv_path) - 1);
        cfg->output_inverse_construct_23_csv_path[sizeof(cfg->output_inverse_construct_23_csv_path) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "inverse_kept_points_dir") == 0) {
        strncpy(cfg->inverse_kept_points_dir, value, sizeof(cfg->inverse_kept_points_dir) - 1);
        cfg->inverse_kept_points_dir[sizeof(cfg->inverse_kept_points_dir) - 1] = '\0';
        return 0;
    }

    if (strcmp(key, "inverse_clear_kept_points_dir") == 0) {
        return parse_int(value, &cfg->inverse_clear_kept_points_dir);
    }

    if (strcmp(key, "delta41_values_deg") == 0) {
        return parse_double_list(value, cfg->delta41_values_deg, 3600, &cfg->delta41_count);
    }
    if (strcmp(key, "delta41_range_enabled") == 0) return parse_int(value, &cfg->delta41_range_enabled);
    if (strcmp(key, "delta41_range_min_deg") == 0) return parse_double(value, &cfg->delta41_range_min_deg);
    if (strcmp(key, "delta41_range_max_deg") == 0) return parse_double(value, &cfg->delta41_range_max_deg);
    if (strcmp(key, "delta41_range_points") == 0) return parse_int(value, &cfg->delta41_range_points);

    if (strcmp(key, "dist_min_km") == 0) return parse_double(value, &cfg->dist_min_km);
    if (strcmp(key, "dist_max_km") == 0) return parse_double(value, &cfg->dist_max_km);
    if (strcmp(key, "dist_step_km") == 0) return parse_double(value, &cfg->dist_step_km);
    if (strcmp(key, "dist_fixed_energy_GeV") == 0) return parse_double(value, &cfg->dist_fixed_energy_GeV);

    if (strcmp(key, "output_dist_csv_path") == 0) {
        strncpy(cfg->output_dist_csv_path, value, sizeof(cfg->output_dist_csv_path) - 1);
        cfg->output_dist_csv_path[sizeof(cfg->output_dist_csv_path) - 1] = '\0';
        return 0;
    }

    return 1;
}

static int finalize_config(SimulationConfig *cfg) {
    if (cfg->n_sterile <= 0) {
        cfg->n_sterile = (cfg->dm54_count > 0) ? 2 : 1;
    }

    if (cfg->n_sterile > MAX_STERILE_NEUTRINOS) {
        return 5;
    }

    if (cfg->operation == OPERATION_UNSET) {
        return 6;
    }

    if (cfg->matter_effects_enabled != 0) {
        cfg->matter_effects_enabled = 1;
    }
    if (cfg->matter_include_neutral_current_sterile != 0) {
        cfg->matter_include_neutral_current_sterile = 1;
    }

    if (cfg->matter_density_g_cm3 <= 0.0) {
        cfg->matter_density_g_cm3 = 2.848;
    }
    if (cfg->matter_electron_fraction <= 0.0) {
        cfg->matter_electron_fraction = 0.5;
    }
    if (cfg->matter_neutron_fraction <= 0.0) {
        cfg->matter_neutron_fraction = fmax(0.0, 1.0 - cfg->matter_electron_fraction);
    }
    if (cfg->matter_evolution_steps <= 0) {
        cfg->matter_evolution_steps = 600;
    }
    if (cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3 <= 0.0) {
        cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3 = 1.52e-4;
    }
    if (cfg->dune_source_model[0] == '\0') {
        strncpy(cfg->dune_source_model, "uniform", sizeof(cfg->dune_source_model) - 1);
    }

    if (cfg->matter_density_g_cm3 <= 0.0 ||
        cfg->matter_electron_fraction < 0.0 || cfg->matter_electron_fraction > 1.0 ||
        cfg->matter_neutron_fraction < 0.0 || cfg->matter_neutron_fraction > 1.0 ||
        cfg->matter_evolution_steps <= 0 ||
        cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3 <= 0.0) {
        return 4;
    }

    if ((cfg->operation == OPERATION_ENERGY_3P1 || cfg->operation == OPERATION_DISTANCE_3P1 || cfg->operation == OPERATION_HEATMAP_DELTA_PMUE) && cfg->dm41_range_enabled) {
        if (build_dm41_values_from_range(cfg) != 0) {
            return 7;
        }
    }

    if ((cfg->operation == OPERATION_ENERGY_3P2 || cfg->operation == OPERATION_HEATMAP_DELTA_PMUE_3P2 || cfg->operation == OPERATION_HEATMAP_DELTA_PMUMU_3P2) && cfg->dm54_range_enabled) {
        if (build_dm54_values_from_range(cfg) != 0) {
            return 8;
        }
    }

    if (cfg->operation == OPERATION_CP_HEATMAP_3P1 && cfg->delta41_range_enabled) {
        if (build_delta41_values_from_range(cfg) != 0) {
            return 11;
        }
    }

    if (cfg->operation == OPERATION_CP_HEATMAP_3P1 && cfg->baseline_count == 0 && cfg->baseline_km > 0.0) {
        cfg->baseline_values_km[0] = cfg->baseline_km;
        cfg->baseline_count = 1;
    }

    if (cfg->operation == OPERATION_ENERGY_3P1) {
        if (cfg->dm41_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_csv_path[0] == '\0' || cfg->n_sterile < 1) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_ENERGY_3P2) {
        if (!cfg->dm41_3p2_is_set || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_csv_3p2_path[0] == '\0' || cfg->n_sterile < 2) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_DISTANCE_3P1) {
        if (cfg->dm41_count <= 0 || cfg->dist_step_km <= 0.0 || cfg->output_dist_csv_path[0] == '\0' || cfg->n_sterile < 1) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_HEATMAP_DELTA_PMUE) {
        if (cfg->dm41_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_csv_path[0] == '\0' || cfg->n_sterile < 1) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_HEATMAP_DELTA_PMUE_3P2) {
        if (cfg->dm41_heatmap_3p2_count != 4 || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_3p2_csv_path[0] == '\0' || cfg->n_sterile < 2) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_HEATMAP_DELTA_PMUMU_3P2) {
        if (cfg->dm41_heatmap_3p2_count != 4 || cfg->dm54_count <= 0 || cfg->energy_step_GeV <= 0.0 || cfg->output_heatmap_pmumu_3p2_csv_path[0] == '\0' || cfg->n_sterile < 2) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_CP_HEATMAP_3P1) {
        const int use_log = cfg->energy_logspace && cfg->energy_points >= 2;
        if (cfg->delta41_count <= 0 || cfg->dm41_count <= 0 || cfg->output_cp_heatmap_csv_path[0] == '\0' || cfg->n_sterile < 1) {
            return 4;
        }
        if (!use_log && cfg->energy_step_GeV <= 0.0) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_INVERSE_PMNS_FILTER_3P1) {
        static const double default_eta_max_nonunitarity_3x3[3][3] = {
            {1.3e-3, 6.8e-4, 2.7e-3},
            {6.8e-4, 2.2e-4, 1.2e-3},
            {2.7e-3, 1.2e-3, 2.8e-3}
        };
        static const double default_eta_max_light_highdm_3x3[3][3] = {
            {2.4e-2, 1.25e-2, 3.45e-2},
            {1.25e-2, 2.2e-2, 6.0e-3},
            {3.45e-2, 6.0e-3, 1.0e-1}
        };
        static const double default_eta_max_light_lowdm_3x3[3][3] = {
            {1.0e-2, 8.5e-3, 2.25e-2},
            {8.5e-3, 1.4e-2, 2.65e-2},
            {2.25e-2, 2.65e-2, 1.0e-1}
        };

        if (cfg->inverse_random_samples <= 0) {
            cfg->inverse_random_samples = 10000;
        }

        if (cfg->inverse_random_mu_min_eV <= 0.0 && cfg->inverse_random_mu_max_eV <= 0.0) {
            cfg->inverse_random_mu_min_eV = 0.1;
            cfg->inverse_random_mu_max_eV = 10.0;
        }
        if (cfg->inverse_random_MR_min_eV <= 0.0 && cfg->inverse_random_MR_max_eV <= 0.0) {
            cfg->inverse_random_MR_min_eV = 1.0e7;
            cfg->inverse_random_MR_max_eV = 1.0e8;
        }
        if (cfg->inverse_random_mD_min_eV <= 0.0 && cfg->inverse_random_mD_max_eV <= 0.0) {
            cfg->inverse_random_mD_min_eV = 1.0e6;
            cfg->inverse_random_mD_max_eV = 1.0e7;
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (cfg->inverse_pmns_abs_max_3x3[i][j] <= 0.0 && cfg->inverse_pmns_abs_min_3x3[i][j] <= 0.0) {
                    cfg->inverse_pmns_abs_min_3x3[i][j] = 0.0;
                    cfg->inverse_pmns_abs_max_3x3[i][j] = 1.0;
                }

                if (cfg->inverse_pmns_abs_min_3x3[i][j] < 0.0 ||
                    cfg->inverse_pmns_abs_max_3x3[i][j] > 1.0 ||
                    cfg->inverse_pmns_abs_min_3x3[i][j] > cfg->inverse_pmns_abs_max_3x3[i][j]) {
                    return 4;
                }

                if (cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j] <= 0.0) {
                    cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j] = default_eta_max_nonunitarity_3x3[i][j];
                }
                if (cfg->inverse_eta_abs_max_light_highdm_3x3[i][j] <= 0.0) {
                    cfg->inverse_eta_abs_max_light_highdm_3x3[i][j] = default_eta_max_light_highdm_3x3[i][j];
                }
                if (cfg->inverse_eta_abs_max_light_lowdm_3x3[i][j] <= 0.0) {
                    cfg->inverse_eta_abs_max_light_lowdm_3x3[i][j] = default_eta_max_light_lowdm_3x3[i][j];
                }

                if (cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j] <= 0.0 || cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j] > 1.0 ||
                    cfg->inverse_eta_abs_max_light_highdm_3x3[i][j] <= 0.0 || cfg->inverse_eta_abs_max_light_highdm_3x3[i][j] > 1.0 ||
                    cfg->inverse_eta_abs_max_light_lowdm_3x3[i][j] <= 0.0 || cfg->inverse_eta_abs_max_light_lowdm_3x3[i][j] > 1.0) {
                    return 4;
                }

                cfg->inverse_eta_abs_max_3x3[i][j] = cfg->inverse_eta_abs_max_nonunitarity_3x3[i][j];
            }
        }

        if (cfg->inverse_eta_dm41_low_min_eV2 <= 0.0) {
            cfg->inverse_eta_dm41_low_min_eV2 = 0.1;
        }
        if (cfg->inverse_eta_dm41_low_max_eV2 <= 0.0) {
            cfg->inverse_eta_dm41_low_max_eV2 = 1.0;
        }
        if (cfg->inverse_eta_dm41_high_min_eV2 <= 0.0) {
            cfg->inverse_eta_dm41_high_min_eV2 = 100.0;
        }

        if (cfg->inverse_br_muegamma_max <= 0.0) {
            cfg->inverse_br_muegamma_max = 1.5e-13;
        }

        if (cfg->inverse_kept_points_dir[0] == '\0') {
            strncpy(cfg->inverse_kept_points_dir,
                    "data/inverse_seesaw/3p1/inverse_pmns_filter_kept_points_8x8",
                    sizeof(cfg->inverse_kept_points_dir) - 1);
        }

        if (cfg->inverse_clear_kept_points_dir != 0) {
            cfg->inverse_clear_kept_points_dir = 1;
        }

        if (cfg->inverse_eta_dm41_low_min_eV2 <= 0.0 ||
            cfg->inverse_eta_dm41_low_max_eV2 < cfg->inverse_eta_dm41_low_min_eV2 ||
            cfg->inverse_eta_dm41_high_min_eV2 < cfg->inverse_eta_dm41_low_max_eV2 ||
            cfg->inverse_br_muegamma_max <= 0.0 || cfg->inverse_br_muegamma_max >= 1.0) {
            return 4;
        }

        if (cfg->output_inverse_pmns_filter_csv_path[0] == '\0' ||
            cfg->inverse_random_samples <= 0 ||
            cfg->inverse_random_mu_min_eV <= 0.0 || cfg->inverse_random_mu_max_eV < cfg->inverse_random_mu_min_eV ||
            cfg->inverse_random_MR_min_eV <= 0.0 || cfg->inverse_random_MR_max_eV < cfg->inverse_random_MR_min_eV ||
            cfg->inverse_random_mD_min_eV <= 0.0 || cfg->inverse_random_mD_max_eV < cfg->inverse_random_mD_min_eV) {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_INVERSE_PMNS_FILTER_3P2) {
        if (cfg->inverse_random_samples <= 0) {
            cfg->inverse_random_samples = 10000;
        }

        if (cfg->inverse_random_mu_min_eV <= 0.0 && cfg->inverse_random_mu_max_eV <= 0.0) {
            cfg->inverse_random_mu_min_eV = 0.1;
            cfg->inverse_random_mu_max_eV = 10.0;
        }
        if (cfg->inverse_random_MR_min_eV <= 0.0 && cfg->inverse_random_MR_max_eV <= 0.0) {
            cfg->inverse_random_MR_min_eV = 1.0e7;
            cfg->inverse_random_MR_max_eV = 1.0e8;
        }
        if (cfg->inverse_random_mD_min_eV <= 0.0 && cfg->inverse_random_mD_max_eV <= 0.0) {
            cfg->inverse_random_mD_min_eV = 1.0e6;
            cfg->inverse_random_mD_max_eV = 1.0e7;
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (cfg->inverse_pmns_abs_max_3x3[i][j] <= 0.0 && cfg->inverse_pmns_abs_min_3x3[i][j] <= 0.0) {
                    cfg->inverse_pmns_abs_min_3x3[i][j] = 0.0;
                    cfg->inverse_pmns_abs_max_3x3[i][j] = 1.0;
                }

                if (cfg->inverse_pmns_abs_min_3x3[i][j] < 0.0 ||
                    cfg->inverse_pmns_abs_max_3x3[i][j] > 1.0 ||
                    cfg->inverse_pmns_abs_min_3x3[i][j] > cfg->inverse_pmns_abs_max_3x3[i][j]) {
                    return 4;
                }
            }
        }

        if (cfg->inverse_kept_points_dir[0] == '\0') {
            strncpy(cfg->inverse_kept_points_dir,
                    "data/inverse_seesaw/3p2/inverse_pmns_filter_kept_points_9x9",
                    sizeof(cfg->inverse_kept_points_dir) - 1);
        }

        if (cfg->inverse_clear_kept_points_dir != 0) {
            cfg->inverse_clear_kept_points_dir = 1;
        }

        if (cfg->output_inverse_pmns_filter_csv_path[0] == '\0') {
            strncpy(cfg->output_inverse_pmns_filter_csv_path,
                    "data/inverse_seesaw/3p2/inverse_pmns_filter_3p2.csv",
                    sizeof(cfg->output_inverse_pmns_filter_csv_path) - 1);
        }

        if (cfg->inverse_random_samples <= 0 ||
            cfg->inverse_random_mu_min_eV <= 0.0 || cfg->inverse_random_mu_max_eV < cfg->inverse_random_mu_min_eV ||
            cfg->inverse_random_MR_min_eV <= 0.0 || cfg->inverse_random_MR_max_eV < cfg->inverse_random_MR_min_eV ||
            cfg->inverse_random_mD_min_eV <= 0.0 || cfg->inverse_random_mD_max_eV < cfg->inverse_random_mD_min_eV ||
            cfg->output_inverse_pmns_filter_csv_path[0] == '\0') {
            return 4;
        }
    }

    if (cfg->operation == OPERATION_INVERSE_CONSTRUCT_23_3P1) {
        if (cfg->inverse_construct_23_samples <= 0) {
            cfg->inverse_construct_23_samples = 10000;
        }

        if (cfg->inverse_kept_points_dir[0] == '\0') {
            strncpy(cfg->inverse_kept_points_dir,
                    "data/inverse_seesaw/3p1/inverse_construct_23_kept_points",
                    sizeof(cfg->inverse_kept_points_dir) - 1);
        }

        if (cfg->inverse_clear_kept_points_dir != 0) {
            cfg->inverse_clear_kept_points_dir = 1;
        }

        if (cfg->inverse_construct_23_dm41_min_eV2 <= 0.0 && cfg->inverse_construct_23_dm41_max_eV2 <= 0.0) {
            cfg->inverse_construct_23_dm41_min_eV2 = 0.1;
            cfg->inverse_construct_23_dm41_max_eV2 = 20.0;
        }

        if (cfg->inverse_construct_23_zeta_norm_min == 0.0 && cfg->inverse_construct_23_zeta_norm_max == 0.0) {
            cfg->inverse_construct_23_zeta_norm_min = 0.0;
            cfg->inverse_construct_23_zeta_norm_max = 0.99;
        }

        if (cfg->inverse_construct_23_zeta_direction_min_deg == 0.0 && cfg->inverse_construct_23_zeta_direction_max_deg == 0.0) {
            cfg->inverse_construct_23_zeta_direction_min_deg = 0.0;
            cfg->inverse_construct_23_zeta_direction_max_deg = 360.0;
        }

        if (cfg->inverse_construct_23_alpha21_min_deg == 0.0 && cfg->inverse_construct_23_alpha21_max_deg == 0.0) {
            cfg->inverse_construct_23_alpha21_min_deg = 0.0;
            cfg->inverse_construct_23_alpha21_max_deg = 360.0;
        }

        if (cfg->inverse_construct_23_alpha31_min_deg == 0.0 && cfg->inverse_construct_23_alpha31_max_deg == 0.0) {
            cfg->inverse_construct_23_alpha31_min_deg = 0.0;
            cfg->inverse_construct_23_alpha31_max_deg = 360.0;
        }

        if (cfg->inverse_construct_23_f11_min == 0.0 && cfg->inverse_construct_23_f11_max == 0.0 &&
            cfg->inverse_construct_23_f12_min == 0.0 && cfg->inverse_construct_23_f12_max == 0.0 &&
            cfg->inverse_construct_23_f21_min == 0.0 && cfg->inverse_construct_23_f21_max == 0.0 &&
            cfg->inverse_construct_23_f22_min == 0.0 && cfg->inverse_construct_23_f22_max == 0.0) {
            cfg->inverse_construct_23_f11_min = 0.0;
            cfg->inverse_construct_23_f11_max = 0.3;
            cfg->inverse_construct_23_f12_min = 0.0;
            cfg->inverse_construct_23_f12_max = 0.3;
            cfg->inverse_construct_23_f21_min = 0.0;
            cfg->inverse_construct_23_f21_max = 0.3;
            cfg->inverse_construct_23_f22_min = 0.0;
            cfg->inverse_construct_23_f22_max = 0.3;
        }

        if (cfg->inverse_construct_23_f11_phase_min_deg == 0.0 && cfg->inverse_construct_23_f11_phase_max_deg == 0.0 &&
            cfg->inverse_construct_23_f12_phase_min_deg == 0.0 && cfg->inverse_construct_23_f12_phase_max_deg == 0.0 &&
            cfg->inverse_construct_23_f21_phase_min_deg == 0.0 && cfg->inverse_construct_23_f21_phase_max_deg == 0.0 &&
            cfg->inverse_construct_23_f22_phase_min_deg == 0.0 && cfg->inverse_construct_23_f22_phase_max_deg == 0.0) {
            cfg->inverse_construct_23_f11_phase_min_deg = 0.0;
            cfg->inverse_construct_23_f11_phase_max_deg = 360.0;
            cfg->inverse_construct_23_f12_phase_min_deg = 0.0;
            cfg->inverse_construct_23_f12_phase_max_deg = 360.0;
            cfg->inverse_construct_23_f21_phase_min_deg = 0.0;
            cfg->inverse_construct_23_f21_phase_max_deg = 360.0;
            cfg->inverse_construct_23_f22_phase_min_deg = 0.0;
            cfg->inverse_construct_23_f22_phase_max_deg = 360.0;
        }

        if (cfg->inverse_construct_23_f_det_min_abs <= 0.0) {
            cfg->inverse_construct_23_f_det_min_abs = 1e-6;
        }

        if (cfg->inverse_construct_23_f_det_max_abs <= 0.0) {
            cfg->inverse_construct_23_f_det_max_abs = 1e30;
        }

        if (cfg->inverse_construct_23_f_sigma_min_min <= 0.0) {
            cfg->inverse_construct_23_f_sigma_min_min = 1e-6;
        }

        if (cfg->inverse_construct_23_kappa_f_max <= 0.0) {
            cfg->inverse_construct_23_kappa_f_max = 1e30;
        }

        if (cfg->inverse_construct_23_M1_min_GeV <= 0.0 && cfg->inverse_construct_23_M1_max_GeV <= 0.0 &&
            cfg->inverse_construct_23_M2_min_GeV <= 0.0 && cfg->inverse_construct_23_M2_max_GeV <= 0.0) {
            cfg->inverse_construct_23_M1_min_GeV = 10.0;
            cfg->inverse_construct_23_M1_max_GeV = 1.0e3;
            cfg->inverse_construct_23_M2_min_GeV = 10.0;
            cfg->inverse_construct_23_M2_max_GeV = 1.0e3;
        }


        if (cfg->inverse_construct_23_samples <= 0 ||
            cfg->inverse_construct_23_dm41_min_eV2 <= 0.0 || cfg->inverse_construct_23_dm41_max_eV2 < cfg->inverse_construct_23_dm41_min_eV2 ||
            cfg->inverse_construct_23_zeta_norm_min < 0.0 || cfg->inverse_construct_23_zeta_norm_max < cfg->inverse_construct_23_zeta_norm_min ||
            cfg->inverse_construct_23_zeta_norm_max >= 1.0 ||
            cfg->inverse_construct_23_zeta_direction_max_deg < cfg->inverse_construct_23_zeta_direction_min_deg ||
            cfg->inverse_construct_23_alpha21_max_deg < cfg->inverse_construct_23_alpha21_min_deg ||
            cfg->inverse_construct_23_alpha31_max_deg < cfg->inverse_construct_23_alpha31_min_deg ||
            cfg->inverse_construct_23_f11_min < 0.0 || cfg->inverse_construct_23_f11_max < cfg->inverse_construct_23_f11_min ||
            cfg->inverse_construct_23_f12_min < 0.0 || cfg->inverse_construct_23_f12_max < cfg->inverse_construct_23_f12_min ||
            cfg->inverse_construct_23_f21_min < 0.0 || cfg->inverse_construct_23_f21_max < cfg->inverse_construct_23_f21_min ||
            cfg->inverse_construct_23_f22_min < 0.0 || cfg->inverse_construct_23_f22_max < cfg->inverse_construct_23_f22_min ||
            cfg->inverse_construct_23_f11_phase_max_deg < cfg->inverse_construct_23_f11_phase_min_deg ||
            cfg->inverse_construct_23_f12_phase_max_deg < cfg->inverse_construct_23_f12_phase_min_deg ||
            cfg->inverse_construct_23_f21_phase_max_deg < cfg->inverse_construct_23_f21_phase_min_deg ||
            cfg->inverse_construct_23_f22_phase_max_deg < cfg->inverse_construct_23_f22_phase_min_deg ||
            cfg->inverse_construct_23_f_det_min_abs <= 0.0 ||
            cfg->inverse_construct_23_f_det_max_abs < cfg->inverse_construct_23_f_det_min_abs ||
            cfg->inverse_construct_23_f_sigma_min_min <= 0.0 ||
            cfg->inverse_construct_23_kappa_f_max <= 0.0 ||
            cfg->inverse_construct_23_M1_min_GeV <= 0.0 || cfg->inverse_construct_23_M1_max_GeV < cfg->inverse_construct_23_M1_min_GeV ||
            cfg->inverse_construct_23_M2_min_GeV <= 0.0 || cfg->inverse_construct_23_M2_max_GeV < cfg->inverse_construct_23_M2_min_GeV) {
            return 4;
        }
    }

    return 0;
}

static int load_config_file_recursive(const char *file_path, SimulationConfig *cfg, int depth) {
    FILE *in = fopen(file_path, "r");
    if (!in) {
        return 1;
    }

    {
        char line[1024];
        char current_section[128] = "";
        int line_no = 0;
        while (fgets(line, sizeof(line), in) != NULL) {
            ++line_no;
            trim_in_place(line);

            if (line[0] == '\0' || line[0] == '#') {
                continue;
            }
            if (line[0] == '[') {
                char *end_section = strchr(line, ']');
                if (!end_section) {
                    fclose(in);
                    return 2;
                }
                *end_section = '\0';
                strncpy(current_section, line + 1, sizeof(current_section) - 1);
                current_section[sizeof(current_section) - 1] = '\0';
                trim_in_place(current_section);
                continue;
            }

            {
                char *eq = strchr(line, '=');
                if (!eq) {
                    fclose(in);
                    return 2;
                }

                *eq = '\0';
                {
                    char *key = line;
                    char *value = eq + 1;
                    trim_in_place(key);
                    trim_in_place(value);

                    if (strcmp(key, "include") == 0) {
                        char include_value[1024];
                        char include_path[1024];

                        strncpy(include_value, value, sizeof(include_value) - 1);
                        include_value[sizeof(include_value) - 1] = '\0';
                        trim_in_place(include_value);
                        strip_optional_quotes(include_value);

                        if (depth >= CONFIG_INCLUDE_MAX_DEPTH) {
                            fclose(in);
                            return 13;
                        }

                        build_include_path(file_path, include_value, include_path, sizeof(include_path));
                        if (include_path[0] == '\0') {
                            fclose(in);
                            return 13;
                        }

                        if (load_config_file_recursive(include_path, cfg, depth + 1) != 0) {
                            fprintf(stderr, "Erreur include ligne %d: %s\n", line_no, include_value);
                            fclose(in);
                            return 3;
                        }
                        continue;
                    }

                    int set_status = set_key_value(cfg, key, value);
                    if (set_status != 0 && current_section[0] != '\0') {
                        char section_key[256];
                        snprintf(section_key, sizeof(section_key), "%s.%s", current_section, key);
                        set_status = set_key_value(cfg, section_key, value);
                    }

                    if (set_status != 0) {
                        fprintf(stderr, "Cle inconnue ou valeur invalide (%s:%d): %s\n", file_path, line_no, key);
                        if (is_removed_legacy_key(key)) {
                            fprintf(stderr, "  La cle '%s' n'est plus supportee dans cette version.\n", key);
                        }
                        fprintf(stderr, "  Preset recommande: %s\n", recommended_preset_for_operation(cfg->operation));
                        fclose(in);
                        return 3;
                    }
                }
            }
        }
    }

    fclose(in);
    return 0;
}

int load_config(const char *file_path, SimulationConfig *cfg) {
    if (!cfg) {
        return 1;
    }

    memset(cfg, 0, sizeof(*cfg));
    {
        const int read_status = load_config_file_recursive(file_path, cfg, 0);
        if (read_status != 0) {
            return read_status;
        }
    }

    {
        const int final_status = finalize_config(cfg);
        if (final_status != 0) {
            return final_status;
        }
    }

    return 0;
}

const char *operation_to_string(SimulationOperation operation) {
    switch (operation) {
        case OPERATION_ENERGY_3P1:
            return "energy_3p1";
        case OPERATION_ENERGY_3P2:
            return "energy_3p2";
        case OPERATION_DISTANCE_3P1:
            return "distance_3p1";
        case OPERATION_HEATMAP_DELTA_PMUE:
            return "heatmap_delta_pmu_e";
        case OPERATION_HEATMAP_DELTA_PMUE_3P2:
            return "heatmap_delta_pmu_e_3p2";
        case OPERATION_HEATMAP_DELTA_PMUMU_3P2:
            return "heatmap_delta_pmu_mu_3p2";
        case OPERATION_CP_HEATMAP_3P1:
            return "cp_heatmap_3p1";
        case OPERATION_INVERSE_PMNS_FILTER_3P1:
            return "inverse_pmns_filter_3p1";
        case OPERATION_INVERSE_PMNS_FILTER_3P2:
            return "inverse_pmns_filter_3p2";
        case OPERATION_INVERSE_CONSTRUCT_23_3P1:
            return "inverse_construct_23_3p1";
        case OPERATION_DUNE_ND_PREDICT_SPECTRUM:
            return "dune_nd_predict_spectrum";
        case OPERATION_DUNE_FD_FIG4_VALIDATION:
            return "dune_fd_fig4_validation";
        case OPERATION_DUNE_ND_FIG4_SOURCE_LINE:
            return "dune_nd_fig4_source_line";
        default:
            return "unset";
    }
}

void print_config_summary(const SimulationConfig *cfg) {
    printf("Configuration chargee depuis config.txt\n");
    fflush(stdout);
    printf("  operation = %s\n", operation_to_string(cfg->operation));
    fflush(stdout);
    printf("  baseline_km = %.6f\n", cfg->baseline_km);
    fflush(stdout);
    printf("  Energie = [%.3f, %.3f] GeV, pas = %.4f GeV\n",
           cfg->energy_min_GeV,
           cfg->energy_max_GeV,
           cfg->energy_step_GeV);
    fflush(stdout);
        printf("  n_sterile = %d\n", cfg->n_sterile);
        fflush(stdout);
        printf("  Phases CP (deg): deltaCP=%.3f, delta14=%.3f, delta24=%.3f, delta34=%.3f\n",
           cfg->delta_cp_deg,
            cfg->delta_active_sterile_deg[0][0],
            cfg->delta_active_sterile_deg[1][0],
            cfg->delta_active_sterile_deg[2][0]);
    fflush(stdout);
        printf("  Filtre gaussien: actif=%d, sigmaE/E=%.4f\n",
            cfg->gaussian_filter_enabled,
            cfg->sigmaE_over_E);
    fflush(stdout);
    printf("  Effets matiere: actif=%d, rho=%.3f g/cm^3, Ye=%.3f, Yn=%.3f, NC_sterile=%d, steps=%d\n",
           cfg->matter_effects_enabled,
           cfg->matter_density_g_cm3,
           cfg->matter_electron_fraction,
           cfg->matter_neutron_fraction,
           cfg->matter_include_neutral_current_sterile,
           cfg->matter_evolution_steps);
    fflush(stdout);
    printf("  Scan dm41 (%d valeurs): ", cfg->dm41_count);
    fflush(stdout);
    for (int i = 0; i < cfg->dm41_count; ++i) {
        printf("%s%.6g", (i == 0 ? "" : ", "), cfg->dm41_values_eV2[i]);
    }
    printf("\n");
    fflush(stdout);

    if (cfg->dm54_count > 0) {
        printf("  Scan dm54 (%d valeurs): ", cfg->dm54_count);
        fflush(stdout);
        for (int i = 0; i < cfg->dm54_count; ++i) {
            printf("%s%.6g", (i == 0 ? "" : ", "), cfg->dm54_values_eV2[i]);
        }
        printf("\n");
        fflush(stdout);
    }

    if (cfg->dm41_3p2_is_set) {
        printf("  dm41 (3+2 fixe) = %.6g\n", cfg->dm41_3p2_eV2);
        fflush(stdout);
    }
}
