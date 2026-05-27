#include "config.h"
#include "scan.h"

#include <stdio.h>

int main(int argc, char **argv) {
    SimulationConfig cfg;
    const char *config_path = (argc >= 2) ? argv[1] : "config.txt";

    const int config_status = load_config(config_path, &cfg);
    if (config_status != 0) {
        fprintf(stderr, "Erreur de lecture %s (code=%d)\n", config_path, config_status);
        return 1;
    }

    int run_status = 0;
    switch (cfg.operation) {
        case OPERATION_ENERGY_3P1:
            run_status = run_scan_energy_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan 3+1 en energie (code=%d)\n", run_status);
                return 2;
            }
            break;
        case OPERATION_ENERGY_3P2:
            run_status = run_scan_energy_3p2(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan 3+2 en energie (code=%d)\n", run_status);
                return 3;
            }
            break;
        case OPERATION_DISTANCE_3P1:
            run_status = run_scan_distance_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan distance 3+1 (code=%d)\n", run_status);
                return 4;
            }
            break;
        case OPERATION_HEATMAP_DELTA_PMUE:
            run_status = run_scan_heatmap_delta_pmue(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur heatmap Delta Pmue (code=%d)\n", run_status);
                return 6;
            }
            break;
        case OPERATION_HEATMAP_DELTA_PMUE_3P2:
            run_status = run_scan_heatmap_delta_pmue_3p2(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur heatmap Delta Pmue 3+2 (code=%d)\n", run_status);
                return 7;
            }
            break;
        case OPERATION_HEATMAP_DELTA_PMUMU_3P2:
            run_status = run_scan_heatmap_delta_pmumu_3p2(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur heatmap Delta Pmumu 3+2 (code=%d)\n", run_status);
                return 11;
            }
            break;
        case OPERATION_CP_HEATMAP_3P1:
            run_status = run_scan_cp_heatmap_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur heatmap ACP 3+1 (code=%d)\n", run_status);
                return 10;
            }
            break;
        case OPERATION_INVERSE_SEESAW_3P1:
            run_status = run_scan_inverse_seesaw_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan inverse seesaw 3+1 (code=%d)\n", run_status);
                return 12;
            }
            break;
        case OPERATION_INVERSE_PMNS_FILTER_3P1:
            run_status = run_scan_inverse_pmns_filter_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan inverse PMNS filtre 3+1 (code=%d)\n", run_status);
                return 13;
            }
            break;
        case OPERATION_INVERSE_PMNS_FILTER_3P2:
            run_status = run_scan_inverse_pmns_filter_3p2(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur scan inverse PMNS filtre 3+2 (code=%d)\n", run_status);
                return 15;
            }
            break;
        case OPERATION_INVERSE_CONSTRUCT_23_3P1:
            run_status = run_scan_inverse_construct_23_3p1(&cfg);
            if (run_status != 0) {
                fprintf(stderr, "Erreur construction adaptee inverse (2,3) 3+1 (code=%d)\n", run_status);
                return 14;
            }
            break;
        default:
            fprintf(stderr, "Operation non supportee: %s\n", operation_to_string(cfg.operation));
            return 5;
    }

    return 0;
}
