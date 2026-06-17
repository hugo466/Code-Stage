#ifndef SCAN_H
#define SCAN_H

#include "config.h"

int run_scan_energy_3p1(const SimulationConfig *cfg);
int run_scan_energy_3p2(const SimulationConfig *cfg);
int run_scan_distance_3p1(const SimulationConfig *cfg);
int run_scan_heatmap_delta_pmue(const SimulationConfig *cfg);
int run_scan_heatmap_delta_pmue_3p2(const SimulationConfig *cfg);
int run_scan_heatmap_delta_pmumu_3p2(const SimulationConfig *cfg);
int run_scan_cp_heatmap_3p1(const SimulationConfig *cfg);
int run_scan_inverse_pmns_filter_3p1(const SimulationConfig *cfg);
int run_scan_inverse_pmns_filter_3p2(const SimulationConfig *cfg);
int run_scan_inverse_construct_23_3p1(const SimulationConfig *cfg);
int run_dune_nd_predict_spectrum(const SimulationConfig *cfg);
int run_dune_fd_fig4_validation(const SimulationConfig *cfg);
int run_dune_nd_fig4_source_line(const SimulationConfig *cfg);

#endif
