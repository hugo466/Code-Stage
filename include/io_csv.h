#ifndef IO_CSV_H
#define IO_CSV_H

#include <complex.h>
#include <stdio.h>

/* Scan en énergie */
FILE *open_probability_csv(const char *path);
void write_probability_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double complex u_mu4,
    double complex u_e4,
    double complex amp_mumu,
    double p_mumu,
    double complex amp_mue,
    double p_mue);

/* Scan en énergie 3+2 */
FILE *open_probability_csv_3p2(const char *path);
void write_probability_row_3p2(
    FILE *out,
    double baseline_km,
    double energy_GeV,
    double dm41_eV2,
    double dm54_eV2,
    double dm51_eV2,
    double complex amp_mumu,
    double p_mumu,
    double complex amp_mue,
    double p_mue);

/* Scan en distance */
FILE *open_distance_csv(const char *path);
void write_distance_row(
    FILE *out,
    double baseline_km,
    double dm41_eV2,
    double p_mumu,
    double p_mue,
    double p_mumu_3nu,
    double p_mue_3nu);

/* Heatmap Delta Pmue */
FILE *open_delta_pmue_heatmap_csv(const char *path);
void write_delta_pmue_heatmap_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double p_mue_3pns,
    double p_mue_3nu,
    double delta_p_mue);

FILE *open_delta_pmue_heatmap_3p2_csv(const char *path);
void write_delta_pmue_heatmap_3p2_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double dm54_eV2,
    double p_mue_3p2,
    double p_mue_3nu,
    double delta_p_mue);

FILE *open_delta_pmumu_heatmap_3p2_csv(const char *path);
void write_delta_pmumu_heatmap_3p2_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double dm54_eV2,
    double p_mumu_3p2,
    double p_mumu_3nu,
    double delta_p_mumu);

/* Heatmap ACP = P(numu->nue) - P(antinumu->antinue) vs E x delta41 */
FILE *open_cp_heatmap_3p1_csv(const char *path);
void write_cp_heatmap_3p1_row(
    FILE *out,
    double baseline_km,
    double energy_GeV,
    double delta41_deg,
    double dm41_eV2,
    double p_mue_nu,
    double p_mue_antinu,
    double acp);

#endif
