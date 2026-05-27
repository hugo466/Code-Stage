#include "io_csv.h"
#include <math.h>

FILE *open_probability_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }

    fprintf(out, "energy_GeV,dm41_eV2,Umu4_re,Umu4_im,Ue4_re,Ue4_im," "A_mumu_re,A_mumu_im,P_mumu_disappearance," "A_mue_re,A_mue_im,P_mue_appearance\n");

    return out;
}

void write_probability_row(FILE *out, double energy_GeV, double dm41_eV2, double complex u_mu4, double complex u_e4, double complex amp_mumu, double p_mumu, double complex amp_mue, double p_mue) {

    fprintf(out, "%.3f,%.10g,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n", energy_GeV, dm41_eV2, creal(u_mu4), cimag(u_mu4), creal(u_e4), cimag(u_e4), creal(amp_mumu), cimag(amp_mumu), p_mumu, creal(amp_mue), cimag(amp_mue), p_mue);
}

FILE *open_probability_csv_3p2(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }

    fprintf(out, "baseline_km,energy_GeV,dm41_eV2,dm54_eV2,dm51_eV2," "A_mumu_re,A_mumu_im,P_mumu_disappearance," "A_mue_re,A_mue_im,P_mue_appearance\n");
    return out;
}

void write_probability_row_3p2(FILE *out, double baseline_km, double energy_GeV, double dm41_eV2, double dm54_eV2, double dm51_eV2,
    double complex amp_mumu, double p_mumu, double complex amp_mue, double p_mue) {

    fprintf(out, "%.3f,%.3f,%.10g,%.10g,%.10g,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
        baseline_km, energy_GeV, dm41_eV2, dm54_eV2, dm51_eV2,
        creal(amp_mumu), cimag(amp_mumu), p_mumu,
        creal(amp_mue), cimag(amp_mue), p_mue);
}

FILE *open_distance_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) return NULL;
    fprintf(out, "baseline_km,dm41_eV2,P_mumu_disappearance,P_mue_appearance,P_mumu_3nu,P_mue_3nu\n");
    return out;
}

void write_distance_row(FILE *out, double baseline_km, double dm41_eV2, double p_mumu, double p_mue, double p_mumu_3nu, double p_mue_3nu) {

    fprintf(out, "%.3f,%.10g,%.10f,%.10f,%.10f,%.10f\n", baseline_km, dm41_eV2, p_mumu, p_mue, p_mumu_3nu, p_mue_3nu);
}

FILE *open_delta_pmue_heatmap_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }

    fprintf(out, "energy_GeV,dm41_eV2,P_mue_3pns,P_mue_3nu,delta_P_mue\n");
    return out;
}

void write_delta_pmue_heatmap_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double p_mue_3pns,
    double p_mue_3nu,
    double delta_p_mue) {

    fprintf(out, "%.3f,%.10g,%.10f,%.10f,%.10f\n", energy_GeV, dm41_eV2, p_mue_3pns, p_mue_3nu, delta_p_mue);
}

FILE *open_delta_pmue_heatmap_3p2_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }

    fprintf(out, "energy_GeV,dm41_eV2,dm54_eV2,P_mue_3p2,P_mue_3nu,delta_P_mue\n");
    return out;
}

void write_delta_pmue_heatmap_3p2_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double dm54_eV2,
    double p_mue_3p2,
    double p_mue_3nu,
    double delta_p_mue) {

    fprintf(out, "%.3f,%.10g,%.10g,%.10f,%.10f,%.10f\n", energy_GeV, dm41_eV2, dm54_eV2, p_mue_3p2, p_mue_3nu, delta_p_mue);
}

FILE *open_delta_pmumu_heatmap_3p2_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }

    fprintf(out, "energy_GeV,dm41_eV2,dm54_eV2,P_mumu_3p2,P_mumu_3nu,delta_P_mumu\n");
    return out;
}

void write_delta_pmumu_heatmap_3p2_row(
    FILE *out,
    double energy_GeV,
    double dm41_eV2,
    double dm54_eV2,
    double p_mumu_3p2,
    double p_mumu_3nu,
    double delta_p_mumu) {

    fprintf(out, "%.3f,%.10g,%.10g,%.10f,%.10f,%.10f\n", energy_GeV, dm41_eV2, dm54_eV2, p_mumu_3p2, p_mumu_3nu, delta_p_mumu);
}

FILE *open_cp_heatmap_3p1_csv(const char *path) {
    FILE *out = fopen(path, "w");
    if (!out) {
        return NULL;
    }
    fprintf(out, "baseline_km,log10_energy_GeV,delta41_deg,dm41_eV2,P_mue_nu,P_mue_antinu,ACP\n");
    return out;
}

void write_cp_heatmap_3p1_row(
    FILE *out,
    double baseline_km,
    double energy_GeV,
    double delta41_deg,
    double dm41_eV2,
    double p_mue_nu,
    double p_mue_antinu,
    double acp) {

    fprintf(out, "%.4f,%.8f,%.6g,%.10g,%.10f,%.10f,%.10f\n",
        baseline_km, log10(energy_GeV), delta41_deg, dm41_eV2, p_mue_nu, p_mue_antinu, acp);
}
