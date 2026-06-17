#ifndef OSCILLATION_H
#define OSCILLATION_H

#include "config.h"

#include <complex.h>

double deg_to_rad(double deg);
double oscillation_phase(double dm2_eV2, double baseline_km, double energy_GeV);

double complex transition_amplitude_n(
	int n_flavors,
	int alpha,
	int beta,
	double energy_GeV,
	double baseline_km,

	const double mass_sq[n_flavors],
	const double complex u[n_flavors][n_flavors]);

double probability_with_gaussian_filter_n(
	int n_flavors,
	int alpha,
	int beta,
	double energy_GeV,
	double baseline_km,
	const double mass_sq[n_flavors],
	const double complex u[n_flavors][n_flavors],
	int gaussian_filter_enabled,
	double sigmaE_over_E);

double matter_a_cc_eV2(
	double energy_GeV,
	double density_g_cm3,
	double electron_fraction,
	double a_cc_coeff_eV2_per_GeV_per_gcm3);

double matter_a_sterile_eV2(
	double energy_GeV,
	double density_g_cm3,
	double neutron_fraction,
	double a_cc_coeff_eV2_per_GeV_per_gcm3);

double probability_in_matter_constant_density_n(
	int n_flavors,
	int alpha,
	int beta,
	double energy_GeV,
	double baseline_km,
	const double mass_sq[n_flavors],
	const double complex u[n_flavors][n_flavors],
	int is_antineutrino,
	double density_g_cm3,
	double electron_fraction,
	double neutron_fraction,
	int include_neutral_current_sterile,
	int evolution_steps,
	double a_cc_coeff_eV2_per_GeV_per_gcm3);

double probability_diagonalization_constant_density_n(
	int n_flavors,
	int alpha,
	int beta,
	double energy_GeV,
	double baseline_km,
	const double mass_sq[n_flavors],
	const double complex u[n_flavors][n_flavors],
	int is_antineutrino,
	double density_g_cm3,
	double electron_fraction,
	double neutron_fraction,
	int include_neutral_current_sterile,
	double a_cc_coeff_eV2_per_GeV_per_gcm3);

double probability_with_config_n(
	int n_flavors,
	int alpha,
	int beta,
	double energy_GeV,
	double baseline_km,
	const double mass_sq[n_flavors],
	const double complex u[n_flavors][n_flavors],
	const SimulationConfig *cfg,
	int is_antineutrino);

#endif