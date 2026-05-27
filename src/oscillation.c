#include "oscillation.h"
#include "constants.h"

#include <math.h>
#include <stdio.h>

static void matvec_mul_complex(int n, const double complex h[n][n], const double complex x[n], double complex out[n]) {
    for (int row = 0; row < n; ++row) {
        double complex acc = 0.0 + 0.0 * I;
        for (int col = 0; col < n; ++col) {
            acc += h[row][col] * x[col];
        }
        out[row] = acc;
    }
}

static int solve_linear_system_complex(int n, double complex a[n][n], double complex b[n], double complex x[n]) {
    const double eps = 1.0e-18;

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = cabs(a[col][col]);
        for (int row = col + 1; row < n; ++row) {
            const double cand = cabs(a[row][col]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = row;
            }
        }

        if (pivot_abs < eps) {
            return 1;
        }

        if (pivot != col) {
            for (int k = col; k < n; ++k) {
                const double complex tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
            {
                const double complex tmpb = b[col];
                b[col] = b[pivot];
                b[pivot] = tmpb;
            }
        }

        for (int row = col + 1; row < n; ++row) {
            const double complex factor = a[row][col] / a[col][col];
            a[row][col] = 0.0 + 0.0 * I;
            for (int k = col + 1; k < n; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        double complex acc = b[row];
        for (int k = row + 1; k < n; ++k) {
            acc -= a[row][k] * x[k];
        }
        if (cabs(a[row][row]) < eps) {
            return 1;
        }
        x[row] = acc / a[row][row];
    }

    return 0;
}

static int crank_nicolson_step_constant_h(int n, const double complex h[n][n], double dx_km, double complex psi[n]) {
    double complex a[n][n], b_mat[n][n], rhs[n], next[n];
    const double complex half_step = 0.5 * dx_km * I;

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const double complex id = (row == col) ? (1.0 + 0.0 * I) : (0.0 + 0.0 * I);
            a[row][col] = id + half_step * h[row][col];
            b_mat[row][col] = id - half_step * h[row][col];
        }
    }

    matvec_mul_complex(n, b_mat, psi, rhs);
    if (solve_linear_system_complex(n, a, rhs, next) != 0) {
        return 1;
    }

    {
        double norm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            psi[i] = next[i];
            norm2 += creal(psi[i] * conj(psi[i]));
        }
        if (!(norm2 > 0.0) || !isfinite(norm2)) {
            return 1;
        }
        {
            const double inv_norm = 1.0 / sqrt(norm2);
            for (int i = 0; i < n; ++i) {
                psi[i] *= inv_norm;
            }
        }
    }

    return 0;
}

/* ========== QR Eigensolve for Complex Matrices ========== */

static int jacobi_eigensolve_hermitian_complex(
    int n,
    double complex a[n][n],
    double complex eigenvalues[n],
    double complex eigenvectors[n][n],
    int max_iter) {

    /* Correct complex-hermitian Jacobi method.
     *
     * For hermitian A, the off-diagonal element a_{pq} = r * e^{i*phi}.
     * We apply the unitary rotation J (acting on the (p,q) 2x2 sub-block):
     *
     *   J[p][p] =  c               J[p][q] =  s * e^{i*phi}
     *   J[q][p] = -s * e^{-i*phi}  J[q][q] =  c
     *
     * with c = cos(theta), s = sin(theta), tau = (a_qq - a_pp) / (2*r),
     * t = s/c chosen to zero out a'_{pq}.
     *
     * A' = J^H A J  zeros out (p,q) and keeps diagonals real.
     * Eigenvectors accumulate: V -> V * J.
     */

    /* Initialize eigenvectors to identity */
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            eigenvectors[i][j] = (i == j) ? 1.0 + 0.0*I : 0.0 + 0.0*I;

    /* Copy a to working matrix */
    double complex a_work[n][n];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a_work[i][j] = a[i][j];

    const double eps = 1.0e-12;

    for (int iter = 0; iter < max_iter; ++iter) {
        /* Find largest off-diagonal element */
        int p = 0, qi = 1;
        double max_elem = cabs(a_work[0][1]);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double elem = cabs(a_work[i][j]);
                if (elem > max_elem) { max_elem = elem; p = i; qi = j; }
            }

        if (max_elem < eps) break; /* Converged */

        /* apq = r * e^{i*phi} */
        double complex apq_c = a_work[p][qi];
        double r   = cabs(apq_c);
        double phi = carg(apq_c);

        double complex ephip = cexp(I * phi);   /* e^{i*phi}  */
        double complex ephim = cexp(-I * phi);  /* e^{-i*phi} */

        double app_r  = creal(a_work[p][p]);
        double aqq_r  = creal(a_work[qi][qi]);
        double tau    = (aqq_r - app_r) / (2.0 * r);

        /* Numerically stable computation of tan(theta) */
        double t_val;
        if (tau >= 0.0)
            t_val =  1.0 / (tau + sqrt(1.0 + tau * tau));
        else
            t_val = -1.0 / (-tau + sqrt(1.0 + tau * tau));

        double c = 1.0 / sqrt(1.0 + t_val * t_val);
        double s = c * t_val;

        /* Step 1: right-multiply A by J  (update columns p and qi):
         *   new_col_p[i]  = c * old_col_p[i]  - s * e^{-i*phi} * old_col_qi[i]
         *   new_col_qi[i] = s * e^{i*phi} * old_col_p[i]  + c * old_col_qi[i]  */
        for (int i = 0; i < n; ++i) {
            double complex aip = a_work[i][p];
            double complex aiq = a_work[i][qi];
            a_work[i][p]  = c * aip - s * ephim * aiq;
            a_work[i][qi] = s * ephip * aip + c * aiq;
        }

        /* Step 2: left-multiply A by J^H  (update rows p and qi):
         *   new_row_p[j]  = c * old_row_p[j]  - s * e^{i*phi} * old_row_qi[j]
         *   new_row_qi[j] = s * e^{-i*phi} * old_row_p[j] + c * old_row_qi[j]  */
        for (int j = 0; j < n; ++j) {
            double complex apj = a_work[p][j];
            double complex aqj = a_work[qi][j];
            a_work[p][j]  = c * apj - s * ephip * aqj;
            a_work[qi][j] = s * ephim * apj + c * aqj;
        }

        /* Force exact zeros at (p,qi) and (qi,p) */
        double new_app  = app_r - t_val * r;
        double new_aqq  = aqq_r + t_val * r;
        a_work[p][p]    = new_app  + 0.0*I;
        a_work[qi][qi]  = new_aqq  + 0.0*I;
        a_work[p][qi]   = 0.0 + 0.0*I;
        a_work[qi][p]   = 0.0 + 0.0*I;

        /* Accumulate eigenvectors: V -> V * J (right-multiply by J)
         *   new_col_p[i]  = c * old_col_p[i]  - s * e^{-i*phi} * old_col_qi[i]
         *   new_col_qi[i] = s * e^{i*phi} * old_col_p[i]  + c * old_col_qi[i]  */
        for (int i = 0; i < n; ++i) {
            double complex vip = eigenvectors[i][p];
            double complex viq = eigenvectors[i][qi];
            eigenvectors[i][p]  = c * vip - s * ephim * viq;
            eigenvectors[i][qi] = s * ephip * vip + c * viq;
        }
    }

    /* Extract eigenvalues from diagonal */
    for (int i = 0; i < n; ++i)
        eigenvalues[i] = a_work[i][i];

    return 0;
}

static void print_mixing_difference_matrix_once(
    int n_flavors,
    double energy_GeV,
    double baseline_km,
    const double complex u_vac[n_flavors][n_flavors],
    const double complex u_eff[n_flavors][n_flavors]) {

    static int already_printed = 0;
    if (already_printed) {
        return;
    }
    already_printed = 1;

    printf("\n=== Difference matrix DeltaU = U_vacuum - U_effective (matter diagonalization) ===\n");
    printf("Context: E = %.6f GeV, L = %.6f km\n", energy_GeV, baseline_km);
    printf("Format: (Re, Im)\n");

    for (int row = 0; row < n_flavors; ++row) {
        printf("row %d : ", row);
        for (int col = 0; col < n_flavors; ++col) {
            const double complex delta = u_vac[row][col] - u_eff[row][col];
            printf("(% .6e,% .6e) ", creal(delta), cimag(delta));
        }
        printf("\n");
    }
    printf("=== End DeltaU ===\n\n");
}

/* ========== Probability via Hamiltonian Diagonalization ========== */

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
    double a_cc_coeff_eV2_per_GeV_per_gcm3) {

    if (n_flavors <= 0 || alpha < 0 || beta < 0 || alpha >= n_flavors || beta >= n_flavors) {
        return 0.0;
    }
    if (energy_GeV <= 0.0 || baseline_km <= 0.0) {
        return (alpha == beta) ? 1.0 : 0.0;
    }

    const double sign = is_antineutrino ? -1.0 : 1.0;
    const double phase_coeff = (OSCILLATION_PHASE_FACTOR) / energy_GeV;
    const double a_cc = matter_a_cc_eV2(energy_GeV, density_g_cm3, electron_fraction, a_cc_coeff_eV2_per_GeV_per_gcm3);
    const double a_s = matter_a_sterile_eV2(energy_GeV, density_g_cm3, neutron_fraction, a_cc_coeff_eV2_per_GeV_per_gcm3);

    /* Build H in flavor basis */
    double complex h[n_flavors][n_flavors];
    for (int row = 0; row < n_flavors; ++row) {
        for (int col = 0; col < n_flavors; ++col) {
            h[row][col] = 0.0 + 0.0 * I;
        }
    }

    for (int i = 0; i < n_flavors; ++i) {
        for (int row = 0; row < n_flavors; ++row) {
            const double complex u_row_i = is_antineutrino ? conj(u[row][i]) : u[row][i];
            for (int col = 0; col < n_flavors; ++col) {
                const double complex u_col_i_conj = is_antineutrino ? u[col][i] : conj(u[col][i]);
                h[row][col] += phase_coeff * mass_sq[i] * u_row_i * u_col_i_conj;
            }
        }
    }

    h[FLAVOR_E][FLAVOR_E] += sign * phase_coeff * a_cc;
    if (include_neutral_current_sterile) {
        for (int sterile = 3; sterile < n_flavors; ++sterile) {
            h[sterile][sterile] += sign * phase_coeff * a_s;
        }
    }

    /* Diagonalize H */
    double complex eigenvalues[n_flavors];
    double complex eigenvectors[n_flavors][n_flavors];
    
    if (jacobi_eigensolve_hermitian_complex(n_flavors, h, eigenvalues, eigenvectors, 500) != 0) {
        return NAN;
    }

    print_mixing_difference_matrix_once(
        n_flavors,
        energy_GeV,
        baseline_km,
        u,
        eigenvectors);

    /* Check for NaN in eigenvalues */
    for (int i = 0; i < n_flavors; ++i) {
        if (!isfinite(creal(eigenvalues[i])) || !isfinite(cimag(eigenvalues[i]))) {
            return NAN;
        }
    }

    /* Compute S_βα(L,E) = Σ_k U_eff[β][k] exp(-i λ_k L) (U_eff^†)[k][α] */
    double complex s_ba = 0.0 + 0.0 * I;
    for (int k = 0; k < n_flavors; ++k) {
        /* λ_k already includes the factor 1/E from the Hamiltonian construction */
        const double lambda_k = creal(eigenvalues[k]);
        const double phase = -lambda_k * baseline_km;
        const double complex exp_phase = cexp(I * phase);
        const double complex term = eigenvectors[beta][k] * exp_phase * conj(eigenvectors[alpha][k]);
        s_ba += term;
        
        if (!isfinite(creal(term)) || !isfinite(cimag(term))) {
            return NAN;
        }
    }

    {
        const double p = creal(s_ba * conj(s_ba));
        if (!isfinite(p)) {
            return NAN;
        }
        if (p < 0.0) return 0.0;
        if (p > 1.0) return 1.0;
        return p;
    }
}

double matter_a_cc_eV2(
    double energy_GeV,
    double density_g_cm3,
    double electron_fraction,
    double a_cc_coeff_eV2_per_GeV_per_gcm3) {

    return a_cc_coeff_eV2_per_GeV_per_gcm3 * electron_fraction * density_g_cm3 * energy_GeV;
}

double matter_a_sterile_eV2(
    double energy_GeV,
    double density_g_cm3,
    double neutron_fraction,
    double a_cc_coeff_eV2_per_GeV_per_gcm3) {

    return 0.5 * a_cc_coeff_eV2_per_GeV_per_gcm3 * neutron_fraction * density_g_cm3 * energy_GeV;
}

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
    double a_cc_coeff_eV2_per_GeV_per_gcm3) {

    if (n_flavors <= 0 || alpha < 0 || beta < 0 || alpha >= n_flavors || beta >= n_flavors) {
        return 0.0;
    }
    if (energy_GeV <= 0.0 || baseline_km <= 0.0) {
        return (alpha == beta) ? 1.0 : 0.0;
    }

    const int n_steps = (evolution_steps > 0) ? evolution_steps : 1;
    const double dx_km = baseline_km / (double)n_steps;
    const double sign = is_antineutrino ? -1.0 : 1.0;
    const double phase_coeff = (OSCILLATION_PHASE_FACTOR) / energy_GeV;
    const double a_cc = matter_a_cc_eV2(energy_GeV, density_g_cm3, electron_fraction, a_cc_coeff_eV2_per_GeV_per_gcm3);
    const double a_s = matter_a_sterile_eV2(energy_GeV, density_g_cm3, neutron_fraction, a_cc_coeff_eV2_per_GeV_per_gcm3);

    double complex h[n_flavors][n_flavors];
    for (int row = 0; row < n_flavors; ++row) {
        for (int col = 0; col < n_flavors; ++col) {
            h[row][col] = 0.0 + 0.0 * I;
        }
    }

    for (int i = 0; i < n_flavors; ++i) {
        for (int row = 0; row < n_flavors; ++row) {
            const double complex u_row_i = is_antineutrino ? conj(u[row][i]) : u[row][i];
            for (int col = 0; col < n_flavors; ++col) {
                const double complex u_col_i_conj = is_antineutrino ? u[col][i] : conj(u[col][i]);
                h[row][col] += phase_coeff * mass_sq[i] * u_row_i * u_col_i_conj;
            }
        }
    }

    h[FLAVOR_E][FLAVOR_E] += sign * phase_coeff * a_cc;
    if (include_neutral_current_sterile) {
        for (int sterile = 3; sterile < n_flavors; ++sterile) {
            h[sterile][sterile] += sign * phase_coeff * a_s;
        }
    }

    double complex psi[n_flavors];
    for (int i = 0; i < n_flavors; ++i) {
        psi[i] = 0.0 + 0.0 * I;
    }
    psi[alpha] = 1.0 + 0.0 * I;

    for (int step = 0; step < n_steps; ++step) {
        if (crank_nicolson_step_constant_h(n_flavors, h, dx_km, psi) != 0) {
            return NAN;
        }
    }

    {
        const double p = creal(psi[beta] * conj(psi[beta]));
        if (!isfinite(p)) {
            return NAN;
        }
        if (p < 0.0) return 0.0;
        if (p > 1.0) return 1.0;
        return p;
    }
}

double deg_to_rad(double deg) {
    return deg * PI / 180.0;
}

double oscillation_phase(double dm2_eV2, double baseline_km, double energy_GeV) {
    return OSCILLATION_PHASE_FACTOR * dm2_eV2 * baseline_km / energy_GeV;
}

double complex transition_amplitude_n(
    int n_flavors,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    const double mass_sq[n_flavors],
    const double complex u[n_flavors][n_flavors]) {

    double complex amplitude = 0.0 + 0.0 * I;
    for (int i = 0; i < n_flavors; ++i) {
        const double phi_i = oscillation_phase(mass_sq[i], baseline_km, energy_GeV);
        amplitude += u[beta][i] * conj(u[alpha][i]) * cexp(- I * phi_i);
    }
    return amplitude;
}

double probability_with_gaussian_filter_n(
    int n_flavors,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    const double mass_sq[n_flavors],
    const double complex u[n_flavors][n_flavors],
    int gaussian_filter_enabled,
    double sigmaE_over_E) {

    double probability = 0.0;

    for (int i = 0; i < n_flavors; ++i) {
        probability += pow(cabs(u[beta][i]), 2.0) * pow(cabs(u[alpha][i]), 2.0);
    }

    for (int i = 0; i < n_flavors; ++i) {
        for (int j = 0; j < i; ++j) {
            const double delta_phi = oscillation_phase(mass_sq[i] - mass_sq[j], baseline_km, energy_GeV);
            const double complex coeff = u[beta][i] * conj(u[alpha][i]) * conj(u[beta][j]) * u[alpha][j];

            double damping = 1.0;
            if (gaussian_filter_enabled) {
                const double sigma_delta_phi = delta_phi * sigmaE_over_E;
                damping = exp(- 0.5 * sigma_delta_phi * sigma_delta_phi);
            }

            probability += 2.0 * damping * creal(coeff * cexp(- I * delta_phi));
        }
    }

    return probability;
}

double probability_with_config_n(
    int n_flavors,
    int alpha,
    int beta,
    double energy_GeV,
    double baseline_km,
    const double mass_sq[n_flavors],
    const double complex u[n_flavors][n_flavors],
    const SimulationConfig *cfg,
    int is_antineutrino) {

    if (!cfg || !cfg->matter_effects_enabled) {
        return probability_with_gaussian_filter_n(
            n_flavors,
            alpha,
            beta,
            energy_GeV,
            baseline_km,
            mass_sq,
            u,
            cfg ? cfg->gaussian_filter_enabled : 0,
            cfg ? cfg->sigmaE_over_E : 0.0);
    }

    if (!cfg->gaussian_filter_enabled || cfg->sigmaE_over_E <= 0.0) {
        return probability_diagonalization_constant_density_n(
            n_flavors,
            alpha,
            beta,
            energy_GeV,
            baseline_km,
            mass_sq,
            u,
            is_antineutrino,
            cfg->matter_density_g_cm3,
            cfg->matter_electron_fraction,
            cfg->matter_neutron_fraction,
            cfg->matter_include_neutral_current_sterile,
            cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3);
    }

    {
        const double sigma_rel = cfg->sigmaE_over_E;
        double max_dm2 = 0.0;
        for (int i = 0; i < n_flavors; ++i) {
            for (int j = 0; j < i; ++j) {
                const double dm2 = fabs(mass_sq[i] - mass_sq[j]);
                if (dm2 > max_dm2) {
                    max_dm2 = dm2;
                }
            }
        }

        const double sigma_phase = fabs(oscillation_phase(max_dm2, baseline_km, energy_GeV) * sigma_rel);
        int n_samples = 9 + 2 * (int)(sigma_phase / 2.0);
        if (n_samples < 9) {
            n_samples = 9;
        }
        if (n_samples > 61) {
            n_samples = 61;
        }
        if ((n_samples % 2) == 0) {
            ++n_samples;
        }

        const double x_min = -4.0;
        const double x_max = 4.0;
        const double dx = (x_max - x_min) / (double)(n_samples - 1);

        double smeared_probability = 0.0;
        double weight_sum = 0.0;

        for (int i = 0; i < n_samples; ++i) {
            const double x = x_min + (double)i * dx;
            const double w = exp(-0.5 * x * x);
            double shifted_energy = energy_GeV * (1.0 + sigma_rel * x);
            if (shifted_energy < 1.0e-6) {
                shifted_energy = 1.0e-6;
            }
            {
                const double p_i = probability_diagonalization_constant_density_n(
                n_flavors,
                alpha,
                beta,
                shifted_energy,
                baseline_km,
                mass_sq,
                u,
                is_antineutrino,
                cfg->matter_density_g_cm3,
                cfg->matter_electron_fraction,
                cfg->matter_neutron_fraction,
                cfg->matter_include_neutral_current_sterile,
                cfg->matter_a_cc_coeff_eV2_per_GeV_per_gcm3);

                if (isfinite(p_i)) {
                    smeared_probability += w * p_i;
                    weight_sum += w;
                }
            }
        }
        if (weight_sum <= 0.0) {
            return NAN;
        }
        {
            const double p = smeared_probability / weight_sum;
            if (p < 0.0) return 0.0;
            if (p > 1.0) return 1.0;
            return p;
        }
    }
}