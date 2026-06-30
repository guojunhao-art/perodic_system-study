#include "mixing.hpp"
#include "scf_modules.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void PulayMixer::clear() {
    trial_history.clear();
    residual_history.clear();
}

int PulayMixer::history_size() const {
    return static_cast<int>(residual_history.size());
}

void update_adaptive_mixing(
    MixingState& mix,
    double drho) {

    if (mix.previous_drho < 0.0) {
        mix.previous_drho = drho;
        return;
    }

    /*
     * If residual increased, back off immediately.
     */
    if (drho > 1.05 * mix.previous_drho) {
        mix.alpha = std::max(
            mix.alpha_min,
            0.5 * mix.alpha
        );

        mix.good_steps = 0;
    } else {
        /*
         * If residual keeps decreasing, cautiously increase alpha.
         */
        mix.good_steps += 1;

        if (mix.good_steps >= 5) {
            mix.alpha = std::min(
                mix.alpha_max,
                1.10 * mix.alpha
            );

            mix.good_steps = 0;
        }
    }

    mix.previous_drho = drho;
}

double grid_dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dV) {

    if (a.size() != b.size()) {
        throw std::runtime_error("grid_dot size mismatch.");
    }

    double s = 0.0;

    for (int p = 0; p < static_cast<int>(a.size()); ++p) {
        s += a[p] * b[p];
    }

    return dV * s;
}

std::vector<double> density_residual(
    const std::vector<double>& rho_in,
    const std::vector<double>& rho_out) {

    if (rho_in.size() != rho_out.size()) {
        throw std::runtime_error("density_residual size mismatch.");
    }

    std::vector<double> r(rho_in.size(), 0.0);

    for (int p = 0; p < static_cast<int>(rho_in.size()); ++p) {
        r[p] = rho_out[p] - rho_in[p];
    }

    return r;
}

double density_norm(
    const std::vector<double>& rho,
    double dV) {

    return std::sqrt(grid_dot(rho, rho, dV));
}

void renormalize_density(
    std::vector<double>& rho,
    double dV,
    double target_nelec) {

    const double nelec_now =
        electron_number_from_density(rho, dV);

    if (std::abs(nelec_now) < 1.0e-14) {
        throw std::runtime_error("Cannot renormalize zero density.");
    }

    const double scale = target_nelec / nelec_now;

    for (double& x : rho) {
        x *= scale;
    }
}
Eigen::VectorXd solve_pulay_coefficients(
    const std::vector<std::vector<double>>& residual_history,
    double dV,
    double regularization) {

    const int m =
        static_cast<int>(residual_history.size());

    if (m == 0) {
        throw std::runtime_error("Empty Pulay residual history.");
    }

    if (m == 1) {
        Eigen::VectorXd c(1);
        c[0] = 1.0;
        return c;
    }

    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(m + 1, m + 1);

    Eigen::VectorXd b =
        Eigen::VectorXd::Zero(m + 1);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            A(i, j) =
                grid_dot(
                    residual_history[i],
                    residual_history[j],
                    dV
                );
        }

        /*
         * Small regularization avoids singular Pulay matrices.
         */
        A(i, i) += regularization;

        A(i, m) = -1.0;
        A(m, i) = -1.0;
    }

    b[m] = -1.0;

    Eigen::VectorXd x =
        A.fullPivLu().solve(b);

    return x.head(m);
}

std::vector<double> combine_density_history(
    const std::vector<std::vector<double>>& rho_history,
    const Eigen::VectorXd& coeffs) {

    const int m =
        static_cast<int>(rho_history.size());

    if (m == 0) {
        throw std::runtime_error("Empty density history.");
    }

    if (coeffs.size() != m) {
        throw std::runtime_error("Pulay coefficient size mismatch.");
    }

    const int ngrid =
        static_cast<int>(rho_history[0].size());

    std::vector<double> rho(ngrid, 0.0);

    for (int i = 0; i < m; ++i) {
        if (static_cast<int>(rho_history[i].size()) != ngrid) {
            throw std::runtime_error("Density history grid size mismatch.");
        }

        for (int p = 0; p < ngrid; ++p) {
            rho[p] += coeffs[i] * rho_history[i][p];
        }
    }

    return rho;
}

double pulay_mix_density(
    PulayMixer& mixer,
    std::vector<double>& rho_inout,
    const std::vector<double>& rho_out,
    double dV,
    double target_nelec) {

    if (rho_inout.size() != rho_out.size()) {
        throw std::runtime_error("pulay_mix_density size mismatch.");
    }

    const int ngrid =
        static_cast<int>(rho_inout.size());

    /*
     * 1. Residual R = rho_out - rho_in.
     */
    std::vector<double> R =
        density_residual(
            rho_inout,
            rho_out
        );

    const double drho =
        density_norm(R, dV);

    /*
     * 2. Build linear-mixed trial density:
     *
     *    rho_trial = rho_in + alpha R
     */
    std::vector<double> rho_trial(ngrid, 0.0);

    for (int p = 0; p < ngrid; ++p) {
        rho_trial[p] =
            rho_inout[p] + mixer.alpha * R[p];
    }

    renormalize_density(
        rho_trial,
        dV,
        target_nelec
    );

    /*
     * 3. Push history.
     */
    mixer.trial_history.push_back(rho_trial);
    mixer.residual_history.push_back(R);

    if (mixer.history_size() > mixer.max_history) {
        mixer.trial_history.erase(
            mixer.trial_history.begin()
        );

        mixer.residual_history.erase(
            mixer.residual_history.begin()
        );
    }

    /*
     * 4. If history too short, use linear mixing.
     */
    if (mixer.history_size() < mixer.min_history) {
        rho_inout = rho_trial;
        return drho;
    }

    /*
     * 5. Pulay coefficients from residual history.
     */
    Eigen::VectorXd coeffs =
        solve_pulay_coefficients(
            mixer.residual_history,
            dV,
            mixer.regularization
        );

    /*
     * 6. Combine trial densities.
     */
    std::vector<double> rho_next =
        combine_density_history(
            mixer.trial_history,
            coeffs
        );

    /*
     * 7. Clean up density.
     *
     * Pulay extrapolation can create tiny negative density.
     * For LDA exchange rho^(1/3), negative rho is problematic.
     */
    for (double& x : rho_next) {
        if (x < 0.0) {
            x = 0.0;
        }
    }

    renormalize_density(
        rho_next,
        dV,
        target_nelec
    );

    rho_inout = rho_next;

    return drho;
}
