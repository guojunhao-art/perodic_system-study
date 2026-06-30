#pragma once

#include <Eigen/Dense>
#include <vector>

struct MixingState {
    double alpha = 0.1;
    double alpha_min = 0.01;
    double alpha_max = 0.2;
    double previous_drho = -1.0;
    int good_steps = 0;
};

void update_adaptive_mixing(
    MixingState& mix,
    double drho);

double grid_dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double dV);

std::vector<double> density_residual(
    const std::vector<double>& rho_in,
    const std::vector<double>& rho_out);

double density_norm(
    const std::vector<double>& rho,
    double dV);

void renormalize_density(
    std::vector<double>& rho,
    double dV,
    double target_nelec);

struct PulayMixer {
    int max_history = 6;
    int min_history = 2;
    double alpha = 0.1;
    double regularization = 1.0e-12;
    std::vector<std::vector<double>> trial_history;
    std::vector<std::vector<double>> residual_history;

    void clear();
    int history_size() const;
};

Eigen::VectorXd solve_pulay_coefficients(
    const std::vector<std::vector<double>>& residual_history,
    double dV,
    double regularization = 1.0e-12);

std::vector<double> combine_density_history(
    const std::vector<std::vector<double>>& rho_history,
    const Eigen::VectorXd& coeffs);

double pulay_mix_density(
    PulayMixer& mixer,
    std::vector<double>& rho_inout,
    const std::vector<double>& rho_out,
    double dV,
    double target_nelec);
