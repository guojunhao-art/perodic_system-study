#pragma once

#include "core.hpp"

#include <Eigen/Dense>
#include <vector>

Eigen::MatrixXcd orthonormalize_columns(
    const Eigen::MatrixXcd& A,
    double drop_tol = 1.0e-12);

Eigen::MatrixXcd append_columns(
    const Eigen::MatrixXcd& A,
    const std::vector<Eigen::VectorXcd>& new_cols);

struct DavidsonResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXcd eigenvectors;
    std::vector<double> residual_norms;
    int iterations = 0;
    bool converged = false;
};

Eigen::MatrixXcd initial_low_kinetic_trials(
    int nbasis,
    int ntrial);

DavidsonResult davidson_lowest_eigenstates(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    int nbands,
    const Eigen::MatrixXcd& V_initial,
    int max_iter = 100,
    int max_subspace = 40,
    double residual_tol = 1.0e-10,
    double denom_floor = 1.0e-6,
    const std::vector<NonlocalProjector>* projectors = nullptr);

struct CanonicalBlock {
    Eigen::MatrixXcd C;
    Eigen::MatrixXcd HC;
    Eigen::VectorXd eps;
    Eigen::MatrixXcd R;
    std::vector<double> residual_norms;
};

Eigen::MatrixXcd orthonormalize_columns_RMM(
    const Eigen::MatrixXcd& A,
    double drop_tol = 1.0e-12);

CanonicalBlock canonicalize_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& C_in);

Eigen::MatrixXcd build_preconditioned_correction_block(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& R,
    const Eigen::VectorXd& eps,
    double denom_floor = 1.0e-6);

Eigen::MatrixXcd orthogonalize_correction_against_C(
    const Eigen::MatrixXcd& C,
    const Eigen::MatrixXcd& T_raw,
    double drop_tol = 1.0e-12);

Eigen::MatrixXcd project_correction_against_C_keep_magnitude(
    const Eigen::MatrixXcd& C,
    const Eigen::MatrixXcd& T_raw);

double max_residual_norm(
    const std::vector<double>& residual_norms);

CanonicalBlock augmented_rmm_step(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const CanonicalBlock& cur,
    double denom_floor,
    double residual_tol);

Eigen::VectorXd solve_block_diis_coefficients(
    const std::vector<Eigen::MatrixXcd>& R_hist,
    double regularization = 1.0e-14);

Eigen::MatrixXcd combine_block_diis_orbitals(
    const std::vector<Eigen::MatrixXcd>& C_hist,
    const Eigen::VectorXd& coeffs);

struct BlockRMMResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXcd eigenvectors;
    std::vector<double> residual_norms;
    int iterations = 0;
    bool converged = false;
};

BlockRMMResult block_rmm_diis_eigenstates(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& C_initial,
    int max_iter = 100,
    int history_size = 5,
    double residual_tol = 1.0e-10,
    double denom_floor = 1.0e-6,
    double step_scale = 1.0);
