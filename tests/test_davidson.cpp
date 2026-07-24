#include "eigensolver.hpp"
#include "hamiltonian.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_less(
    double value,
    double tolerance,
    const std::string& label) {

    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

std::vector<double> make_test_potential(const FFTGrid& grid) {
    std::vector<double> potential(grid.ngrid, 0.0);

    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3d s = grid.frac_coord(i, j, k);
                potential[p] =
                    -0.55 * std::cos(2.0 * M_PI * s[0])
                    + 0.31 * std::cos(2.0 * M_PI * s[1])
                    - 0.23 * std::cos(2.0 * M_PI * s[2])
                    + 0.17 * std::cos(
                        2.0 * M_PI * (s[0] + s[1])
                    );
            }
        }
    }
    return potential;
}

Eigen::MatrixXcd make_deterministic_matrix(
    int rows,
    int cols,
    int offset = 0) {

    Eigen::MatrixXcd matrix(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const double x = static_cast<double>(
                (i + 1) * (j + offset + 2)
            );
            matrix(i, j) = std::complex<double>(
                std::sin(0.137 * x + 0.31 * (j + offset)),
                std::cos(0.193 * x - 0.17 * (j + offset))
            );
        }
    }
    return matrix;
}

void test_block_correction_orthonormalization() {
    constexpr int rows = 128;
    constexpr int subspace_columns = 7;

    const Eigen::MatrixXcd subspace = orthonormalize_columns(
        make_deterministic_matrix(rows, subspace_columns)
    );
    const Eigen::MatrixXcd independent =
        make_deterministic_matrix(rows, 3, 13);

    Eigen::MatrixXcd raw(rows, 4);
    raw.leftCols(3) = independent;
    raw.col(3) = raw.col(0) - 0.375 * raw.col(1);

    const Eigen::MatrixXcd coefficients =
        make_deterministic_matrix(
            subspace_columns,
            raw.cols(),
            29
        );
    raw.noalias() += subspace * coefficients;

    const Eigen::MatrixXcd correction =
        orthonormalize_correction_block(subspace, raw);

    if (correction.cols() != 3) {
        throw std::runtime_error(
            "Block correction orthogonalization did not drop a "
            "dependent direction."
        );
    }

    const double cross_error =
        (subspace.adjoint() * correction).norm();
    const double internal_error = (
        correction.adjoint() * correction
        - Eigen::MatrixXcd::Identity(
            correction.cols(),
            correction.cols()
        )
    ).norm();

    require_less(
        cross_error,
        1.0e-11,
        "block correction/subspace orthogonality error"
    );
    require_less(
        internal_error,
        1.0e-11,
        "block correction internal orthogonality error"
    );
}

void test_davidson_against_dense_reference() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    const FFTGrid grid(16, 16, 16);
    FFTWorkspace fft(grid);
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 1.45);

    const std::vector<double> potential = make_test_potential(grid);
    const int nbands = 4;
    const int ntrial = std::min(basis.size(), nbands + 4);
    const Eigen::MatrixXcd trials =
        initial_low_kinetic_trials(basis.size(), ntrial);

    const Eigen::MatrixXcd identity =
        Eigen::MatrixXcd::Identity(basis.size(), basis.size());
    Eigen::MatrixXcd dense_hamiltonian =
        apply_hamiltonian_to_block(
            basis,
            fft,
            potential,
            identity
        );
    const double dense_hermiticity =
        (dense_hamiltonian - dense_hamiltonian.adjoint()).norm();
    require_less(dense_hermiticity, 1.0e-11, "dense H Hermiticity error");
    dense_hamiltonian = 0.5 * (
        dense_hamiltonian + dense_hamiltonian.adjoint()
    );

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> dense_solver(
        dense_hamiltonian
    );
    if (dense_solver.info() != Eigen::Success) {
        throw std::runtime_error("Dense reference diagonalization failed.");
    }

    const double residual_tolerance = 2.0e-10;
    const DavidsonResult result = davidson_lowest_eigenstates(
        basis,
        fft,
        potential,
        nbands,
        trials,
        100,
        basis.size() + nbands,
        residual_tolerance,
        1.0e-6,
        nullptr,
        false
    );
    if (!result.converged) {
        throw std::runtime_error("Davidson did not converge in regression test.");
    }

    const double eigenvalue_error = (
        result.eigenvalues
        - dense_solver.eigenvalues().head(nbands)
    ).cwiseAbs().maxCoeff();
    require_less(eigenvalue_error, 2.0e-9, "Davidson eigenvalue error");

    const double maximum_residual = *std::max_element(
        result.residual_norms.begin(),
        result.residual_norms.end()
    );
    require_less(
        maximum_residual,
        1.01 * residual_tolerance,
        "Davidson maximum residual"
    );

    /*
     * With no max-subspace restart, cached W = HV means that each accepted
     * independent direction is acted on by H only once. A full-subspace
     * recomputation on every iteration would violate this bound.
     */
    std::cout << "Davidson iterations = " << result.iterations
              << "  Hpsi applications = "
              << result.hamiltonian_applications
              << "  H blocks = "
              << result.hamiltonian_block_calls
              << "  basis size = " << basis.size() << "\n";

    if (result.hamiltonian_block_calls <= 0 ||
        result.hamiltonian_block_calls >= result.hamiltonian_applications) {
        throw std::runtime_error(
            "Davidson Hamiltonian applications were not grouped into blocks."
        );
    }
    if (result.hamiltonian_applications > basis.size()) {
        throw std::runtime_error(
            "Davidson W=HV cache applied H repeatedly to old directions."
        );
    }
    if (result.hamiltonian_block_calls > result.iterations) {
        throw std::runtime_error(
            "Davidson made more H block calls than iterations."
        );
    }

    const DavidsonTimingBreakdown& timing = result.timing;
    const double detailed_other = timing.detailed_other_seconds();
    const double total_other =
        result.total_seconds
        - result.hamiltonian_seconds
        - result.subspace_diagonalization_seconds;
    if (timing.initial_orthonormalization_seconds < 0.0 ||
        timing.projected_matrix_seconds < 0.0 ||
        timing.ritz_rotation_seconds < 0.0 ||
        timing.residual_preconditioner_seconds < 0.0 ||
        timing.result_copy_seconds < 0.0 ||
        timing.correction_orthogonalization_seconds < 0.0 ||
        timing.restart_seconds < 0.0 ||
        timing.correction_block_assembly_seconds < 0.0 ||
        timing.subspace_expansion_seconds < 0.0 ||
        detailed_other <= 0.0 ||
        detailed_other > total_other + 1.0e-9) {
        throw std::runtime_error(
            "Davidson detailed timing is incomplete or double-counted."
        );
    }
}

} // namespace

int main() {
    try {
        test_block_correction_orthonormalization();
        test_davidson_against_dense_reference();
        std::cout << "Davidson regression test passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Davidson regression test failed: "
                  << error.what() << "\n";
        return 1;
    }
}
