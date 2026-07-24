#include "hamiltonian.hpp"
#include "potentials.hpp"
#include "scf_modules.hpp"

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

std::vector<double> make_potential(const FFTGrid& grid) {
    std::vector<double> potential(grid.ngrid, 0.0);
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3d s = grid.frac_coord(i, j, k);
                potential[p] =
                    -0.61 * std::cos(2.0 * M_PI * s[0])
                    + 0.29 * std::sin(2.0 * M_PI * s[1])
                    - 0.18 * std::cos(2.0 * M_PI * (s[0] + s[2]))
                    + 0.07 * std::sin(4.0 * M_PI * s[2]);
            }
        }
    }
    return potential;
}

Eigen::MatrixXcd make_states(int basis_size, int state_count) {
    Eigen::MatrixXcd states(basis_size, state_count);
    for (int ib = 0; ib < state_count; ++ib) {
        for (int ig = 0; ig < basis_size; ++ig) {
            states(ig, ib) = std::complex<double>(
                std::sin(0.31 * ig + 0.47 * ib),
                std::cos(0.19 * ig - 0.23 * ib)
            );
        }
        states.col(ib).normalize();
    }
    return states;
}

std::vector<NonlocalProjector> make_projectors(int basis_size) {
    std::vector<NonlocalProjector> projectors(2);
    projectors[0].beta_G = make_states(basis_size, 1).col(0);
    projectors[0].D = 0.37;
    projectors[1].beta_G = make_states(basis_size, 2).col(1);
    projectors[1].D = -0.21;
    return projectors;
}

Eigen::MatrixXcd scalar_hamiltonian_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& potential,
    const Eigen::MatrixXcd& states,
    const std::vector<NonlocalProjector>& projectors) {

    Eigen::MatrixXcd images(states.rows(), states.cols());
    for (int ib = 0; ib < states.cols(); ++ib) {
        images.col(ib) = apply_hamiltonian_eigen(
            basis,
            fft,
            potential,
            states.col(ib),
            &projectors
        );
    }
    return images;
}

void test_batched_hamiltonian_matches_scalar() {
    const double cell = 7.0;
    const Lattice lattice(
        Eigen::Vector3d(cell, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell)
    );
    const FFTGrid grid(14, 14, 14);
    FFTWorkspace fft(grid);
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 1.35);

    const std::vector<double> potential = make_potential(grid);
    const std::vector<NonlocalProjector> projectors =
        make_projectors(basis.size());

    /*
     * The 19-state case exercises the bounded 16+3 chunk path.  Returning
     * to smaller blocks also checks that cached plans remain usable.
     */
    for (int state_count : {1, 4, 7, 19, 3}) {
        const Eigen::MatrixXcd states =
            make_states(basis.size(), state_count);
        const Eigen::MatrixXcd scalar = scalar_hamiltonian_block(
            basis,
            fft,
            potential,
            states,
            projectors
        );
        const Eigen::MatrixXcd batched = apply_hamiltonian_to_block(
            basis,
            fft,
            potential,
            states,
            &projectors
        );
        require_less(
            (batched - scalar).cwiseAbs().maxCoeff(),
            2.0e-12,
            "batched/scalar Hpsi error for block "
                + std::to_string(state_count)
        );
    }

    if (fft.batch_capacity > 16) {
        throw std::runtime_error(
            "Batched Hamiltonian exceeded its bounded FFT workspace."
        );
    }

    const Eigen::MatrixXcd empty(basis.size(), 0);
    const Eigen::MatrixXcd empty_image = apply_hamiltonian_to_block(
        basis,
        fft,
        potential,
        empty,
        &projectors
    );
    if (empty_image.rows() != basis.size() || empty_image.cols() != 0) {
        throw std::runtime_error("Empty Hamiltonian block shape changed.");
    }
}

void test_nonlocal_projector_block_matches_scalar() {
    const int basis_size = 23;
    const Eigen::MatrixXcd states = make_states(basis_size, 6);
    const std::vector<NonlocalProjector> projectors =
        make_projectors(basis_size);

    Eigen::MatrixXcd scalar(basis_size, states.cols());
    for (int ib = 0; ib < states.cols(); ++ib) {
        const Eigen::VectorXcd state = states.col(ib);
        scalar.col(ib) = apply_nonlocal_projectors(
            projectors,
            state
        );
    }
    const Eigen::MatrixXcd batched = apply_nonlocal_projectors(
        projectors,
        states
    );
    require_less(
        (batched - scalar).cwiseAbs().maxCoeff(),
        2.0e-14,
        "batched/scalar nonlocal-projector error"
    );
}

void test_threaded_fftw_matches_serial() {
#ifdef PWDFT_USE_FFTW_THREADS
    const double cell = 8.0;
    const Lattice lattice(
        Eigen::Vector3d(cell, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell)
    );
    const FFTGrid grid(32, 32, 32);
    FFTWorkspace serial_fft(grid, 1);
    FFTWorkspace threaded_fft(grid, 2);
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 1.8);

    const std::vector<double> potential = make_potential(grid);
    const std::vector<NonlocalProjector> projectors =
        make_projectors(basis.size());
    const Eigen::MatrixXcd states = make_states(basis.size(), 7);

    const Eigen::MatrixXcd serial = apply_hamiltonian_to_block(
        basis,
        serial_fft,
        potential,
        states,
        &projectors
    );
    const Eigen::MatrixXcd threaded = apply_hamiltonian_to_block(
        basis,
        threaded_fft,
        potential,
        states,
        &projectors
    );
    require_less(
        (threaded - serial).cwiseAbs().maxCoeff(),
        3.0e-12,
        "two-thread/serial Hpsi error"
    );
    const std::vector<double> occupations(
        static_cast<std::size_t>(states.cols()), 1.0
    );
    const std::vector<double> serial_density =
        build_density_from_orbitals(
            basis,
            serial_fft,
            states,
            occupations,
            lattice.volume()
        );
    const std::vector<double> threaded_density =
        build_density_from_orbitals(
            basis,
            threaded_fft,
            states,
            occupations,
            lattice.volume()
        );
    double density_error = 0.0;
    for (int p = 0; p < grid.ngrid; ++p) {
        density_error = std::max(
            density_error,
            std::abs(threaded_density[p] - serial_density[p])
        );
    }
    require_less(
        density_error,
        3.0e-13,
        "two-thread/serial density error"
    );
    const std::vector<double> serial_hartree =
        build_hartree_potential(lattice, serial_fft, serial_density);
    const std::vector<double> threaded_hartree =
        build_hartree_potential(lattice, threaded_fft, threaded_density);
    double hartree_error = 0.0;
    for (int p = 0; p < grid.ngrid; ++p) {
        hartree_error = std::max(
            hartree_error,
            std::abs(threaded_hartree[p] - serial_hartree[p])
        );
    }
    require_less(
        hartree_error,
        3.0e-12,
        "two-thread/serial Hartree error"
    );
    if (threaded_fft.thread_count != 2) {
        throw std::runtime_error(
            "The requested FFTW thread count was not retained."
        );
    }
    if (threaded_fft.performance.hamiltonian_vectors != states.cols() ||
        threaded_fft.performance.hamiltonian_block_calls != 1 ||
        threaded_fft.performance.hamiltonian_backward_fft_seconds <= 0.0 ||
        threaded_fft.performance.hamiltonian_forward_fft_seconds <= 0.0 ||
        threaded_fft.performance.density_orbitals != states.cols() ||
        threaded_fft.performance.density_backward_fft_seconds <= 0.0) {
        throw std::runtime_error(
            "Threaded Hpsi performance counters were not populated."
        );
    }
#endif
}

} // namespace

int main() {
    try {
        test_batched_hamiltonian_matches_scalar();
        test_nonlocal_projector_block_matches_scalar();
        test_threaded_fftw_matches_serial();
    } catch (const std::exception& error) {
        std::cerr << "Batched-Hamiltonian test failed: "
                  << error.what() << "\n";
        return 1;
    }

    std::cout << "Batched-Hamiltonian tests passed.\n";
    return 0;
}
