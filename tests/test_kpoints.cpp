#include "input.hpp"
#include "parallel.hpp"
#include "scf.hpp"
#include "scf_modules.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& label) {

    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + ": actual = " + std::to_string(actual)
            + ", expected = " + std::to_string(expected)
        );
    }
}

void test_shifted_basis() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    const Eigen::Vector3d k_cart = lattice.B * Eigen::Vector3d(0.25, 0.0, 0.0);
    PlaneWaveBasis3D basis;
    basis.generate(lattice, k_cart, 2.0);
    if (basis.size() == 0) {
        throw std::runtime_error("Shifted plane-wave basis is empty.");
    }
    for (const GVector& vector : basis.gvectors) {
        require_close(
            (vector.G_cart + k_cart - vector.q_cart).norm(),
            0.0,
            1.0e-14,
            "G+k wavevector"
        );
        require_close(
            vector.kinetic,
            0.5 * vector.q_cart.squaredNorm(),
            1.0e-14,
            "Shifted kinetic energy"
        );
    }
}

void test_global_occupations() {
    std::vector<Eigen::VectorXd> eigenvalues(2);
    eigenvalues[0].resize(2);
    eigenvalues[1].resize(2);
    eigenvalues[0] << -1.0, 1.0;
    eigenvalues[1] << 0.0, 2.0;
    const std::vector<double> weights{0.25, 0.75};

    const KPointOccupationResult zero_temperature =
        compute_kpoint_occupations(
            eigenvalues,
            weights,
            1.5,
            OccupationMode::DegeneracyAwareZeroT,
            {},
            0.0,
            1.0e-10
        );
    require_close(zero_temperature.occupations[0][0], 2.0, 1.0e-14,
                  "First zero-T occupation");
    require_close(zero_temperature.occupations[1][0], 4.0 / 3.0, 1.0e-14,
                  "Weighted partial occupation");
    require_close(zero_temperature.nelec_sum, 1.5, 1.0e-13,
                  "Weighted zero-T electron count");
    require_close(zero_temperature.mu, 0.0, 1.0e-14,
                  "Global zero-T chemical potential");

    const KPointOccupationResult smeared = compute_kpoint_occupations(
        eigenvalues,
        weights,
        1.5,
        OccupationMode::FermiDirac,
        {},
        0.05,
        1.0e-10
    );
    require_close(smeared.nelec_sum, 1.5, 1.0e-12,
                  "Weighted Fermi-Dirac electron count");
    if (!(smeared.entropy > 0.0)) {
        throw std::runtime_error("Weighted Fermi-Dirac entropy must be positive.");
    }
}

void test_two_kpoint_scf() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    const FFTGrid grid(12, 12, 12);
    FFTWorkspace fft(grid);
    std::vector<KPointHamiltonian> kpoints(2);
    kpoints[0].fractional_position = Eigen::Vector3d::Zero();
    kpoints[1].fractional_position = Eigen::Vector3d(0.25, 0.0, 0.0);
    for (KPointHamiltonian& point : kpoints) {
        point.weight = 0.5;
        point.basis.generate(
            lattice,
            lattice.B * point.fractional_position,
            1.5
        );
    }

    SCFOptions options;
    options.nelec = 2.0;
    options.nbands = 1;
    options.occupation_mode = OccupationMode::Fixed;
    options.fixed_occupations = {2.0};
    options.max_iterations = 8;
    options.density_tolerance = 1.0e-10;
    options.energy_tolerance = 1.0e-12;
    options.eigensolver_max_iterations = 30;
    options.eigensolver_tolerance = 1.0e-11;
    options.mixing_alpha = 0.5;
    options.verbosity = SCFVerbosity::Silent;

    const std::vector<double> zero_potential(grid.ngrid, 0.0);
    const KPointSCFResult result = run_kpoint_scf(
        lattice,
        kpoints,
        fft,
        zero_potential,
        0.0,
        options
    );
    if (!result.converged) {
        throw std::runtime_error("Two-k-point jellium SCF did not converge.");
    }
    require_close(result.occupations.nelec_sum, 2.0, 1.0e-12,
                  "SCF weighted occupation count");
    require_close(result.electron_number_from_density, 2.0, 1.0e-10,
                  "SCF integrated electron count");
    if (result.kpoints.size() != 2) {
        throw std::runtime_error("SCF did not preserve both k-point states.");
    }

    const double shifted_kinetic =
        std::pow(std::acos(-1.0), 2) / 512.0;
    require_close(
        result.energy.kinetic,
        shifted_kinetic,
        1.0e-11,
        "MPI-reduced kinetic energy"
    );
    require_close(
        result.kpoints[1].eigenvalues[0]
            - result.kpoints[0].eigenvalues[0],
        shifted_kinetic,
        1.0e-11,
        "MPI-gathered eigenvalue separation"
    );
    const double uniform_density = options.nelec / lattice.volume();
    double maximum_density_error = 0.0;
    for (double density : result.density) {
        maximum_density_error = std::max(
            maximum_density_error,
            std::abs(density - uniform_density)
        );
    }
    require_close(
        maximum_density_error,
        0.0,
        1.0e-11,
        "MPI-reduced uniform density"
    );

    const parallel::KPointDistribution distribution(2);
    int local_ownership_failures = 0;
    int local_orbital_blocks = 0;
    for (int ik = 0; ik < 2; ++ik) {
        const bool owns_orbitals = result.kpoints[ik].orbitals.size() != 0;
        if (result.kpoints[ik].owner_rank != distribution.owner(ik) ||
            owns_orbitals != distribution.owns(ik)) {
            ++local_ownership_failures;
        }
        if (owns_orbitals) {
            ++local_orbital_blocks;
        }
    }
    if (parallel::sum(local_ownership_failures) != 0) {
        throw std::runtime_error(
            "MPI k-point orbital ownership is inconsistent."
        );
    }
    if (parallel::sum(local_orbital_blocks) != 2) {
        throw std::runtime_error(
            "Each k-point orbital block must exist on exactly one rank."
        );
    }
}

void test_gamma_wrapper_equivalence() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    const FFTGrid grid(12, 12, 12);
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 1.5);
    SCFOptions options;
    options.nelec = 2.0;
    options.nbands = 1;
    options.occupation_mode = OccupationMode::Fixed;
    options.fixed_occupations = {2.0};
    options.max_iterations = 8;
    options.density_tolerance = 1.0e-10;
    options.energy_tolerance = 1.0e-12;
    options.eigensolver_max_iterations = 30;
    options.eigensolver_tolerance = 1.0e-11;
    options.mixing_alpha = 0.5;
    options.verbosity = SCFVerbosity::Silent;
    const std::vector<double> zero_potential(grid.ngrid, 0.0);

    FFTWorkspace legacy_fft(grid);
    const SCFResult legacy = run_scf(
        lattice,
        basis,
        legacy_fft,
        zero_potential,
        0.0,
        {},
        options
    );
    KPointHamiltonian gamma;
    gamma.weight = 1.0;
    gamma.basis = basis;
    FFTWorkspace kpoint_fft(grid);
    const KPointSCFResult generalized = run_kpoint_scf(
        lattice,
        {gamma},
        kpoint_fft,
        zero_potential,
        0.0,
        options
    );
    if (!legacy.converged || !generalized.converged) {
        throw std::runtime_error("Gamma equivalence SCF did not converge.");
    }
    require_close(
        generalized.variational_energy,
        legacy.variational_energy,
        1.0e-11,
        "Gamma generalized/legacy energy"
    );
    double maximum_density_error = 0.0;
    for (int index = 0; index < grid.ngrid; ++index) {
        maximum_density_error = std::max(
            maximum_density_error,
            std::abs(generalized.density[index] - legacy.density[index])
        );
    }
    require_close(maximum_density_error, 0.0, 1.0e-11,
                  "Gamma generalized/legacy density");
}

} // namespace

int main(int argc, char** argv) {
    try {
        parallel::Environment environment(argc, argv);
        try {
            test_shifted_basis();
            test_global_occupations();
            test_gamma_wrapper_equivalence();
            test_two_kpoint_scf();
            if (parallel::is_root()) {
                std::cout << "Multi-k-point tests passed with "
                    << parallel::size() << " MPI rank(s).\n";
            }
            return 0;
        } catch (const std::exception& error) {
            if (parallel::is_root()) {
                std::cerr << "Multi-k-point test failed: "
                    << error.what() << "\n";
            }
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "MPI initialization failed in multi-k-point test: "
            << error.what() << "\n";
        return 1;
    }
}
