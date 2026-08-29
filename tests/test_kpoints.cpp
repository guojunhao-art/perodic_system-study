#include "calculation.hpp"
#include "input.hpp"
#include "parallel.hpp"
#include "scf.hpp"
#include "scf_convergence.hpp"
#include "scf_modules.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
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

void test_eigensolver_tolerance_schedule() {
    EigensolverToleranceSchedule schedule(1.0e-7, 2.0e-10, 1.0e-7);
    require_close(schedule.current(), 1.0e-7, 0.0,
                  "Initial eigensolver tolerance");

    schedule.advance(1.0e-3);
    require_close(schedule.current(), 2.0e-8, 1.0e-22,
                  "Density-controlled eigensolver tolerance");
    schedule.advance(1.0e-1);
    require_close(schedule.current(), 2.0e-8, 1.0e-22,
                  "Eigensolver tolerance must not loosen");
    schedule.advance(1.0e-5);
    require_close(schedule.current(), 2.0e-9, 1.0e-23,
                  "Second density-controlled tolerance");
    schedule.advance(1.0e-7);
    require_close(schedule.current(), 2.0e-10, 1.0e-24,
                  "Final eigensolver tolerance");
    if (!schedule.at_final_tolerance()) {
        throw std::runtime_error("Tolerance schedule did not reach final accuracy.");
    }

    EigensolverToleranceSchedule capped(1.0e-5, 1.0e-10, 1.0e-7);
    capped.advance(0.0);
    require_close(capped.current(), 1.0e-6, 1.0e-20,
                  "One-decade tightening cap");
    capped.force_final_tolerance();
    require_close(capped.current(), 1.0e-10, 0.0,
                  "Forced final eigensolver tolerance");
}

void test_vasp_style_scf_convergence() {
    if (!scf_energy_changes_converged(
            9.0e-7, -4.0e-7, 1.0e-6)) {
        throw std::runtime_error(
            "SCF convergence rejected energy changes below EDIFF."
        );
    }
    if (scf_energy_changes_converged(
            1.1e-6, 4.0e-7, 1.0e-6)) {
        throw std::runtime_error(
            "SCF convergence accepted |dE| above EDIFF."
        );
    }
    if (scf_energy_changes_converged(
            4.0e-7, -1.1e-6, 1.0e-6)) {
        throw std::runtime_error(
            "SCF convergence accepted |d eps| above EDIFF."
        );
    }
}

void test_automatic_fft_grid() {
    const double cell_bohr = 12.0 * ANGSTROM_TO_BOHR;
    const Lattice lattice(
        Eigen::Vector3d(cell_bohr, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell_bohr, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell_bohr)
    );
    KPointHamiltonian gamma;
    gamma.basis.generate(
        lattice, Eigen::Vector3d::Zero(), 10.0
    );

    const std::array<int, 3> dimensions =
        automatic_fft_grid_dimensions({gamma});
    if (dimensions != std::array<int, 3>{{72, 72, 72}}) {
        throw std::runtime_error(
            "The 12-Angstrom, 10-Ha automatic FFT grid is not 72^3."
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

    std::vector<Eigen::VectorXd> spin_eigenvalues(2);
    spin_eigenvalues[0].resize(2);
    spin_eigenvalues[1].resize(2);
    spin_eigenvalues[0] << -1.0, 1.0;
    spin_eigenvalues[1] << 0.0, 2.0;
    const KPointOccupationResult spin_occupations =
        compute_kpoint_occupations(
            spin_eigenvalues,
            {1.0, 1.0},
            1.0,
            OccupationMode::DegeneracyAwareZeroT,
            {},
            0.0,
            1.0e-10,
            1.0,
            2.0
        );
    require_close(
        spin_occupations.occupations[0][0],
        1.0,
        1.0e-14,
        "Spin-up state capacity"
    );
    require_close(
        spin_occupations.occupations[1][0],
        0.0,
        1.0e-14,
        "Common-Fermi-level spin-down occupation"
    );
    require_close(
        spin_occupations.nelec_sum,
        1.0,
        1.0e-14,
        "Spin-resolved electron count"
    );
}

KPointHamiltonian make_gamma_hamiltonian(
    const Lattice& lattice,
    double cutoff) {

    KPointHamiltonian gamma;
    gamma.weight = 1.0;
    gamma.basis.generate(
        lattice, Eigen::Vector3d::Zero(), cutoff
    );
    return gamma;
}

SCFOptions make_zero_temperature_options() {
    SCFOptions options;
    options.nelec = 2.0;
    options.nbands = 2;
    options.occupation_mode =
        OccupationMode::DegeneracyAwareZeroT;
    options.max_iterations = 12;
    options.density_tolerance = 1.0e-10;
    options.energy_tolerance = 1.0e-12;
    options.eigensolver_max_iterations = 40;
    options.eigensolver_initial_tolerance = 1.0e-11;
    options.eigensolver_tolerance = 1.0e-11;
    options.mixing_alpha = 0.5;
    options.verbosity = SCFVerbosity::Silent;
    return options;
}

void test_collinear_spin_scf() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    const FFTGrid grid(12, 12, 12);
    const KPointHamiltonian gamma =
        make_gamma_hamiltonian(lattice, 1.5);
    const std::vector<double> zero_potential(grid.ngrid, 0.0);

    SCFOptions nonspin_options =
        make_zero_temperature_options();
    FFTWorkspace nonspin_fft(grid);
    const KPointSCFResult nonspin = run_kpoint_scf(
        lattice,
        {gamma},
        nonspin_fft,
        zero_potential,
        0.0,
        nonspin_options
    );

    SCFOptions unpolarized_options = nonspin_options;
    unpolarized_options.nspin = 2;
    unpolarized_options.starting_magnetization = 0.0;
    FFTWorkspace unpolarized_fft(grid);
    const KPointSCFResult unpolarized = run_kpoint_scf(
        lattice,
        {gamma},
        unpolarized_fft,
        zero_potential,
        0.0,
        unpolarized_options
    );
    if (!nonspin.converged || !unpolarized.converged) {
        throw std::runtime_error(
            "Unpolarized nspin equivalence SCF did not converge."
        );
    }
    require_close(
        unpolarized.variational_energy,
        nonspin.variational_energy,
        2.0e-11,
        "nspin=2 unpolarized energy"
    );
    require_close(
        unpolarized.magnetization,
        0.0,
        1.0e-12,
        "nspin=2 unpolarized magnetization"
    );
    require_close(
        unpolarized.spin_electron_counts[0],
        1.0,
        1.0e-12,
        "Unpolarized spin-up electron count"
    );
    require_close(
        unpolarized.spin_electron_counts[1],
        1.0,
        1.0e-12,
        "Unpolarized spin-down electron count"
    );

    SCFOptions hydrogen_options =
        make_zero_temperature_options();
    hydrogen_options.nelec = 1.0;
    hydrogen_options.nspin = 2;
    hydrogen_options.starting_magnetization = 1.0;
    hydrogen_options.xc_functional =
        XCFunctional::PerdewBurkeErnzerhof;
    FFTWorkspace hydrogen_fft(grid);
    const KPointSCFResult hydrogen = run_kpoint_scf(
        lattice,
        {gamma},
        hydrogen_fft,
        zero_potential,
        0.0,
        hydrogen_options
    );
    if (!hydrogen.converged) {
        throw std::runtime_error(
            "One-electron spin-PBE SCF did not converge."
        );
    }
    require_close(
        hydrogen.magnetization,
        1.0,
        1.0e-12,
        "One-electron magnetization"
    );
    require_close(
        hydrogen.spin_electron_counts[0],
        1.0,
        1.0e-12,
        "One-electron spin-up count"
    );
    require_close(
        hydrogen.spin_electron_counts[1],
        0.0,
        1.0e-12,
        "One-electron spin-down count"
    );
    if (hydrogen.kpoints.size() != 2 ||
        hydrogen.spin_densities.size() != 2) {
        throw std::runtime_error(
            "Spin-resolved electronic state was not retained."
        );
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
    options.verbosity = SCFVerbosity::Compact;

    const std::vector<double> zero_potential(grid.ngrid, 0.0);
    std::ostringstream log;
    const KPointSCFResult result = run_kpoint_scf(
        lattice,
        kpoints,
        fft,
        zero_potential,
        0.0,
        options,
        {},
        parallel::is_root() ? &log : nullptr
    );
    if (!result.converged) {
        throw std::runtime_error("Two-k-point jellium SCF did not converge.");
    }
    if (!scf_energy_changes_converged(
            result.final_energy_change,
            result.final_band_energy_change,
            options.energy_tolerance)) {
        throw std::runtime_error(
            "Two-k-point SCF stopped without satisfying dE and d eps."
        );
    }
    require_close(
        result.final_eigensolver_tolerance,
        options.eigensolver_tolerance,
        0.0,
        "Final multi-k-point eigensolver refinement"
    );
    require_close(result.occupations.nelec_sum, 2.0, 1.0e-12,
                  "SCF weighted occupation count");
    require_close(result.electron_number_from_density, 2.0, 1.0e-10,
                  "SCF integrated electron count");
    if (result.kpoints.size() != 2) {
        throw std::runtime_error("SCF did not preserve both k-point states.");
    }
    if (result.performance.fft.hamiltonian_vectors !=
            result.eigensolver_hamiltonian_applications ||
        result.performance.fft.hamiltonian_block_calls !=
            result.eigensolver_hamiltonian_block_calls ||
        result.performance.fft.hamiltonian_backward_fft_seconds <= 0.0 ||
        result.performance.fft.density_orbitals <= 0 ||
        result.eigensolver_detail.detailed_other_seconds() <= 0.0 ||
        result.eigensolver_detail.detailed_other_seconds() >
            result.eigensolver_other_seconds + 1.0e-9) {
        throw std::runtime_error(
            "SCF performance counters are incomplete or inconsistent."
        );
    }
    if (parallel::is_root()) {
        const std::string output = log.str();
        if (output.find("d eps") == std::string::npos ||
            output.find("ncg") == std::string::npos ||
            output.find("rms(c)") == std::string::npos ||
            output.find("DAV:") == std::string::npos ||
            output.find("Hpsi breakdown") == std::string::npos ||
            output.find("ortho/Ritz/other") == std::string::npos ||
            output.find("Davidson breakdown") == std::string::npos ||
            output.find("Davidson reuse") == std::string::npos ||
            output.find("unaccounted") == std::string::npos) {
            throw std::runtime_error(
                "VASP-style DAV diagnostics are incomplete."
            );
        }
        std::ostringstream summary_marker;
        summary_marker << " " << std::setw(4) << result.iterations
            << " F= ";
        if (log.str().find(summary_marker.str()) == std::string::npos) {
            throw std::runtime_error(
                "The compact SCF summary did not report the actual "
                "SCF iteration count."
            );
        }
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
    if (!scf_energy_changes_converged(
            legacy.final_energy_change,
            legacy.final_band_energy_change,
            options.energy_tolerance) ||
        !scf_energy_changes_converged(
            generalized.final_energy_change,
            generalized.final_band_energy_change,
            options.energy_tolerance)) {
        throw std::runtime_error(
            "Gamma SCF stopped without satisfying dE and d eps."
        );
    }
    require_close(
        legacy.final_eigensolver_tolerance,
        options.eigensolver_tolerance,
        0.0,
        "Final Gamma eigensolver refinement"
    );
    require_close(
        generalized.final_eigensolver_tolerance,
        options.eigensolver_tolerance,
        0.0,
        "Final generalized eigensolver refinement"
    );
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

    SCFOptions pbe_options = options;
    pbe_options.xc_functional =
        XCFunctional::PerdewBurkeErnzerhof;
    FFTWorkspace pbe_legacy_fft(grid);
    const SCFResult pbe_legacy = run_scf(
        lattice,
        basis,
        pbe_legacy_fft,
        zero_potential,
        0.0,
        {},
        pbe_options
    );
    FFTWorkspace pbe_kpoint_fft(grid);
    const KPointSCFResult pbe_generalized = run_kpoint_scf(
        lattice,
        {gamma},
        pbe_kpoint_fft,
        zero_potential,
        0.0,
        pbe_options
    );
    if (!pbe_legacy.converged || !pbe_generalized.converged) {
        throw std::runtime_error("PBE Gamma equivalence SCF did not converge.");
    }
    require_close(
        pbe_generalized.variational_energy,
        pbe_legacy.variational_energy,
        1.0e-11,
        "PBE Gamma generalized/legacy energy"
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        parallel::Environment environment(argc, argv);
        try {
            test_eigensolver_tolerance_schedule();
            test_vasp_style_scf_convergence();
            test_automatic_fft_grid();
            test_shifted_basis();
            test_global_occupations();
            test_collinear_spin_scf();
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
