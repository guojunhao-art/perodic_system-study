#include "scf.hpp"

#include "eigensolver.hpp"
#include "mixing.hpp"
#include "parallel.hpp"
#include "potentials.hpp"
#include "scf_convergence.hpp"
#include "symmetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr long long openmp_minimum_work = 32768;

double maximum_residual(const std::vector<double>& residuals) {
    double maximum = 0.0;
    for (double residual : residuals) {
        maximum = std::max(maximum, residual);
    }
    return maximum;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

FFTPerformanceCounters sum_fft_performance(
    const FFTPerformanceCounters& local) {

    FFTPerformanceCounters total;
    total.hamiltonian_vectors =
        parallel::sum(local.hamiltonian_vectors);
    total.hamiltonian_block_calls =
        parallel::sum(local.hamiltonian_block_calls);
    total.hamiltonian_scatter_seconds =
        parallel::sum(local.hamiltonian_scatter_seconds);
    total.hamiltonian_backward_fft_seconds =
        parallel::sum(local.hamiltonian_backward_fft_seconds);
    total.hamiltonian_local_multiply_seconds =
        parallel::sum(local.hamiltonian_local_multiply_seconds);
    total.hamiltonian_forward_fft_seconds =
        parallel::sum(local.hamiltonian_forward_fft_seconds);
    total.hamiltonian_gather_kinetic_seconds =
        parallel::sum(local.hamiltonian_gather_kinetic_seconds);
    total.hamiltonian_nonlocal_seconds =
        parallel::sum(local.hamiltonian_nonlocal_seconds);
    total.density_orbitals =
        parallel::sum(local.density_orbitals);
    total.density_scatter_seconds =
        parallel::sum(local.density_scatter_seconds);
    total.density_backward_fft_seconds =
        parallel::sum(local.density_backward_fft_seconds);
    total.density_accumulation_seconds =
        parallel::sum(local.density_accumulation_seconds);
    return total;
}

DavidsonTimingBreakdown sum_davidson_timing(
    const DavidsonTimingBreakdown& local) {

    std::vector<double> values{
        local.initial_orthonormalization_seconds,
        local.projected_matrix_seconds,
        local.ritz_rotation_seconds,
        local.residual_preconditioner_seconds,
        local.result_copy_seconds,
        local.correction_orthogonalization_seconds,
        local.restart_seconds,
        local.correction_block_assembly_seconds,
        local.subspace_expansion_seconds
    };
    parallel::sum_in_place(values);

    DavidsonTimingBreakdown total;
    total.initial_orthonormalization_seconds = values[0];
    total.projected_matrix_seconds = values[1];
    total.ritz_rotation_seconds = values[2];
    total.residual_preconditioner_seconds = values[3];
    total.result_copy_seconds = values[4];
    total.correction_orthogonalization_seconds = values[5];
    total.restart_seconds = values[6];
    total.correction_block_assembly_seconds = values[7];
    total.subspace_expansion_seconds = values[8];
    total.projected_matrix_full_builds =
        parallel::sum(local.projected_matrix_full_builds);
    total.projected_matrix_incremental_updates =
        parallel::sum(local.projected_matrix_incremental_updates);
    total.projected_matrix_ritz_reuses =
        parallel::sum(local.projected_matrix_ritz_reuses);
    total.correction_blocks =
        parallel::sum(local.correction_blocks);
    total.correction_reorthogonalizations =
        parallel::sum(local.correction_reorthogonalizations);
    return total;
}

SCFPerformanceBreakdown reduce_performance(
    const SCFPerformanceBreakdown& local) {

    SCFPerformanceBreakdown reduced;
    reduced.input_potential_seconds =
        parallel::maximum(local.input_potential_seconds);
    reduced.eigensolver_seconds =
        parallel::maximum(local.eigensolver_seconds);
    reduced.occupations_seconds =
        parallel::maximum(local.occupations_seconds);
    reduced.density_energy_seconds =
        parallel::maximum(local.density_energy_seconds);
    reduced.output_potential_energy_seconds =
        parallel::maximum(local.output_potential_energy_seconds);
    reduced.mixing_seconds =
        parallel::maximum(local.mixing_seconds);
    reduced.fft = sum_fft_performance(local.fft);
    return reduced;
}

int electronic_state_index(
    int spin,
    int kpoint,
    int kpoint_count) {

    return spin * kpoint_count + kpoint;
}

std::vector<double> sum_spin_densities(
    const std::vector<std::vector<double>>& spin_densities) {

    if (spin_densities.empty()) {
        return {};
    }
    std::vector<double> total(spin_densities.front().size(), 0.0);
    for (const std::vector<double>& density : spin_densities) {
        if (density.size() != total.size()) {
            throw std::runtime_error(
                "Spin-density grids have inconsistent sizes."
            );
        }
        for (int p = 0; p < static_cast<int>(total.size()); ++p) {
            total[p] += density[p];
        }
    }
    return total;
}

double spin_density_residual(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<double>>& output,
    double dV) {

    if (input.size() != output.size() || input.empty()) {
        throw std::runtime_error(
            "Spin-density residual channel mismatch."
        );
    }
    const std::vector<double> charge_input =
        sum_spin_densities(input);
    const std::vector<double> charge_output =
        sum_spin_densities(output);
    double residual = density_norm(
        density_residual(charge_input, charge_output), dV
    );
    if (input.size() == 2) {
        std::vector<double> magnetization_input(
            charge_input.size(), 0.0
        );
        std::vector<double> magnetization_output(
            charge_input.size(), 0.0
        );
        for (int p = 0;
             p < static_cast<int>(charge_input.size());
             ++p) {
            magnetization_input[p] =
                input[0][p] - input[1][p];
            magnetization_output[p] =
                output[0][p] - output[1][p];
        }
        residual = std::max(
            residual,
            density_norm(
                density_residual(
                    magnetization_input,
                    magnetization_output
                ),
                dV
            )
        );
    }
    return residual;
}

void validate_inputs(
    const std::vector<KPointHamiltonian>& kpoints,
    const FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    const SCFOptions& options,
    const KPointSCFInitialGuess& initial_guess) {

    if (kpoints.empty()) {
        throw std::runtime_error("A k-point SCF calculation needs at least one k point.");
    }
    if (!std::isfinite(options.nelec) || options.nelec < 0.0) {
        throw std::runtime_error("SCF electron number cannot be negative.");
    }
    if (options.nbands <= 0) {
        throw std::runtime_error("SCF band count must be positive.");
    }
    if (options.nspin != 1 && options.nspin != 2) {
        throw std::runtime_error("SCF nspin must be 1 or 2.");
    }
    if (!std::isfinite(options.starting_magnetization) ||
        std::abs(options.starting_magnetization) >
            options.nelec + 1.0e-12) {
        throw std::runtime_error(
            "The starting magnetization must be finite and satisfy "
            "|M| <= nelect."
        );
    }
    if (options.nspin == 1 &&
        std::abs(options.starting_magnetization) > 1.0e-14) {
        throw std::runtime_error(
            "A nonzero starting magnetization requires nspin = 2."
        );
    }
    if (static_cast<int>(ionic_potential.size()) != fft.grid.ngrid) {
        throw std::runtime_error("Ionic potential size does not match FFT grid.");
    }
    if (options.max_iterations <= 0 ||
        options.eigensolver_max_iterations <= 0) {
        throw std::runtime_error(
            "SCF and eigensolver iteration limits must be positive."
        );
    }
    if (options.occupation_mode == OccupationMode::FermiDirac &&
        options.smearing_sigma <= 0.0) {
        throw std::runtime_error(
            "Fermi-Dirac occupations require positive smearing sigma."
        );
    }
    if (options.occupation_mode == OccupationMode::Fixed &&
        static_cast<int>(options.fixed_occupations.size()) != options.nbands) {
        throw std::runtime_error(
            "Fixed occupations must contain one value per band."
        );
    }
    if (options.nspin == 2 &&
        options.occupation_mode == OccupationMode::Fixed) {
        throw std::runtime_error(
            "Fixed occupations are not supported with nspin = 2."
        );
    }
    if (!initial_guess.density.empty() &&
        static_cast<int>(initial_guess.density.size()) != fft.grid.ngrid) {
        throw std::runtime_error("Initial density size does not match FFT grid.");
    }
    if (!initial_guess.spin_densities.empty()) {
        if (static_cast<int>(initial_guess.spin_densities.size())
            != options.nspin) {
            throw std::runtime_error(
                "Initial spin-density channels do not match nspin."
            );
        }
        for (const std::vector<double>& density :
             initial_guess.spin_densities) {
            if (static_cast<int>(density.size()) != fft.grid.ngrid) {
                throw std::runtime_error(
                    "An initial spin-density grid has the wrong size."
                );
            }
        }
    }
    if (!initial_guess.orbitals.empty() &&
        initial_guess.orbitals.size() !=
            static_cast<std::size_t>(options.nspin) * kpoints.size()) {
        throw std::runtime_error(
            "Initial orbital blocks must match nspin times the k-point count."
        );
    }
    if (!initial_guess.orbital_kpoints.empty() &&
        initial_guess.orbital_kpoints.size() != kpoints.size()) {
        throw std::runtime_error(
            "Initial orbital k-point provenance has the wrong size."
        );
    }

    double weight_sum = 0.0;
    for (int ik = 0; ik < static_cast<int>(kpoints.size()); ++ik) {
        const KPointHamiltonian& point = kpoints[ik];
        if (!point.fractional_position.allFinite() ||
            !std::isfinite(point.weight) || point.weight <= 0.0) {
            throw std::runtime_error(
                "K-point coordinates and weights must be finite; weights must be positive."
            );
        }
        if (point.basis.size() < options.nbands) {
            throw std::runtime_error(
                "A k-point plane-wave basis has fewer vectors than nbands."
            );
        }
        if (!initial_guess.orbitals.empty()) {
            for (int spin = 0; spin < options.nspin; ++spin) {
                const Eigen::MatrixXcd& orbitals =
                    initial_guess.orbitals[electronic_state_index(
                        spin, ik, static_cast<int>(kpoints.size())
                    )];
                if (orbitals.size() != 0 &&
                    (orbitals.rows() != point.basis.size() ||
                     orbitals.cols() < options.nbands)) {
                    throw std::runtime_error(
                        "Initial k-point orbitals have incompatible "
                        "dimensions."
                    );
                }
            }
        }
        weight_sum += point.weight;
    }
    if (std::abs(weight_sum - 1.0) > 1.0e-12) {
        throw std::runtime_error("Normalized k-point weights must sum to one.");
    }
}

void print_header(std::ostream& out) {
    out << "\n"
        << "       N       E                     dE"
        << "             d eps       ncg     rms"
        << "          rms(c)\n";
}

void print_iteration(
    std::ostream& out,
    int iteration,
    double energy,
    double signed_energy_change,
    double signed_band_energy_change,
    double density_residual,
    int hamiltonian_applications,
    double eigensolver_residual,
    bool print_density_residual) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "DAV: " << std::setw(4) << iteration + 1
        << "  " << std::scientific << std::setprecision(12)
        << std::setw(20) << energy
        << "  " << std::setprecision(4) << std::setw(12)
        << signed_energy_change
        << "  " << std::setw(12) << signed_band_energy_change
        << "  " << std::setw(7) << hamiltonian_applications
        << "  " << std::setw(12) << eigensolver_residual
        << (print_density_residual ? "  " : "              ");
    if (print_density_residual) {
        out << std::setw(12) << density_residual;
    }
    out << "\n";
    out.flags(flags);
    out.precision(precision);
}

void print_summary(
    std::ostream& out,
    const KPointSCFResult& result,
    double signed_energy_change) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    const double free_energy_with_ions =
        result.energy.free_energy + result.energy.ion_smooth;
    const double sigma0_with_ions =
        result.energy.sigma0_estimate + result.energy.ion_smooth;
    const FFTPerformanceCounters& fft_timing = result.performance.fft;
    const double hpsi_accounted =
        fft_timing.hamiltonian_scatter_seconds
        + fft_timing.hamiltonian_backward_fft_seconds
        + fft_timing.hamiltonian_local_multiply_seconds
        + fft_timing.hamiltonian_forward_fft_seconds
        + fft_timing.hamiltonian_gather_kinetic_seconds
        + fft_timing.hamiltonian_nonlocal_seconds;
    const double hpsi_overhead = std::max(
        0.0,
        result.eigensolver_hamiltonian_seconds - hpsi_accounted
    );
    const DavidsonTimingBreakdown& davidson =
        result.eigensolver_detail;
    const double davidson_unaccounted = std::max(
        0.0,
        result.eigensolver_other_seconds
            - davidson.detailed_other_seconds()
    );
    out << "--------------------------------------------------------------------------------\n"
        << " " << std::setw(4) << result.iterations
        << " F= " << std::scientific << std::setprecision(12)
        << std::setw(20) << free_energy_with_ions
        << " E0= " << std::setw(20) << sigma0_with_ions
        << "  d E = " << std::setprecision(4) << std::setw(12)
        << signed_energy_change << "\n"
        << "  global mu = " << std::setprecision(12)
        << result.occupations.mu
        << "  weighted Ne = " << result.occupations.nelec_sum;
    if (result.spin_densities.size() == 2) {
        out << "  magnetization = " << result.magnetization
            << " mu_B";
    }
    out << "\n"
        << "  eigensolver work: N_Hpsi = "
        << result.eigensolver_hamiltonian_applications
        << "  N_Hblock = "
        << result.eigensolver_hamiltonian_block_calls
        << "  N_iter = " << result.eigensolver_iterations
        << "  restarts = " << result.eigensolver_restarts
        << "  Hpsi_time"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << " = " << std::fixed << std::setprecision(3)
        << result.eigensolver_hamiltonian_seconds << " s"
        << "  subspace_time"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << " = "
        << result.eigensolver_subspace_seconds << " s"
        << "  ortho/Ritz/other"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << " = "
        << result.eigensolver_other_seconds << " s\n"
        << "  Davidson breakdown"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << ": initial_ortho = "
        << davidson.initial_orthonormalization_seconds
        << "  VtW = " << davidson.projected_matrix_seconds
        << "  Ritz(X/HX) = " << davidson.ritz_rotation_seconds
        << "  residual+prec = "
        << davidson.residual_preconditioner_seconds
        << "  result_copy = " << davidson.result_copy_seconds
        << "  correction_ortho = "
        << davidson.correction_orthogonalization_seconds << " s\n"
        << "  Davidson update"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << ": restart = " << davidson.restart_seconds
        << "  assemble(T) = "
        << davidson.correction_block_assembly_seconds
        << "  expand/copy = "
        << davidson.subspace_expansion_seconds
        << "  unaccounted = " << davidson_unaccounted << " s\n"
        << "  Davidson reuse"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << ": VtW full/incremental/Ritz = "
        << davidson.projected_matrix_full_builds << "/"
        << davidson.projected_matrix_incremental_updates << "/"
        << davidson.projected_matrix_ritz_reuses
        << "  correction reorth/blocks = "
        << davidson.correction_reorthogonalizations << "/"
        << davidson.correction_blocks << "\n"
        << "  SCF wall time"
        << (parallel::size() > 1 ? "(max-rank)" : "")
        << " = "
        << result.wall_time_seconds << " s\n"
        << "  phase wall time"
        << (parallel::size() > 1 ? "(max-rank)" : "")
        << ": V_in = " << result.performance.input_potential_seconds
        << "  eigensolver = " << result.performance.eigensolver_seconds
        << "  occupations = " << result.performance.occupations_seconds
        << "  density/kinetic = "
        << result.performance.density_energy_seconds
        << "  V_out/energy = "
        << result.performance.output_potential_energy_seconds
        << "  mixing = " << result.performance.mixing_seconds << " s\n"
        << "  Hpsi breakdown"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << ": vectors/blocks = "
        << result.performance.fft.hamiltonian_vectors << " / "
        << result.performance.fft.hamiltonian_block_calls
        << "  scatter = "
        << result.performance.fft.hamiltonian_scatter_seconds
        << "  FFT(back/forward) = "
        << result.performance.fft.hamiltonian_backward_fft_seconds
        << " / "
        << result.performance.fft.hamiltonian_forward_fft_seconds
        << "  V(r)*psi = "
        << result.performance.fft.hamiltonian_local_multiply_seconds
        << "  gather+T = "
        << result.performance.fft.hamiltonian_gather_kinetic_seconds
        << "  V_NL = "
        << result.performance.fft.hamiltonian_nonlocal_seconds
        << "  overhead = " << hpsi_overhead << " s\n"
        << "  density breakdown"
        << (parallel::size() > 1 ? "(rank-sum)" : "")
        << ": orbitals = "
        << result.performance.fft.density_orbitals
        << "  scatter = "
        << result.performance.fft.density_scatter_seconds
        << "  FFT(back) = "
        << result.performance.fft.density_backward_fft_seconds
        << "  accumulate = "
        << result.performance.fft.density_accumulation_seconds << " s\n";
    out.flags(flags);
    out.precision(precision);
}

} // namespace

KPointSCFResult run_kpoint_scf(
    const Lattice& lattice,
    const std::vector<KPointHamiltonian>& kpoints,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    double ion_ion_energy,
    const SCFOptions& options,
    const KPointSCFInitialGuess& initial_guess,
    std::ostream* log_stream,
    const std::vector<SpaceGroupOperation>& symmetry_operations) {

    const auto scf_start = std::chrono::steady_clock::now();
    const FFTPerformanceCounters fft_performance_start = fft.performance;
    SCFPerformanceBreakdown local_performance;
    validate_inputs(kpoints, fft, ionic_potential, options, initial_guess);
    EigensolverToleranceSchedule eigensolver_tolerances(
        options.eigensolver_initial_tolerance,
        options.eigensolver_tolerance,
        options.density_tolerance
    );
    const parallel::KPointDistribution distribution(
        static_cast<int>(kpoints.size())
    );

    const double volume = lattice.volume();
    const double dV = volume / static_cast<double>(fft.grid.ngrid);
    const int kpoint_count = static_cast<int>(kpoints.size());
    const int state_count = options.nspin * kpoint_count;
    std::vector<std::vector<double>> spin_densities =
        initial_guess.spin_densities;
    if (spin_densities.empty()) {
        const double up_electrons = options.nspin == 1
            ? options.nelec
            : 0.5 * (
                options.nelec + options.starting_magnetization
            );
        const double down_electrons = options.nspin == 1
            ? 0.0
            : options.nelec - up_electrons;
        spin_densities.assign(
            options.nspin,
            std::vector<double>(fft.grid.ngrid, 0.0)
        );
        if (!initial_guess.density.empty()) {
            std::vector<double> total_density = initial_guess.density;
            renormalize_density(
                total_density, dV, options.nelec
            );
            const double up_fraction = options.nelec > 0.0
                ? up_electrons / options.nelec
                : 1.0;
            for (int p = 0; p < fft.grid.ngrid; ++p) {
                spin_densities[0][p] =
                    up_fraction * total_density[p];
                if (options.nspin == 2) {
                    spin_densities[1][p] =
                        total_density[p] - spin_densities[0][p];
                }
            }
        } else {
            std::fill(
                spin_densities[0].begin(),
                spin_densities[0].end(),
                up_electrons / volume
            );
            if (options.nspin == 2) {
                std::fill(
                    spin_densities[1].begin(),
                    spin_densities[1].end(),
                    down_electrons / volume
                );
            }
        }
    } else {
        const std::vector<double> total_density =
            sum_spin_densities(spin_densities);
        const double current_electrons =
            electron_number_from_density(total_density, dV);
        if (std::abs(current_electrons) < 1.0e-14) {
            throw std::runtime_error(
                "The initial spin density contains no electrons."
            );
        }
        const double scale = options.nelec / current_electrons;
        for (std::vector<double>& density : spin_densities) {
            for (double& value : density) {
                value *= scale;
            }
        }
    }
    for (std::vector<double>& spin_density : spin_densities) {
        symmetrize_scalar_field(
            lattice, fft, symmetry_operations, spin_density
        );
    }

    std::vector<Eigen::MatrixXcd> orbital_guesses(state_count);
    for (int ik : distribution.local_kpoints()) {
        for (int spin = 0; spin < options.nspin; ++spin) {
            const int state =
                electronic_state_index(spin, ik, kpoint_count);
            if (!initial_guess.orbitals.empty() &&
                initial_guess.orbitals[state].size() != 0) {
                orbital_guesses[state] =
                    initial_guess.orbitals[state];
            } else {
                const int trial_count = std::min(
                    kpoints[ik].basis.size(), options.nbands + 4
                );
                orbital_guesses[state] = initial_low_kinetic_trials(
                    kpoints[ik].basis.size(), trial_count
                );
            }
        }
    }

    std::vector<PulayMixer> pulay(options.nspin);
    for (PulayMixer& mixer : pulay) {
        mixer.alpha = options.mixing_alpha;
        mixer.max_history = options.pulay_max_history;
        mixer.min_history = options.pulay_min_history;
        mixer.regularization = options.pulay_regularization;
    }
    LibXCFunctional xc(
        options.xc_functional, options.nspin
    );
    std::vector<double> weights;
    weights.reserve(state_count);
    for (int spin = 0; spin < options.nspin; ++spin) {
        for (const KPointHamiltonian& point : kpoints) {
            weights.push_back(point.weight);
        }
    }

    KPointSCFResult result;
    double previous_energy = 0.0;
    double previous_band_energy = 0.0;
    double last_signed_energy_change = 0.0;
    std::vector<std::vector<double>> previous_occupations;
    if (!options.eigensolver_full_band_accuracy &&
        options.occupation_mode == OccupationMode::Fixed) {
        previous_occupations.assign(
            state_count,
            options.fixed_occupations
        );
    }
    const std::vector<double> no_previous_occupations;
    const bool logging_enabled =
        parallel::is_root() && log_stream != nullptr
        && options.verbosity != SCFVerbosity::Silent;
    if (logging_enabled) {
        print_header(*log_stream);
    }

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        const double eigensolver_tolerance = eigensolver_tolerances.current();
        auto phase_start = std::chrono::steady_clock::now();
        const std::vector<double> density_input =
            sum_spin_densities(spin_densities);
        const auto hartree_input = build_hartree_potential(
            lattice, fft, density_input
        );
        std::vector<std::vector<double>> effective_potentials(
            options.nspin
        );
        if (options.nspin == 1) {
            const auto xc_input = xc.evaluate(
                lattice, fft, spin_densities[0], dV
            );
            effective_potentials[0] = combine_effective_potential(
                ionic_potential, hartree_input, xc_input.Vxc
            );
        } else {
            const auto xc_input = xc.evaluate_spin(
                spin_densities[0], spin_densities[1], dV
            );
            effective_potentials[0] = combine_effective_potential(
                ionic_potential, hartree_input, xc_input.Vxc_up
            );
            effective_potentials[1] = combine_effective_potential(
                ionic_potential, hartree_input, xc_input.Vxc_down
            );
        }
        local_performance.input_potential_seconds +=
            elapsed_seconds(phase_start);

        std::vector<DavidsonResult> solutions(state_count);
        int local_eigensolver_steps = 0;
        int local_hamiltonian_applications = 0;
        int local_hamiltonian_block_calls = 0;
        int local_eigensolver_restarts = 0;
        double local_hamiltonian_seconds = 0.0;
        double local_subspace_seconds = 0.0;
        double local_eigensolver_other_seconds = 0.0;
        DavidsonTimingBreakdown local_eigensolver_detail;
        double local_maximum_residual = 0.0;
        double local_residual_square_sum = 0.0;
        int local_residual_count = 0;
        int local_relaxed_solve_count = 0;
        int local_relaxed_band_failures = 0;
        double local_maximum_relaxed_residual = 0.0;
        double local_maximum_relaxed_hard_limit = 0.0;
        std::string local_eigensolver_error;

        try {
            for (int ik : distribution.local_kpoints()) {
                const KPointHamiltonian& point = kpoints[ik];
                const int maximum_subspace =
                    options.eigensolver_max_subspace > 0
                    ? std::min(
                        options.eigensolver_max_subspace,
                        point.basis.size()
                    )
                    : std::min(
                        point.basis.size(), 4 * options.nbands + 8
                    );
                const std::vector<NonlocalProjector>* projector_pointer =
                    point.projectors.empty() ? nullptr : &point.projectors;
                for (int spin = 0; spin < options.nspin; ++spin) {
                    const int state = electronic_state_index(
                        spin, ik, kpoint_count
                    );
                    const auto eigensolver_start =
                        std::chrono::steady_clock::now();
                    const std::vector<double>& state_previous_occupations =
                        previous_occupations.empty()
                        ? no_previous_occupations
                        : previous_occupations[state];
                    const std::vector<double> band_residual_tolerances =
                        davidson_band_residual_tolerances(
                            options.nbands,
                            eigensolver_tolerance,
                            state_previous_occupations,
                            options.eigensolver_empty_tolerance
                        );
                    DavidsonResult solution =
                        davidson_lowest_eigenstates(
                            point.basis,
                            fft,
                            effective_potentials[spin],
                            options.nbands,
                            orbital_guesses[state],
                            options.eigensolver_max_iterations,
                            maximum_subspace,
                            eigensolver_tolerance,
                            options.eigensolver_denom_floor,
                            projector_pointer,
                            logging_enabled
                                && distribution.size() == 1
                                && options.verbosity ==
                                    SCFVerbosity::Detailed,
                            &band_residual_tolerances
                        );
                    local_performance.eigensolver_seconds +=
                        elapsed_seconds(eigensolver_start);
                    const double residual = maximum_residual(
                        solution.residual_norms
                    );
                    const DavidsonResidualAssessment residual_assessment =
                        assess_davidson_residuals(
                            solution.residual_norms,
                            band_residual_tolerances,
                            eigensolver_tolerance
                        );
                    if (!solution.converged &&
                        !residual_assessment.acceptable) {
                        std::ostringstream message;
                        message
                            << "Davidson eigensolver did not converge at "
                            << "spin " << spin << ", k point " << ik
                            << " ("
                            << point.fractional_position.transpose()
                            << "): max residual = " << residual
                            << ", requested tolerance = "
                            << eigensolver_tolerance
                            << ", worst band = "
                            << solution.max_residual_band
                            << ", subspace = "
                            << solution.final_subspace_size
                            << ", Hpsi applications = "
                            << solution.hamiltonian_applications
                            << ", strict bands converged = "
                            << (residual_assessment.strict_bands_converged
                                ? "yes" : "no")
                            << ", worst strict band = "
                            << residual_assessment.worst_strict_band
                            << ", strict-band residual = "
                            << residual_assessment
                                .maximum_strict_residual
                            << ", relaxed-band failures = "
                            << residual_assessment.relaxed_band_failures;
                        throw std::runtime_error(message.str());
                    }
                    if (!solution.converged) {
                        ++local_relaxed_solve_count;
                        local_relaxed_band_failures +=
                            residual_assessment.relaxed_band_failures;
                        local_maximum_relaxed_residual = std::max(
                            local_maximum_relaxed_residual,
                            residual_assessment.maximum_relaxed_residual
                        );
                        local_maximum_relaxed_hard_limit = std::max(
                            local_maximum_relaxed_hard_limit,
                            residual_assessment
                                .maximum_accepted_relaxed_residual
                        );
                    }

                    local_eigensolver_steps += solution.iterations;
                    local_hamiltonian_applications +=
                        solution.hamiltonian_applications;
                    local_hamiltonian_block_calls +=
                        solution.hamiltonian_block_calls;
                    local_eigensolver_restarts +=
                        solution.subspace_restarts;
                    local_hamiltonian_seconds +=
                        solution.hamiltonian_seconds;
                    local_subspace_seconds +=
                        solution.subspace_diagonalization_seconds;
                    local_eigensolver_other_seconds += std::max(
                        0.0,
                        solution.total_seconds
                            - solution.hamiltonian_seconds
                            - solution.subspace_diagonalization_seconds
                    );
                    local_eigensolver_detail.accumulate(
                        solution.timing
                    );
                    local_maximum_residual = std::max(
                        local_maximum_residual, residual
                    );
                    for (double band_residual :
                         solution.residual_norms) {
                        local_residual_square_sum +=
                            band_residual * band_residual;
                        ++local_residual_count;
                    }
                    solutions[state] = std::move(solution);
                }
            }
        } catch (const std::exception& error) {
            std::ostringstream message;
            message << "k-point solve failed on MPI rank "
                << distribution.rank() << ": " << error.what();
            local_eigensolver_error = message.str();
        } catch (...) {
            local_eigensolver_error =
                "k-point solve failed with an unknown exception on MPI rank "
                + std::to_string(distribution.rank()) + ".";
        }
        const std::string eigensolver_error =
            parallel::first_error(local_eigensolver_error);
        if (!eigensolver_error.empty()) {
            throw std::runtime_error(eigensolver_error);
        }

        const int iteration_eigensolver_steps =
            parallel::sum(local_eigensolver_steps);
        const int iteration_hamiltonian_applications =
            parallel::sum(local_hamiltonian_applications);
        const int iteration_hamiltonian_block_calls =
            parallel::sum(local_hamiltonian_block_calls);
        const int iteration_eigensolver_restarts =
            parallel::sum(local_eigensolver_restarts);
        const double iteration_hamiltonian_seconds =
            parallel::sum(local_hamiltonian_seconds);
        const double iteration_subspace_seconds =
            parallel::sum(local_subspace_seconds);
        const double iteration_eigensolver_other_seconds =
            parallel::sum(local_eigensolver_other_seconds);
        const DavidsonTimingBreakdown iteration_eigensolver_detail =
            sum_davidson_timing(local_eigensolver_detail);
        const double iteration_maximum_residual =
            parallel::maximum(local_maximum_residual);
        const double iteration_residual_square_sum =
            parallel::sum(local_residual_square_sum);
        const int iteration_residual_count =
            parallel::sum(local_residual_count);
        const int iteration_relaxed_solve_count =
            parallel::sum(local_relaxed_solve_count);
        const int iteration_relaxed_band_failures =
            parallel::sum(local_relaxed_band_failures);
        const double iteration_maximum_relaxed_residual =
            parallel::maximum(local_maximum_relaxed_residual);
        const double iteration_maximum_relaxed_hard_limit =
            parallel::maximum(local_maximum_relaxed_hard_limit);
        const double iteration_residual_rms =
            iteration_residual_count > 0
            ? std::sqrt(
                iteration_residual_square_sum /
                static_cast<double>(iteration_residual_count)
            )
            : 0.0;
        if (logging_enabled && iteration_relaxed_solve_count > 0) {
            *log_stream
                << "WARNING: accepted "
                << iteration_relaxed_band_failures
                << " effectively empty Davidson band failure(s) across "
                << iteration_relaxed_solve_count
                << " k-point solve(s); maximum residual = "
                << iteration_maximum_relaxed_residual
                << ", hard limit = "
                << iteration_maximum_relaxed_hard_limit
                << ".\n";
        }

        result.eigensolver_hamiltonian_applications +=
            iteration_hamiltonian_applications;
        result.eigensolver_hamiltonian_block_calls +=
            iteration_hamiltonian_block_calls;
        result.eigensolver_iterations += iteration_eigensolver_steps;
        result.eigensolver_restarts += iteration_eigensolver_restarts;
        result.eigensolver_hamiltonian_seconds +=
            iteration_hamiltonian_seconds;
        result.eigensolver_subspace_seconds += iteration_subspace_seconds;
        result.eigensolver_other_seconds +=
            iteration_eigensolver_other_seconds;
        result.eigensolver_detail.accumulate(
            iteration_eigensolver_detail
        );

        phase_start = std::chrono::steady_clock::now();
        std::vector<double> packed_eigenvalues(
            static_cast<std::size_t>(state_count) * options.nbands,
            0.0
        );
        for (int ik : distribution.local_kpoints()) {
            for (int spin = 0; spin < options.nspin; ++spin) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                for (int ib = 0; ib < options.nbands; ++ib) {
                    packed_eigenvalues[
                        static_cast<std::size_t>(state) *
                            options.nbands + ib
                    ] = solutions[state].eigenvalues[ib];
                }
            }
        }
        parallel::sum_in_place(packed_eigenvalues);

        std::vector<Eigen::VectorXd> eigenvalues(state_count);
        for (int state = 0; state < state_count; ++state) {
            eigenvalues[state].resize(options.nbands);
            for (int ib = 0; ib < options.nbands; ++ib) {
                eigenvalues[state][ib] = packed_eigenvalues[
                    static_cast<std::size_t>(state) *
                        options.nbands + ib
                ];
            }
        }

        const KPointOccupationResult occupations =
            compute_kpoint_occupations(
                eigenvalues,
                weights,
                options.nelec,
                options.occupation_mode,
                options.fixed_occupations,
                options.smearing_sigma,
                options.degeneracy_tolerance,
                options.nspin == 1 ? 2.0 : 1.0,
                static_cast<double>(options.nspin)
            );
        local_performance.occupations_seconds +=
            elapsed_seconds(phase_start);
        bool occupation_refinement_needed = false;
        if (!previous_occupations.empty()) {
            for (int state = 0; state < state_count; ++state) {
                occupation_refinement_needed =
                    occupation_refinement_needed ||
                    davidson_occupation_refinement_needed(
                        previous_occupations[state],
                        occupations.occupations[state]
                    );
            }
        }

        phase_start = std::chrono::steady_clock::now();
        std::vector<std::vector<double>> spin_density_output(
            options.nspin,
            std::vector<double>(fft.grid.ngrid, 0.0)
        );
        double local_kinetic_energy = 0.0;
        double local_nonlocal_energy = 0.0;
        std::vector<KPointElectronicState> electronic_states(
            state_count
        );
        for (int spin = 0; spin < options.nspin; ++spin) {
            for (int ik = 0; ik < kpoint_count; ++ik) {
                const int state_index = electronic_state_index(
                    spin, ik, kpoint_count
                );
                const KPointHamiltonian& point = kpoints[ik];
                KPointElectronicState& state =
                    electronic_states[state_index];
                state.spin_channel = spin;
                state.kpoint_index = ik;
                state.fractional_position =
                    point.fractional_position;
                state.weight = point.weight;
                state.owner_rank = distribution.owner(ik);
                state.eigenvalues = eigenvalues[state_index];
                state.occupations =
                    occupations.occupations[state_index];
            }
        }

        std::string local_density_error;
        try {
            for (int ik : distribution.local_kpoints()) {
                const KPointHamiltonian& point = kpoints[ik];
                for (int spin = 0; spin < options.nspin; ++spin) {
                    const int state = electronic_state_index(
                        spin, ik, kpoint_count
                    );
                    Eigen::MatrixXcd orbitals =
                        solutions[state].eigenvectors.leftCols(
                            options.nbands
                        );
                    const auto density_at_k =
                        build_density_from_orbitals(
                            point.basis,
                            fft,
                            orbitals,
                            occupations.occupations[state],
                            volume
                        );
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && \
       fft.grid.ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
                    for (int grid_index = 0;
                         grid_index < fft.grid.ngrid;
                         ++grid_index) {
                        spin_density_output[spin][grid_index] +=
                            point.weight *
                            density_at_k[grid_index];
                    }
                    local_kinetic_energy +=
                        point.weight * compute_kinetic_energy(
                            point.basis,
                            orbitals,
                            occupations.occupations[state]
                        );
                    local_nonlocal_energy +=
                        point.weight * compute_nonlocal_energy(
                            point.projectors,
                            orbitals,
                            occupations.occupations[state]
                        );
                    electronic_states[state].orbitals =
                        std::move(orbitals);
                }
            }
        } catch (const std::exception& error) {
            std::ostringstream message;
            message << "k-point density/energy assembly failed on MPI rank "
                << distribution.rank() << ": " << error.what();
            local_density_error = message.str();
        } catch (...) {
            local_density_error =
                "k-point density/energy assembly failed with an unknown "
                "exception on MPI rank "
                + std::to_string(distribution.rank()) + ".";
        }
        const std::string density_error =
            parallel::first_error(local_density_error);
        if (!density_error.empty()) {
            throw std::runtime_error(density_error);
        }

        for (std::vector<double>& density : spin_density_output) {
            parallel::sum_in_place(density);
            symmetrize_scalar_field(
                lattice, fft, symmetry_operations, density
            );
        }
        const std::vector<double> density_output =
            sum_spin_densities(spin_density_output);
        const double kinetic_energy = parallel::sum(local_kinetic_energy);
        const double nonlocal_energy = parallel::sum(local_nonlocal_energy);
        local_performance.density_energy_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const auto hartree_output = build_hartree_potential(
            lattice, fft, density_output
        );
        double exchange_energy = 0.0;
        double correlation_energy = 0.0;
        if (options.nspin == 1) {
            const auto xc_output = xc.evaluate(
                lattice, fft, spin_density_output[0], dV
            );
            exchange_energy = xc_output.exchange_energy;
            correlation_energy = xc_output.correlation_energy;
        } else {
            const auto xc_output = xc.evaluate_spin(
                spin_density_output[0],
                spin_density_output[1],
                dV
            );
            exchange_energy = xc_output.exchange_energy;
            correlation_energy = xc_output.correlation_energy;
        }
        EnergyTerms energy;
        energy.kinetic = kinetic_energy;
        double external_energy = 0.0;
        double hartree_energy = 0.0;
#pragma omp parallel for schedule(static) \
    reduction(+:external_energy, hartree_energy) \
    if(fft.thread_count > 1 && \
       fft.grid.ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
        for (int grid_index = 0;
             grid_index < fft.grid.ngrid;
             ++grid_index) {
            external_energy += dV * density_output[grid_index]
                * ionic_potential[grid_index];
            hartree_energy += 0.5 * dV * density_output[grid_index]
                * hartree_output[grid_index];
        }
        energy.external = external_energy;
        energy.hartree = hartree_energy;
        energy.exchange = exchange_energy;
        energy.correlation = correlation_energy;
        energy.nonlocal = nonlocal_energy;
        energy.total = energy.kinetic + energy.external + energy.hartree
            + energy.exchange + energy.correlation + energy.nonlocal;
        energy.electronic_entropy = occupations.entropy;
        energy.entropy_correction = options.occupation_mode
                == OccupationMode::FermiDirac
            ? options.smearing_sigma * occupations.entropy
            : 0.0;
        energy.free_energy = energy.total - energy.entropy_correction;
        energy.sigma0_estimate = 0.5 * (energy.total + energy.free_energy);
        energy.ion_smooth = ion_ion_energy;
        energy.total_with_ion_smooth = energy.total + ion_ion_energy;

        const bool finite_temperature =
            options.occupation_mode == OccupationMode::FermiDirac;
        const double energy_for_convergence = finite_temperature
            ? energy.free_energy
            : energy.total;
        const double signed_energy_change = iteration == 0
            ? 0.0
            : energy_for_convergence - previous_energy;
        const double energy_change = std::abs(signed_energy_change);
        double band_energy = 0.0;
        for (int state = 0; state < state_count; ++state) {
            for (int ib = 0; ib < options.nbands; ++ib) {
                band_energy += weights[state]
                    * occupations.occupations[state][ib]
                    * eigenvalues[state][ib];
            }
        }
        const double signed_band_energy_change = iteration == 0
            ? 0.0
            : band_energy - previous_band_energy;
        const double band_energy_change =
            std::abs(signed_band_energy_change);
        local_performance.output_potential_energy_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const double density_residual = spin_density_residual(
            spin_densities, spin_density_output, dV
        );
        std::array<double, 2> spin_electron_counts{{0.0, 0.0}};
        for (int spin = 0; spin < options.nspin; ++spin) {
            for (int ik = 0; ik < kpoint_count; ++ik) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                for (double occupation :
                     occupations.occupations[state]) {
                    spin_electron_counts[spin] +=
                        kpoints[ik].weight * occupation;
                }
            }
            pulay_mix_density(
                pulay[spin],
                spin_densities[spin],
                spin_density_output[spin],
                dV,
                spin_electron_counts[spin]
            );
        }
        local_performance.mixing_seconds += elapsed_seconds(phase_start);

        result.iterations = iteration + 1;
        result.final_density_residual = density_residual;
        result.final_energy_change = energy_change;
        result.final_band_energy_change = band_energy_change;
        result.final_eigensolver_tolerance = eigensolver_tolerance;
        result.kpoints = std::move(electronic_states);
        result.occupations = occupations;
        result.density = density_output;
        result.spin_densities = spin_density_output;
        result.spin_electron_counts = spin_electron_counts;
        result.magnetization = options.nspin == 2
            ? spin_electron_counts[0] - spin_electron_counts[1]
            : 0.0;
        result.energy = energy;
        result.electron_number_from_density =
            electron_number_from_density(density_output, dV);
        result.variational_energy = energy_for_convergence + ion_ion_energy;
        last_signed_energy_change = signed_energy_change;

        if (logging_enabled) {
            print_iteration(
                *log_stream,
                iteration,
                result.variational_energy,
                signed_energy_change,
                signed_band_energy_change,
                density_residual,
                iteration_hamiltonian_applications,
                iteration_residual_rms,
                true
            );
            if (options.verbosity == SCFVerbosity::Detailed) {
                *log_stream << "       mu = " << occupations.mu
                    << "  weighted Ne = " << occupations.nelec_sum
                    << "  eig_tol = " << eigensolver_tolerance
                    << "  max residual = "
                    << iteration_maximum_residual;
                if (options.nspin == 2) {
                    *log_stream << "  N(up/down) = "
                        << spin_electron_counts[0] << " / "
                        << spin_electron_counts[1]
                        << "  M = " << result.magnetization;
                }
                *log_stream << "\n";
                const bool print_bands =
                    iteration < options.band_print_interval
                    || (options.band_print_interval > 0 &&
                        iteration % options.band_print_interval == 0);
                if (print_bands) {
                    const int bands_to_print = std::min(
                        options.bands_to_print, options.nbands
                    );
                    for (int spin = 0; spin < options.nspin; ++spin) {
                        for (int ik = 0; ik < kpoint_count; ++ik) {
                            const int state = electronic_state_index(
                                spin, ik, kpoint_count
                            );
                            *log_stream << "       spin " << spin
                                << "  k[" << ik << "] = "
                                << kpoints[ik].fractional_position.transpose()
                                << "  w = " << kpoints[ik].weight << "\n";
                            for (int ib = 0; ib < bands_to_print; ++ib) {
                                *log_stream << "         band " << ib
                                    << "  eps = "
                                    << eigenvalues[state][ib]
                                    << "  occ = "
                                    << occupations.occupations[state][ib]
                                    << "\n";
                            }
                        }
                    }
                }
            }
        }

        const bool energy_changes_converged = iteration > 0 &&
            scf_energy_changes_converged(
                signed_energy_change,
                signed_band_energy_change,
                options.energy_tolerance
            );
        const bool outer_converged =
            energy_changes_converged &&
            !occupation_refinement_needed;
        if (energy_changes_converged &&
            occupation_refinement_needed &&
            logging_enabled) {
            *log_stream
                << "       A previously empty band became occupied; "
                << "requesting one strict Davidson refinement.\n";
        }
        if (outer_converged &&
            eigensolver_tolerances.at_final_tolerance()) {
            result.converged = true;
            break;
        }
        if (outer_converged) {
            eigensolver_tolerances.force_final_tolerance();
            if (logging_enabled &&
                options.verbosity == SCFVerbosity::Detailed) {
                *log_stream
                    << "       Outer SCF criteria reached; requesting final "
                    << "Davidson refinement.\n";
            }
        } else {
            eigensolver_tolerances.advance(density_residual);
        }

        previous_energy = energy_for_convergence;
        previous_band_energy = band_energy;
        if (!options.eigensolver_full_band_accuracy) {
            previous_occupations = occupations.occupations;
        }
        for (int ik : distribution.local_kpoints()) {
            for (int spin = 0; spin < options.nspin; ++spin) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                orbital_guesses[state] =
                    result.kpoints[state].orbitals;
            }
        }
    }

    local_performance.fft =
        fft.performance.delta_from(fft_performance_start);
    result.performance = reduce_performance(local_performance);
    const double local_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scf_start
    ).count();
    result.wall_time_seconds = parallel::maximum(local_wall_time);
    if (logging_enabled) {
        print_summary(*log_stream, result, last_signed_energy_change);
    }
    return result;
}
