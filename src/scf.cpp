#include "scf.hpp"

#include "eigensolver.hpp"
#include "mixing.hpp"
#include "potentials.hpp"
#include "scf_convergence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace {

void validate_scf_inputs(
    const PlaneWaveBasis3D& basis,
    const FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    const SCFOptions& options,
    const SCFInitialGuess& initial_guess) {

    if (options.nelec < 0.0) {
        throw std::runtime_error("SCF electron number cannot be negative.");
    }
    if (options.nbands <= 0 || options.nbands > basis.size()) {
        throw std::runtime_error("SCF band count must be in [1, basis size].");
    }
    if (options.nspin != 1) {
        throw std::runtime_error(
            "The legacy Gamma-only SCF entry point supports only nspin = 1; "
            "use run_kpoint_scf() for spin polarization."
        );
    }
    if (static_cast<int>(ionic_potential.size()) != fft.grid.ngrid) {
        throw std::runtime_error("Ionic potential size does not match FFT grid.");
    }
    if (options.max_iterations <= 0 ||
        options.eigensolver_max_iterations <= 0) {
        throw std::runtime_error("SCF and eigensolver iteration limits must be positive.");
    }
    if (options.occupation_mode == OccupationMode::FermiDirac &&
        options.smearing_sigma <= 0.0) {
        throw std::runtime_error("Fermi-Dirac occupations require positive smearing sigma.");
    }
    if (options.occupation_mode == OccupationMode::Fixed &&
        static_cast<int>(options.fixed_occupations.size()) != options.nbands) {
        throw std::runtime_error("Fixed occupations must contain one value per band.");
    }
    if (!initial_guess.density.empty() &&
        static_cast<int>(initial_guess.density.size()) != fft.grid.ngrid) {
        throw std::runtime_error("Initial density size does not match FFT grid.");
    }
    if (initial_guess.orbitals.size() != 0 &&
        (initial_guess.orbitals.rows() != basis.size() ||
         initial_guess.orbitals.cols() < options.nbands)) {
        throw std::runtime_error(
            "Initial orbitals need basis.size() rows and at least nbands columns."
        );
    }
}

double maximum_residual(const std::vector<double>& residuals) {
    double result = 0.0;
    for (double residual : residuals) {
        result = std::max(result, residual);
    }
    return result;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

void print_compact_header(std::ostream& out) {
    out << "\n"
        << "       N       E                     dE"
        << "             d eps       ncg     rms"
        << "          rms(c)\n";
}

void print_compact_iteration(
    std::ostream& out,
    int iteration,
    const SCFResult& result,
    double signed_dE,
    double signed_band_energy_change,
    double drho,
    int hamiltonian_applications,
    double eigensolver_residual) {

    const auto old_flags = out.flags();
    const auto old_precision = out.precision();

    out << "DAV: " << std::setw(4) << iteration + 1
        << "  " << std::scientific << std::setprecision(12)
        << std::setw(20) << result.variational_energy
        << "  " << std::setprecision(4) << std::setw(12) << signed_dE
        << "  " << std::setw(12) << signed_band_energy_change
        << "  " << std::setw(7) << hamiltonian_applications
        << "  " << std::setw(12) << eigensolver_residual
        << "  " << std::setw(12) << drho
        << "\n";

    out.flags(old_flags);
    out.precision(old_precision);
}

void print_detailed_iteration(
    std::ostream& out,
    int iteration,
    const SCFOptions& options,
    const SCFResult& result,
    double dE,
    double drho,
    int pulay_history_size,
    double eigensolver_tolerance) {

    out << "E_NL = " << result.energy.nonlocal << "\n";
    out << "E_electronic = " << result.energy.total
        << "  E_ion_smooth = " << result.energy.ion_smooth
        << "  E_total_smooth = " << result.energy.total_with_ion_smooth
        << "\n";
    out << "SCF iter "
        << std::setw(3) << iteration + 1
        << "  E = " << std::setw(20) << result.energy.total
        << "  F = " << std::setw(20) << result.energy.free_energy
        << "  E0est = " << std::setw(20) << result.energy.sigma0_estimate
        << "  sigmaS = " << std::setw(12) << result.energy.entropy_correction
        << "  dE = " << std::setw(12) << dE
        << "  drho = " << std::setw(12) << drho
        << "  eig_tol = " << eigensolver_tolerance
        << "  pulay_hist = " << pulay_history_size
        << "  Ne_occ = " << result.occupations.nelec_sum
        << "  Ne_out = " << result.electron_number_from_density
        << "  mu = " << result.occupations.mu
        << "  eps0 = " << result.eigenvalues[0]
        << "\n";

    const bool print_bands =
        iteration < options.band_print_interval ||
        (options.band_print_interval > 0 &&
         iteration % options.band_print_interval == 0);
    if (print_bands) {
        const int nprint = std::min(options.bands_to_print, options.nbands);
        for (int ib = 0; ib < nprint; ++ib) {
            out << "  band " << ib
                << "  eps = " << std::setw(20) << result.eigenvalues[ib]
                << "  occ = " << result.occupations.occ[ib]
                << "\n";
        }
    }
}

void print_compact_summary(
    std::ostream& out,
    const SCFResult& result,
    double signed_dE) {

    const auto old_flags = out.flags();
    const auto old_precision = out.precision();
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
        << "  d E = " << std::setprecision(4) << std::setw(12) << signed_dE
        << "\n"
        << "  eigensolver work: N_Hpsi = "
        << result.eigensolver_hamiltonian_applications
        << "  N_Hblock = "
        << result.eigensolver_hamiltonian_block_calls
        << "  N_iter = " << result.eigensolver_iterations
        << "  restarts = " << result.eigensolver_restarts
        << "  Hpsi_time = " << std::fixed << std::setprecision(3)
        << result.eigensolver_hamiltonian_seconds << " s"
        << "  subspace_time = "
        << result.eigensolver_subspace_seconds << " s"
        << "  ortho/Ritz/other = "
        << result.eigensolver_other_seconds << " s\n"
        << "  Davidson breakdown: initial_ortho = "
        << davidson.initial_orthonormalization_seconds
        << "  VtW = " << davidson.projected_matrix_seconds
        << "  Ritz(X/HX) = " << davidson.ritz_rotation_seconds
        << "  residual+prec = "
        << davidson.residual_preconditioner_seconds
        << "  result_copy = " << davidson.result_copy_seconds
        << "  correction_ortho = "
        << davidson.correction_orthogonalization_seconds << " s\n"
        << "  Davidson update: restart = "
        << davidson.restart_seconds
        << "  assemble(T) = "
        << davidson.correction_block_assembly_seconds
        << "  expand/copy = "
        << davidson.subspace_expansion_seconds
        << "  unaccounted = " << davidson_unaccounted << " s\n"
        << "  SCF wall time = " << result.wall_time_seconds << " s\n"
        << "  phase wall time: V_in = "
        << result.performance.input_potential_seconds
        << "  eigensolver = " << result.performance.eigensolver_seconds
        << "  occupations = " << result.performance.occupations_seconds
        << "  density/kinetic = "
        << result.performance.density_energy_seconds
        << "  V_out/energy = "
        << result.performance.output_potential_energy_seconds
        << "  mixing = " << result.performance.mixing_seconds << " s\n"
        << "  Hpsi breakdown: vectors/blocks = "
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
        << "  density breakdown: orbitals = "
        << result.performance.fft.density_orbitals
        << "  scatter = "
        << result.performance.fft.density_scatter_seconds
        << "  FFT(back) = "
        << result.performance.fft.density_backward_fft_seconds
        << "  accumulate = "
        << result.performance.fft.density_accumulation_seconds << " s\n";

    out.flags(old_flags);
    out.precision(old_precision);
}

} // namespace

SCFResult run_scf(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    double ion_ion_energy,
    const std::vector<NonlocalProjector>& projectors,
    const SCFOptions& options,
    const SCFInitialGuess& initial_guess,
    std::ostream* log_stream) {

    const auto scf_start = std::chrono::steady_clock::now();
    const FFTPerformanceCounters fft_performance_start = fft.performance;

    validate_scf_inputs(
        basis,
        fft,
        ionic_potential,
        options,
        initial_guess
    );
    EigensolverToleranceSchedule eigensolver_tolerances(
        options.eigensolver_initial_tolerance,
        options.eigensolver_tolerance,
        options.density_tolerance
    );

    const double volume = lattice.volume();
    const double dV = volume / static_cast<double>(fft.grid.ngrid);

    std::vector<double> rho = initial_guess.density;
    if (rho.empty()) {
        rho.assign(fft.grid.ngrid, options.nelec / volume);
    } else {
        renormalize_density(rho, dV, options.nelec);
    }

    Eigen::MatrixXcd C_guess = initial_guess.orbitals;
    if (C_guess.size() == 0) {
        const int ntrial = std::min(
            basis.size(),
            options.nbands + 4
        );
        C_guess = initial_low_kinetic_trials(basis.size(), ntrial);
    }

    PulayMixer pulay;
    pulay.alpha = options.mixing_alpha;
    pulay.max_history = options.pulay_max_history;
    pulay.min_history = options.pulay_min_history;
    pulay.regularization = options.pulay_regularization;

    const int max_subspace = options.eigensolver_max_subspace > 0
        ? options.eigensolver_max_subspace
        : std::min(basis.size(), 4 * options.nbands + 8);

    const std::vector<NonlocalProjector>* projector_ptr =
        projectors.empty() ? nullptr : &projectors;
    LibXCLDAFunctional xc(options.lda_functional);

    SCFResult result;
    double previous_energy = 0.0;
    double previous_band_energy = 0.0;
    double last_signed_energy_change = 0.0;
    const bool logging_enabled =
        log_stream != nullptr && options.verbosity != SCFVerbosity::Silent;

    if (logging_enabled && options.verbosity == SCFVerbosity::Compact) {
        print_compact_header(*log_stream);
    }

    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const double eigensolver_tolerance = eigensolver_tolerances.current();
        auto phase_start = std::chrono::steady_clock::now();
        const auto VH = build_hartree_potential(lattice, fft, rho);
        const auto xc_input = xc.evaluate(rho, dV);
        const auto Veff = combine_effective_potential(
            ionic_potential,
            VH,
            xc_input.Vxc
        );
        result.performance.input_potential_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const DavidsonResult ks = davidson_lowest_eigenstates(
            basis,
            fft,
            Veff,
            options.nbands,
            C_guess,
            options.eigensolver_max_iterations,
            max_subspace,
            eigensolver_tolerance,
            options.eigensolver_denom_floor,
            projector_ptr,
            logging_enabled && options.verbosity == SCFVerbosity::Detailed
        );
        result.performance.eigensolver_seconds +=
            elapsed_seconds(phase_start);
        if (!ks.converged) {
            const double maximum_ks_residual =
                maximum_residual(ks.residual_norms);
            std::ostringstream message;
            message
                << "Davidson eigensolver did not converge during SCF: "
                << "max residual = " << maximum_ks_residual
                << ", requested tolerance = "
                << eigensolver_tolerance
                << ", worst band = " << ks.max_residual_band
                << ", subspace = " << ks.final_subspace_size
                << ", Hpsi applications = "
                << ks.hamiltonian_applications
                << ", projected Hermiticity error = "
                << ks.projected_hermiticity_error;
            if (ks.stagnated) {
                message << ", stopped after repeated stagnation restarts";
            }
            throw std::runtime_error(message.str());
        }

        result.eigensolver_hamiltonian_applications +=
            ks.hamiltonian_applications;
        result.eigensolver_hamiltonian_block_calls +=
            ks.hamiltonian_block_calls;
        result.eigensolver_iterations += ks.iterations;
        result.eigensolver_restarts += ks.subspace_restarts;
        result.eigensolver_hamiltonian_seconds +=
            ks.hamiltonian_seconds;
        result.eigensolver_subspace_seconds +=
            ks.subspace_diagonalization_seconds;
        result.eigensolver_other_seconds += std::max(
            0.0,
            ks.total_seconds - ks.hamiltonian_seconds
                - ks.subspace_diagonalization_seconds
        );
        result.eigensolver_detail.accumulate(ks.timing);

        phase_start = std::chrono::steady_clock::now();
        const OccupationResult occupations = compute_occupations(
            ks.eigenvalues,
            options.nelec,
            options.occupation_mode,
            options.fixed_occupations,
            options.smearing_sigma,
            options.degeneracy_tolerance
        );
        result.performance.occupations_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const Eigen::MatrixXcd orbitals =
            ks.eigenvectors.leftCols(options.nbands);
        const auto rho_out = build_density_from_orbitals(
            basis,
            fft,
            orbitals,
            occupations.occ,
            volume
        );

        const double nonlocal_energy = compute_nonlocal_energy(
            projectors,
            orbitals,
            occupations.occ
        );
        result.performance.density_energy_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const auto VH_out = build_hartree_potential(lattice, fft, rho_out);
        const auto xc_output = xc.evaluate(rho_out, dV);
        const bool finite_temperature =
            options.occupation_mode == OccupationMode::FermiDirac;
        EnergyTerms energy = compute_total_energy(
            basis,
            orbitals,
            occupations.occ,
            rho_out,
            ionic_potential,
            VH_out,
            xc_output.exchange_energy,
            xc_output.correlation_energy,
            dV,
            finite_temperature ? occupations.entropy : 0.0,
            finite_temperature ? options.smearing_sigma : 0.0,
            nonlocal_energy
        );
        energy.ion_smooth = ion_ion_energy;
        energy.total_with_ion_smooth = energy.total + ion_ion_energy;

        const double energy_for_convergence = finite_temperature
            ? energy.free_energy
            : energy.total;
        const double signed_dE = iter == 0
            ? 0.0
            : energy_for_convergence - previous_energy;
        const double dE = std::abs(signed_dE);
        double band_energy = 0.0;
        for (int ib = 0; ib < options.nbands; ++ib) {
            band_energy +=
                occupations.occ[ib] * ks.eigenvalues[ib];
        }
        const double signed_band_energy_change = iter == 0
            ? 0.0
            : band_energy - previous_band_energy;
        const double band_energy_change =
            std::abs(signed_band_energy_change);
        result.performance.output_potential_energy_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        const double drho = pulay_mix_density(
            pulay,
            rho,
            rho_out,
            dV,
            options.nelec
        );
        result.performance.mixing_seconds += elapsed_seconds(phase_start);

        result.iterations = iter + 1;
        result.final_density_residual = drho;
        result.final_energy_change = dE;
        result.final_band_energy_change = band_energy_change;
        result.final_eigensolver_tolerance = eigensolver_tolerance;
        result.eigenvalues = ks.eigenvalues;
        result.orbitals = orbitals;
        result.occupations = occupations;
        result.density = rho_out;
        result.energy = energy;
        result.electron_number_from_density =
            electron_number_from_density(rho_out, dV);
        result.variational_energy = energy_for_convergence + ion_ion_energy;
        last_signed_energy_change = signed_dE;

        if (logging_enabled && options.verbosity == SCFVerbosity::Compact) {
            print_compact_iteration(
                *log_stream,
                iter,
                result,
                signed_dE,
                signed_band_energy_change,
                drho,
                ks.hamiltonian_applications,
                std::sqrt(
                    std::inner_product(
                        ks.residual_norms.begin(),
                        ks.residual_norms.end(),
                        ks.residual_norms.begin(),
                        0.0
                    ) /
                    static_cast<double>(ks.residual_norms.size())
                )
            );
        } else if (logging_enabled &&
                   options.verbosity == SCFVerbosity::Detailed) {
            print_detailed_iteration(
                *log_stream,
                iter,
                options,
                result,
                dE,
                drho,
                pulay.history_size(),
                eigensolver_tolerance
            );
        }

        const bool outer_converged = iter > 0 &&
            scf_energy_changes_converged(
                signed_dE,
                signed_band_energy_change,
                options.energy_tolerance
            );
        if (outer_converged &&
            eigensolver_tolerances.at_final_tolerance()) {
            result.converged = true;
            if (logging_enabled &&
                options.verbosity == SCFVerbosity::Detailed) {
                *log_stream << "SCF converged.\n";
            }
            break;
        }
        if (outer_converged) {
            eigensolver_tolerances.force_final_tolerance();
            if (logging_enabled &&
                options.verbosity == SCFVerbosity::Detailed) {
                *log_stream
                    << "Outer SCF criteria reached; requesting final "
                    << "Davidson refinement.\n";
            }
        } else {
            eigensolver_tolerances.advance(drho);
        }

        previous_energy = energy_for_convergence;
        previous_band_energy = band_energy;
        C_guess = orbitals;
    }

    result.performance.fft =
        fft.performance.delta_from(fft_performance_start);
    result.wall_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scf_start
    ).count();

    if (logging_enabled && options.verbosity == SCFVerbosity::Compact) {
        print_compact_summary(*log_stream, result, last_signed_energy_change);
    }

    return result;
}
