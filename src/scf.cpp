#include "scf.hpp"

#include "eigensolver.hpp"
#include "mixing.hpp"
#include "potentials.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
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

void print_compact_header(std::ostream& out) {
    out << "\n"
        << "-------------------------------- electronic minimization "
        << "--------------------------------\n"
        << "          N                     E              dE"
        << "        rms(rho)  Niter  N_Hpsi      rms(eig)\n";
}

void print_compact_iteration(
    std::ostream& out,
    int iteration,
    const SCFResult& result,
    double signed_dE,
    double drho,
    int davidson_iterations,
    int hamiltonian_applications,
    double eigensolver_residual) {

    const auto old_flags = out.flags();
    const auto old_precision = out.precision();

    out << "DAV: " << std::setw(4) << iteration + 1
        << "  " << std::scientific << std::setprecision(12)
        << std::setw(20) << result.variational_energy
        << "  " << std::setprecision(4) << std::setw(12) << signed_dE
        << "  " << std::setw(12) << drho
        << "  " << std::setw(5) << davidson_iterations
        << "  " << std::setw(7) << hamiltonian_applications
        << "  " << std::setw(12) << eigensolver_residual
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
    int pulay_history_size) {

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

    out << "--------------------------------------------------------------------------------\n"
        << "  1 F= " << std::scientific << std::setprecision(12)
        << std::setw(20) << free_energy_with_ions
        << " E0= " << std::setw(20) << sigma0_with_ions
        << "  d E = " << std::setprecision(4) << std::setw(12) << signed_dE
        << "\n"
        << "  eigensolver work: N_Hpsi = "
        << result.eigensolver_hamiltonian_applications
        << "  N_iter = " << result.eigensolver_iterations
        << "  restarts = " << result.eigensolver_restarts
        << "  Hpsi_time = " << std::fixed << std::setprecision(3)
        << result.eigensolver_hamiltonian_seconds << " s"
        << "  subspace_time = "
        << result.eigensolver_subspace_seconds << " s\n"
        << "  SCF wall time = " << result.wall_time_seconds << " s\n";

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

    validate_scf_inputs(
        basis,
        fft,
        ionic_potential,
        options,
        initial_guess
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
    double last_signed_energy_change = 0.0;
    const bool logging_enabled =
        log_stream != nullptr && options.verbosity != SCFVerbosity::Silent;

    if (logging_enabled && options.verbosity == SCFVerbosity::Compact) {
        print_compact_header(*log_stream);
    }

    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const auto VH = build_hartree_potential(lattice, fft, rho);
        const auto xc_input = xc.evaluate(rho, dV);
        const auto Veff = combine_effective_potential(
            ionic_potential,
            VH,
            xc_input.Vxc
        );

        const DavidsonResult ks = davidson_lowest_eigenstates(
            basis,
            fft,
            Veff,
            options.nbands,
            C_guess,
            options.eigensolver_max_iterations,
            max_subspace,
            options.eigensolver_tolerance,
            options.eigensolver_denom_floor,
            projector_ptr,
            logging_enabled && options.verbosity == SCFVerbosity::Detailed
        );
        if (!ks.converged) {
            const double maximum_ks_residual =
                maximum_residual(ks.residual_norms);
            std::ostringstream message;
            message
                << "Davidson eigensolver did not converge during SCF: "
                << "max residual = " << maximum_ks_residual
                << ", requested tolerance = "
                << options.eigensolver_tolerance
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
        result.eigensolver_iterations += ks.iterations;
        result.eigensolver_restarts += ks.subspace_restarts;
        result.eigensolver_hamiltonian_seconds +=
            ks.hamiltonian_seconds;
        result.eigensolver_subspace_seconds +=
            ks.subspace_diagonalization_seconds;

        const OccupationResult occupations = compute_occupations(
            ks.eigenvalues,
            options.nelec,
            options.occupation_mode,
            options.fixed_occupations,
            options.smearing_sigma,
            options.degeneracy_tolerance
        );

        const Eigen::MatrixXcd orbitals =
            ks.eigenvectors.leftCols(options.nbands);
        const auto rho_out = build_density_from_orbitals(
            basis,
            fft,
            orbitals,
            occupations.occ,
            volume
        );

        const auto VH_out = build_hartree_potential(lattice, fft, rho_out);
        const auto xc_output = xc.evaluate(rho_out, dV);
        const double nonlocal_energy = compute_nonlocal_energy(
            projectors,
            orbitals,
            occupations.occ
        );

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
        const double drho = pulay_mix_density(
            pulay,
            rho,
            rho_out,
            dV,
            options.nelec
        );

        result.iterations = iter + 1;
        result.final_density_residual = drho;
        result.final_energy_change = dE;
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
                drho,
                ks.iterations,
                ks.hamiltonian_applications,
                maximum_residual(ks.residual_norms)
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
                pulay.history_size()
            );
        }

        if (iter > 0 &&
            drho < options.density_tolerance &&
            dE < options.energy_tolerance) {
            result.converged = true;
            if (logging_enabled &&
                options.verbosity == SCFVerbosity::Detailed) {
                *log_stream << "SCF converged.\n";
            }
            break;
        }

        previous_energy = energy_for_convergence;
        C_guess = orbitals;
    }

    result.wall_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scf_start
    ).count();

    if (logging_enabled && options.verbosity == SCFVerbosity::Compact) {
        print_compact_summary(*log_stream, result, last_signed_energy_change);
    }

    return result;
}
