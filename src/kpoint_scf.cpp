#include "scf.hpp"

#include "eigensolver.hpp"
#include "mixing.hpp"
#include "parallel.hpp"
#include "potentials.hpp"
#include "scf_convergence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

double maximum_residual(const std::vector<double>& residuals) {
    double maximum = 0.0;
    for (double residual : residuals) {
        maximum = std::max(maximum, residual);
    }
    return maximum;
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
    if (!initial_guess.density.empty() &&
        static_cast<int>(initial_guess.density.size()) != fft.grid.ngrid) {
        throw std::runtime_error("Initial density size does not match FFT grid.");
    }
    if (!initial_guess.orbitals.empty() &&
        initial_guess.orbitals.size() != kpoints.size()) {
        throw std::runtime_error(
            "Initial k-point orbital blocks must match the k-point count."
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
            const Eigen::MatrixXcd& orbitals = initial_guess.orbitals[ik];
            if (orbitals.size() != 0 &&
                (orbitals.rows() != point.basis.size() ||
                 orbitals.cols() < options.nbands)) {
                throw std::runtime_error(
                    "Initial k-point orbitals have incompatible dimensions."
                );
            }
        }
        weight_sum += point.weight;
    }
    if (std::abs(weight_sum - 1.0) > 1.0e-12) {
        throw std::runtime_error("Normalized k-point weights must sum to one.");
    }
}

void print_header(std::ostream& out, int nkpoints, int process_count) {
    out << "\n"
        << "------------------------- k-point electronic minimization "
        << "-------------------------\n"
        << "  NKPTS = " << nkpoints
        << "    MPI ranks = " << process_count
        << "    active k-point ranks = "
        << std::min(nkpoints, process_count) << "\n"
        << "          N                     E              dE"
        << "        rms(rho)  Niter  N_Hpsi      rms(eig)"
        << "       eig_tol\n";
}

void print_iteration(
    std::ostream& out,
    int iteration,
    double energy,
    double signed_energy_change,
    double density_residual,
    int eigensolver_iterations,
    int hamiltonian_applications,
    double eigensolver_residual,
    double eigensolver_tolerance) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "DAV: " << std::setw(4) << iteration + 1
        << "  " << std::scientific << std::setprecision(12)
        << std::setw(20) << energy
        << "  " << std::setprecision(4) << std::setw(12)
        << signed_energy_change
        << "  " << std::setw(12) << density_residual
        << "  " << std::setw(5) << eigensolver_iterations
        << "  " << std::setw(7) << hamiltonian_applications
        << "  " << std::setw(12) << eigensolver_residual
        << "  " << std::setw(12) << eigensolver_tolerance
        << "\n";
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
    out << "--------------------------------------------------------------------------------\n"
        << " " << std::setw(4) << result.iterations
        << " F= " << std::scientific << std::setprecision(12)
        << std::setw(20) << free_energy_with_ions
        << " E0= " << std::setw(20) << sigma0_with_ions
        << "  d E = " << std::setprecision(4) << std::setw(12)
        << signed_energy_change << "\n"
        << "  global mu = " << std::setprecision(12)
        << result.occupations.mu
        << "  weighted Ne = " << result.occupations.nelec_sum << "\n"
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
        << result.eigensolver_subspace_seconds << " s\n"
        << "  SCF wall time"
        << (parallel::size() > 1 ? "(max-rank)" : "")
        << " = "
        << result.wall_time_seconds << " s\n";
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
    std::ostream* log_stream) {

    const auto scf_start = std::chrono::steady_clock::now();
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
    std::vector<double> rho = initial_guess.density;
    if (rho.empty()) {
        rho.assign(fft.grid.ngrid, options.nelec / volume);
    } else {
        renormalize_density(rho, dV, options.nelec);
    }

    std::vector<Eigen::MatrixXcd> orbital_guesses(kpoints.size());
    for (int ik : distribution.local_kpoints()) {
        if (!initial_guess.orbitals.empty() &&
            initial_guess.orbitals[ik].size() != 0) {
            orbital_guesses[ik] = initial_guess.orbitals[ik];
        } else {
            const int trial_count = std::min(
                kpoints[ik].basis.size(), options.nbands + 4
            );
            orbital_guesses[ik] = initial_low_kinetic_trials(
                kpoints[ik].basis.size(), trial_count
            );
        }
    }

    PulayMixer pulay;
    pulay.alpha = options.mixing_alpha;
    pulay.max_history = options.pulay_max_history;
    pulay.min_history = options.pulay_min_history;
    pulay.regularization = options.pulay_regularization;
    LibXCLDAFunctional xc(options.lda_functional);
    std::vector<double> weights;
    weights.reserve(kpoints.size());
    for (const KPointHamiltonian& point : kpoints) {
        weights.push_back(point.weight);
    }

    KPointSCFResult result;
    double previous_energy = 0.0;
    double last_signed_energy_change = 0.0;
    const bool logging_enabled =
        parallel::is_root() && log_stream != nullptr
        && options.verbosity != SCFVerbosity::Silent;
    if (logging_enabled) {
        print_header(
            *log_stream,
            static_cast<int>(kpoints.size()),
            distribution.size()
        );
    }

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        const double eigensolver_tolerance = eigensolver_tolerances.current();
        const auto hartree_input = build_hartree_potential(lattice, fft, rho);
        const auto xc_input = xc.evaluate(rho, dV);
        const auto effective_potential = combine_effective_potential(
            ionic_potential, hartree_input, xc_input.Vxc
        );

        std::vector<DavidsonResult> solutions(kpoints.size());
        int local_eigensolver_steps = 0;
        int local_hamiltonian_applications = 0;
        int local_hamiltonian_block_calls = 0;
        int local_eigensolver_restarts = 0;
        double local_hamiltonian_seconds = 0.0;
        double local_subspace_seconds = 0.0;
        double local_maximum_residual = 0.0;
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
                DavidsonResult solution = davidson_lowest_eigenstates(
                    point.basis,
                    fft,
                    effective_potential,
                    options.nbands,
                    orbital_guesses[ik],
                    options.eigensolver_max_iterations,
                    maximum_subspace,
                    eigensolver_tolerance,
                    options.eigensolver_denom_floor,
                    projector_pointer,
                    logging_enabled
                        && distribution.size() == 1
                        && options.verbosity == SCFVerbosity::Detailed
                );
                const double residual = maximum_residual(
                    solution.residual_norms
                );
                if (!solution.converged) {
                    std::ostringstream message;
                    message
                        << "Davidson eigensolver did not converge at k point "
                        << ik << " ("
                        << point.fractional_position.transpose()
                        << "): max residual = " << residual
                        << ", requested tolerance = "
                        << eigensolver_tolerance
                        << ", worst band = "
                        << solution.max_residual_band
                        << ", subspace = "
                        << solution.final_subspace_size
                        << ", Hpsi applications = "
                        << solution.hamiltonian_applications;
                    throw std::runtime_error(message.str());
                }

                local_eigensolver_steps += solution.iterations;
                local_hamiltonian_applications +=
                    solution.hamiltonian_applications;
                local_hamiltonian_block_calls +=
                    solution.hamiltonian_block_calls;
                local_eigensolver_restarts += solution.subspace_restarts;
                local_hamiltonian_seconds += solution.hamiltonian_seconds;
                local_subspace_seconds +=
                    solution.subspace_diagonalization_seconds;
                local_maximum_residual = std::max(
                    local_maximum_residual, residual
                );
                solutions[ik] = std::move(solution);
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
        const double iteration_maximum_residual =
            parallel::maximum(local_maximum_residual);

        result.eigensolver_hamiltonian_applications +=
            iteration_hamiltonian_applications;
        result.eigensolver_hamiltonian_block_calls +=
            iteration_hamiltonian_block_calls;
        result.eigensolver_iterations += iteration_eigensolver_steps;
        result.eigensolver_restarts += iteration_eigensolver_restarts;
        result.eigensolver_hamiltonian_seconds +=
            iteration_hamiltonian_seconds;
        result.eigensolver_subspace_seconds += iteration_subspace_seconds;

        std::vector<double> packed_eigenvalues(
            kpoints.size() * static_cast<std::size_t>(options.nbands),
            0.0
        );
        for (int ik : distribution.local_kpoints()) {
            for (int ib = 0; ib < options.nbands; ++ib) {
                packed_eigenvalues[
                    static_cast<std::size_t>(ik) * options.nbands + ib
                ] = solutions[ik].eigenvalues[ib];
            }
        }
        parallel::sum_in_place(packed_eigenvalues);

        std::vector<Eigen::VectorXd> eigenvalues(kpoints.size());
        for (int ik = 0; ik < static_cast<int>(kpoints.size()); ++ik) {
            eigenvalues[ik].resize(options.nbands);
            for (int ib = 0; ib < options.nbands; ++ib) {
                eigenvalues[ik][ib] = packed_eigenvalues[
                    static_cast<std::size_t>(ik) * options.nbands + ib
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
                options.degeneracy_tolerance
            );

        std::vector<double> density_output(fft.grid.ngrid, 0.0);
        double local_kinetic_energy = 0.0;
        double local_nonlocal_energy = 0.0;
        std::vector<KPointElectronicState> electronic_states(kpoints.size());
        for (int ik = 0; ik < static_cast<int>(kpoints.size()); ++ik) {
            const KPointHamiltonian& point = kpoints[ik];
            KPointElectronicState& state = electronic_states[ik];
            state.fractional_position = point.fractional_position;
            state.weight = point.weight;
            state.owner_rank = distribution.owner(ik);
            state.eigenvalues = eigenvalues[ik];
            state.occupations = occupations.occupations[ik];
        }

        std::string local_density_error;
        try {
            for (int ik : distribution.local_kpoints()) {
                const KPointHamiltonian& point = kpoints[ik];
                Eigen::MatrixXcd orbitals =
                    solutions[ik].eigenvectors.leftCols(options.nbands);
                const auto density_at_k = build_density_from_orbitals(
                    point.basis,
                    fft,
                    orbitals,
                    occupations.occupations[ik],
                    volume
                );
                for (int grid_index = 0;
                     grid_index < fft.grid.ngrid;
                     ++grid_index) {
                    density_output[grid_index] +=
                        point.weight * density_at_k[grid_index];
                }
                local_kinetic_energy +=
                    point.weight * compute_kinetic_energy(
                        point.basis,
                        orbitals,
                        occupations.occupations[ik]
                    );
                local_nonlocal_energy +=
                    point.weight * compute_nonlocal_energy(
                        point.projectors,
                        orbitals,
                        occupations.occupations[ik]
                    );
                electronic_states[ik].orbitals = std::move(orbitals);
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

        parallel::sum_in_place(density_output);
        const double kinetic_energy = parallel::sum(local_kinetic_energy);
        const double nonlocal_energy = parallel::sum(local_nonlocal_energy);

        const auto hartree_output = build_hartree_potential(
            lattice, fft, density_output
        );
        const auto xc_output = xc.evaluate(density_output, dV);
        EnergyTerms energy;
        energy.kinetic = kinetic_energy;
        for (int grid_index = 0;
             grid_index < fft.grid.ngrid;
             ++grid_index) {
            energy.external += dV * density_output[grid_index]
                * ionic_potential[grid_index];
            energy.hartree += 0.5 * dV * density_output[grid_index]
                * hartree_output[grid_index];
        }
        energy.exchange = xc_output.exchange_energy;
        energy.correlation = xc_output.correlation_energy;
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
        const double density_residual = pulay_mix_density(
            pulay,
            rho,
            density_output,
            dV,
            options.nelec
        );

        result.iterations = iteration + 1;
        result.final_density_residual = density_residual;
        result.final_energy_change = energy_change;
        result.final_eigensolver_tolerance = eigensolver_tolerance;
        result.kpoints = std::move(electronic_states);
        result.occupations = occupations;
        result.density = density_output;
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
                density_residual,
                iteration_eigensolver_steps,
                iteration_hamiltonian_applications,
                iteration_maximum_residual,
                eigensolver_tolerance
            );
            if (options.verbosity == SCFVerbosity::Detailed) {
                *log_stream << "       mu = " << occupations.mu
                    << "  weighted Ne = " << occupations.nelec_sum
                    << "  eig_tol = " << eigensolver_tolerance << "\n";
                const bool print_bands =
                    iteration < options.band_print_interval
                    || (options.band_print_interval > 0 &&
                        iteration % options.band_print_interval == 0);
                if (print_bands) {
                    const int bands_to_print = std::min(
                        options.bands_to_print, options.nbands
                    );
                    for (int ik = 0;
                         ik < static_cast<int>(kpoints.size());
                         ++ik) {
                        *log_stream << "       k[" << ik << "] = "
                            << kpoints[ik].fractional_position.transpose()
                            << "  w = " << kpoints[ik].weight << "\n";
                        for (int ib = 0; ib < bands_to_print; ++ib) {
                            *log_stream << "         band " << ib
                                << "  eps = " << eigenvalues[ik][ib]
                                << "  occ = "
                                << occupations.occupations[ik][ib]
                                << "\n";
                        }
                    }
                }
            }
        }

        const bool outer_converged = iteration > 0 &&
            density_residual < options.density_tolerance &&
            energy_change < options.energy_tolerance;
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
        for (int ik : distribution.local_kpoints()) {
            orbital_guesses[ik] = result.kpoints[ik].orbitals;
        }
    }

    const double local_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scf_start
    ).count();
    result.wall_time_seconds = parallel::maximum(local_wall_time);
    if (logging_enabled) {
        print_summary(*log_stream, result, last_signed_energy_change);
    }
    return result;
}
