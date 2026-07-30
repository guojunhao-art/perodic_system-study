#include "nscf.hpp"

#include "eigensolver.hpp"
#include "parallel.hpp"
#include "scf_convergence.hpp"
#include "symmetry.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

int electronic_state_index(
    int spin,
    int kpoint,
    int kpoint_count) {

    return spin * kpoint_count + kpoint;
}

double maximum_residual(
    const std::vector<double>& residuals) {

    double maximum = 0.0;
    for (double residual : residuals) {
        maximum = std::max(maximum, residual);
    }
    return maximum;
}

double maximum_residual(
    const NSCFDiagonalizationResult& result) {

    double maximum = 0.0;
    for (const NSCFElectronicState& state :
         result.states) {
        maximum = std::max(
            maximum,
            maximum_residual(state.residual_norms)
        );
    }
    return maximum;
}

Eigen::Vector3i maximum_required_fft_frequency(
    const PlaneWaveBasis3D& basis) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error(
            "The NSCF plane-wave basis is empty."
        );
    }
    Eigen::Vector3i minimum =
        basis.gvectors.front().n;
    Eigen::Vector3i maximum =
        basis.gvectors.front().n;
    for (const GVector& vector : basis.gvectors) {
        minimum = minimum.cwiseMin(vector.n);
        maximum = maximum.cwiseMax(vector.n);
    }
    return maximum.cwiseAbs()
        .cwiseMax(minimum.cwiseAbs())
        .cwiseMax(maximum - minimum);
}

void require_checkpoint_grid_for_basis(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    int kpoint_index) {

    const Eigen::Vector3i required =
        maximum_required_fft_frequency(basis);
    const Eigen::Vector3i dimensions(
        grid.n1, grid.n2, grid.n3
    );
    for (int direction = 0; direction < 3; ++direction) {
        if (required[direction] >
            (dimensions[direction] - 1) / 2) {
            throw std::runtime_error(
                "The checkpoint FFT grid is too small for NSCF k point "
                + std::to_string(kpoint_index)
                + " in direction "
                + std::to_string(direction)
                + ". Regenerate the SCF checkpoint with a larger explicit "
                  "fft_grid."
            );
        }
    }
}

std::vector<double> sum_spin_densities(
    const std::vector<std::vector<double>>& spin_densities) {

    if (spin_densities.empty()) {
        throw std::runtime_error(
            "A fixed-density NSCF calculation needs spin densities."
        );
    }
    std::vector<double> total(
        spin_densities.front().size(), 0.0
    );
    for (const std::vector<double>& density :
         spin_densities) {
        if (density.size() != total.size()) {
            throw std::runtime_error(
                "NSCF spin-density grids have inconsistent sizes."
            );
        }
        for (std::size_t point = 0;
             point < total.size();
             ++point) {
            total[point] += density[point];
        }
    }
    return total;
}

std::vector<std::vector<double>>
build_fixed_effective_potentials(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    const std::vector<std::vector<double>>& spin_densities,
    const SCFOptions& options) {

    if (static_cast<int>(spin_densities.size()) !=
        options.nspin) {
        throw std::runtime_error(
            "Checkpoint spin-density channels do not match NSCF nspin."
        );
    }
    const double dV = lattice.volume()
        / static_cast<double>(fft.grid.ngrid);
    const std::vector<double> total_density =
        sum_spin_densities(spin_densities);
    const std::vector<double> hartree =
        build_hartree_potential(
            lattice, fft, total_density
        );
    LibXCLDAFunctional xc(
        options.lda_functional, options.nspin
    );
    std::vector<std::vector<double>> potentials(
        options.nspin
    );
    if (options.nspin == 1) {
        const XCResult xc_result =
            xc.evaluate(spin_densities[0], dV);
        potentials[0] = combine_effective_potential(
            ionic_potential, hartree, xc_result.Vxc
        );
    } else {
        const SpinXCResult xc_result = xc.evaluate_spin(
            spin_densities[0], spin_densities[1], dV
        );
        potentials[0] = combine_effective_potential(
            ionic_potential, hartree, xc_result.Vxc_up
        );
        potentials[1] = combine_effective_potential(
            ionic_potential, hartree, xc_result.Vxc_down
        );
    }
    return potentials;
}

SCFOptions resolve_nscf_options(
    const CalculationConfig& config,
    double resolved_nelect,
    const SCFCheckpoint& checkpoint,
    const Lattice& lattice) {

    SCFOptions options = config.scf;
    options.nelec = resolved_nelect;
    if (config.nbands_auto) {
        double majority_electrons =
            0.5 * options.nelec;
        if (options.nspin == 2) {
            const std::size_t grid_points =
                checkpoint.spin_densities.front().size();
            const double dV = lattice.volume()
                / static_cast<double>(grid_points);
            majority_electrons = std::max(
                electron_number_from_density(
                    checkpoint.spin_densities[0], dV
                ),
                electron_number_from_density(
                    checkpoint.spin_densities[1], dV
                )
            );
        }
        options.nbands = std::max(
            8,
            static_cast<int>(
                std::ceil(majority_electrons)
            ) + 4
        );
    }
    if (options.occupation_mode ==
            OccupationMode::Fixed &&
        static_cast<int>(
            options.fixed_occupations.size()
        ) != options.nbands) {
        throw std::runtime_error(
            "fixed_occupations must contain exactly nbands values."
        );
    }
    return options;
}

std::vector<Eigen::VectorXd> collect_eigenvalues(
    const NSCFDiagonalizationResult& diagonalization) {

    std::vector<Eigen::VectorXd> eigenvalues;
    eigenvalues.reserve(diagonalization.states.size());
    for (const NSCFElectronicState& state :
         diagonalization.states) {
        eigenvalues.push_back(state.eigenvalues);
    }
    return eigenvalues;
}

std::vector<double> repeated_kpoint_weights(
    const KPointSet& kpoints,
    int nspin) {

    std::vector<double> weights;
    weights.reserve(
        static_cast<std::size_t>(nspin)
        * kpoints.points.size()
    );
    for (int spin = 0; spin < nspin; ++spin) {
        for (const KPoint& point : kpoints.points) {
            weights.push_back(point.weight);
        }
    }
    return weights;
}

KPointOccupationResult occupations_from_diagonalization(
    const NSCFDiagonalizationResult& diagonalization,
    const KPointSet& kpoints,
    const SCFOptions& options) {

    return compute_kpoint_occupations(
        collect_eigenvalues(diagonalization),
        repeated_kpoint_weights(
            kpoints, options.nspin
        ),
        options.nelec,
        options.occupation_mode,
        options.fixed_occupations,
        options.smearing_sigma,
        options.degeneracy_tolerance,
        options.nspin == 1 ? 2.0 : 1.0,
        static_cast<double>(options.nspin)
    );
}

std::vector<std::vector<double>>
uniform_state_tolerances(
    int state_count,
    int nbands,
    double tolerance) {

    return std::vector<std::vector<double>>(
        state_count,
        std::vector<double>(nbands, tolerance)
    );
}

std::vector<std::vector<double>>
occupation_state_tolerances(
    const KPointOccupationResult& occupations,
    const SCFOptions& options) {

    std::vector<std::vector<double>> result;
    result.reserve(occupations.occupations.size());
    for (const std::vector<double>& state_occupations :
         occupations.occupations) {
        result.push_back(
            davidson_band_residual_tolerances(
                options.nbands,
                options.eigensolver_tolerance,
                state_occupations,
                options.eigensolver_empty_tolerance
            )
        );
    }
    return result;
}

std::vector<std::vector<double>>
path_state_tolerances(
    const NSCFDiagonalizationResult& preliminary,
    const SCFOptions& options,
    double checkpoint_fermi_energy_ha) {

    const double strict_window = std::max(
        5.0 * options.smearing_sigma,
        1.0e-4
    );
    std::vector<std::vector<double>> tolerances;
    tolerances.reserve(preliminary.states.size());
    for (const NSCFElectronicState& state :
         preliminary.states) {
        std::vector<double> pseudo_occupations(
            options.nbands, 0.0
        );
        for (int band = 0;
             band < options.nbands;
             ++band) {
            if (state.eigenvalues[band] <=
                checkpoint_fermi_energy_ha
                    + strict_window) {
                pseudo_occupations[band] = 1.0;
            }
        }
        tolerances.push_back(
            davidson_band_residual_tolerances(
                options.nbands,
                options.eigensolver_tolerance,
                pseudo_occupations,
                options.eigensolver_empty_tolerance
            )
        );
    }
    return tolerances;
}

NSCFDiagonalizationResult solve_fixed_potential_stage(
    const std::vector<KPointHamiltonian>& hamiltonians,
    FFTWorkspace& fft,
    const std::vector<std::vector<double>>& effective_potentials,
    const SCFOptions& options,
    const std::vector<std::vector<double>>& band_tolerances,
    double fermi_energy_ha,
    const NSCFDiagonalizationResult* initial_result = nullptr) {

    if (hamiltonians.empty() ||
        (options.nspin != 1 && options.nspin != 2) ||
        static_cast<int>(effective_potentials.size()) !=
            options.nspin) {
        throw std::runtime_error(
            "Fixed-potential NSCF stage has invalid dimensions."
        );
    }
    const int kpoint_count =
        static_cast<int>(hamiltonians.size());
    const int state_count =
        options.nspin * kpoint_count;
    if (static_cast<int>(band_tolerances.size()) !=
        state_count) {
        throw std::runtime_error(
            "NSCF band-tolerance state count is invalid."
        );
    }
    if (initial_result != nullptr &&
        static_cast<int>(initial_result->states.size()) !=
            state_count) {
        throw std::runtime_error(
            "NSCF initial-state count is invalid."
        );
    }

    const auto start =
        std::chrono::steady_clock::now();
    const parallel::KPointDistribution distribution(
        kpoint_count
    );
    const std::size_t band_value_count =
        static_cast<std::size_t>(state_count)
        * options.nbands;
    std::vector<double> packed_eigenvalues(
        band_value_count, 0.0
    );
    std::vector<double> packed_residuals(
        band_value_count, 0.0
    );
    std::vector<double> packed_iterations(
        state_count, 0.0
    );
    std::vector<Eigen::MatrixXcd> local_orbitals(
        state_count
    );
    std::string local_error;

    try {
        for (int ik : distribution.local_kpoints()) {
            const KPointHamiltonian& point =
                hamiltonians[ik];
            if (point.basis.size() < options.nbands) {
                throw std::runtime_error(
                    "NSCF k point " + std::to_string(ik)
                    + " has fewer plane waves than nbands."
                );
            }
            const int maximum_subspace =
                options.eigensolver_max_subspace > 0
                ? std::min(
                    options.eigensolver_max_subspace,
                    point.basis.size()
                )
                : std::min(
                    point.basis.size(),
                    4 * options.nbands + 8
                );
            if (maximum_subspace < options.nbands) {
                throw std::runtime_error(
                    "eigensolver_max_subspace is smaller than nbands "
                    "at NSCF k point " + std::to_string(ik) + "."
                );
            }
            const std::vector<NonlocalProjector>*
                projector_pointer =
                    point.projectors.empty()
                    ? nullptr
                    : &point.projectors;

            for (int spin = 0;
                 spin < options.nspin;
                 ++spin) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                Eigen::MatrixXcd trials;
                if (initial_result != nullptr &&
                    initial_result->states[state]
                        .orbitals.rows() ==
                        point.basis.size() &&
                    initial_result->states[state]
                        .orbitals.cols() >=
                        options.nbands) {
                    trials = initial_result->states[state]
                        .orbitals;
                } else {
                    const int trial_count = std::min(
                        maximum_subspace,
                        std::min(
                            point.basis.size(),
                            options.nbands + 4
                        )
                    );
                    trials = initial_low_kinetic_trials(
                        point.basis.size(), trial_count
                    );
                }
                DavidsonResult solution =
                    davidson_lowest_eigenstates(
                        point.basis,
                        fft,
                        effective_potentials[spin],
                        options.nbands,
                        trials,
                        options.eigensolver_max_iterations,
                        maximum_subspace,
                        options.eigensolver_tolerance,
                        options.eigensolver_denom_floor,
                        projector_pointer,
                        false,
                        &band_tolerances[state]
                    );
                const DavidsonResidualAssessment assessment =
                    assess_davidson_residuals(
                        solution.residual_norms,
                        band_tolerances[state],
                        options.eigensolver_tolerance
                    );
                if (!solution.converged &&
                    !assessment.acceptable) {
                    std::ostringstream message;
                    message
                        << "NSCF Davidson solve did not converge at spin "
                        << spin << ", k point " << ik
                        << " ("
                        << point.fractional_position.transpose()
                        << "): max residual = "
                        << maximum_residual(
                            solution.residual_norms
                        )
                        << ", worst band = "
                        << solution.max_residual_band
                        << ", strict-band residual = "
                        << assessment.maximum_strict_residual
                        << ", relaxed-band failures = "
                        << assessment.relaxed_band_failures;
                    throw std::runtime_error(message.str());
                }
                packed_iterations[state] =
                    static_cast<double>(
                        solution.iterations
                    );
                local_orbitals[state] =
                    std::move(solution.eigenvectors);
                for (int band = 0;
                     band < options.nbands;
                     ++band) {
                    const std::size_t index =
                        static_cast<std::size_t>(state)
                            * options.nbands + band;
                    packed_eigenvalues[index] =
                        solution.eigenvalues[band];
                    packed_residuals[index] =
                        solution.residual_norms[band];
                }
            }
        }
    } catch (const std::exception& error) {
        local_error =
            "NSCF solve failed on MPI rank "
            + std::to_string(distribution.rank())
            + ": " + error.what();
    } catch (...) {
        local_error =
            "NSCF solve failed with an unknown exception on MPI rank "
            + std::to_string(distribution.rank()) + ".";
    }
    const std::string error =
        parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }

    parallel::sum_in_place(packed_eigenvalues);
    parallel::sum_in_place(packed_residuals);
    parallel::sum_in_place(packed_iterations);

    NSCFDiagonalizationResult result;
    result.converged = true;
    result.nspin = options.nspin;
    result.nbands = options.nbands;
    result.fermi_energy_ha = fermi_energy_ha;
    result.states.resize(state_count);
    for (int spin = 0;
         spin < options.nspin;
         ++spin) {
        for (int ik = 0; ik < kpoint_count; ++ik) {
            const int state_index =
                electronic_state_index(
                    spin, ik, kpoint_count
                );
            NSCFElectronicState& state =
                result.states[state_index];
            state.spin_channel = spin;
            state.kpoint_index = ik;
            state.owner_rank =
                distribution.owner(ik);
            state.iterations = static_cast<int>(
                std::llround(
                    packed_iterations[state_index]
                )
            );
            state.eigenvalues.resize(options.nbands);
            state.residual_norms.resize(options.nbands);
            for (int band = 0;
                 band < options.nbands;
                 ++band) {
                const std::size_t index =
                    static_cast<std::size_t>(
                        state_index
                    ) * options.nbands + band;
                state.eigenvalues[band] =
                    packed_eigenvalues[index];
                state.residual_norms[band] =
                    packed_residuals[index];
            }
            if (distribution.owns(ik)) {
                state.orbitals =
                    std::move(local_orbitals[state_index]);
            }
        }
    }
    const double local_wall_time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
    result.wall_time_seconds =
        parallel::maximum(local_wall_time);
    return result;
}

void add_stage_iterations(
    NSCFDiagonalizationResult& result,
    const NSCFDiagonalizationResult& preceding) {

    if (result.states.size() != preceding.states.size()) {
        throw std::runtime_error(
            "Cannot combine NSCF stage iteration counts."
        );
    }
    for (int state = 0;
         state < static_cast<int>(result.states.size());
         ++state) {
        result.states[state].iterations +=
            preceding.states[state].iterations;
    }
    result.wall_time_seconds +=
        preceding.wall_time_seconds;
}

bool occupation_refinement_needed(
    const KPointOccupationResult& previous,
    const KPointOccupationResult& current) {

    if (previous.occupations.size() !=
        current.occupations.size()) {
        throw std::runtime_error(
            "NSCF occupation-state counts do not match."
        );
    }
    for (int state = 0;
         state < static_cast<int>(
             current.occupations.size()
         );
         ++state) {
        if (davidson_occupation_refinement_needed(
                previous.occupations[state],
                current.occupations[state]
            )) {
            return true;
        }
    }
    return false;
}

NSCFDiagonalizationResult solve_fixed_potential_nscf(
    const std::vector<KPointHamiltonian>& hamiltonians,
    FFTWorkspace& fft,
    const std::vector<std::vector<double>>& effective_potentials,
    const SCFOptions& options,
    const KPointSet& kpoints,
    bool band_path,
    double checkpoint_fermi_energy_ha) {

    const int state_count =
        options.nspin
        * static_cast<int>(hamiltonians.size());
    if (options.eigensolver_full_band_accuracy) {
        return solve_fixed_potential_stage(
            hamiltonians,
            fft,
            effective_potentials,
            options,
            uniform_state_tolerances(
                state_count,
                options.nbands,
                options.eigensolver_tolerance
            ),
            checkpoint_fermi_energy_ha
        );
    }

    const double preliminary_tolerance = std::max(
        DAVIDSON_EMPTY_TOLERANCE_FACTOR
            * options.eigensolver_tolerance,
        options.eigensolver_empty_tolerance
    );
    NSCFDiagonalizationResult preliminary =
        solve_fixed_potential_stage(
            hamiltonians,
            fft,
            effective_potentials,
            options,
            uniform_state_tolerances(
                state_count,
                options.nbands,
                preliminary_tolerance
            ),
            checkpoint_fermi_energy_ha
        );

    std::vector<std::vector<double>> final_tolerances;
    KPointOccupationResult preliminary_occupations;
    if (band_path) {
        final_tolerances = path_state_tolerances(
            preliminary,
            options,
            checkpoint_fermi_energy_ha
        );
    } else {
        preliminary_occupations =
            occupations_from_diagonalization(
                preliminary, kpoints, options
            );
        final_tolerances = occupation_state_tolerances(
            preliminary_occupations, options
        );
    }

    NSCFDiagonalizationResult final =
        solve_fixed_potential_stage(
            hamiltonians,
            fft,
            effective_potentials,
            options,
            final_tolerances,
            checkpoint_fermi_energy_ha,
            &preliminary
        );
    add_stage_iterations(final, preliminary);

    if (!band_path) {
        const KPointOccupationResult final_occupations =
            occupations_from_diagonalization(
                final, kpoints, options
            );
        if (occupation_refinement_needed(
                preliminary_occupations,
                final_occupations
            )) {
            NSCFDiagonalizationResult refined =
                solve_fixed_potential_stage(
                    hamiltonians,
                    fft,
                    effective_potentials,
                    options,
                    occupation_state_tolerances(
                        final_occupations, options
                    ),
                    checkpoint_fermi_energy_ha,
                    &final
                );
            add_stage_iterations(refined, final);
            final = std::move(refined);
        }
    }
    return final;
}

AtomicProjectionResult compute_nscf_projections(
    const Lattice& lattice,
    const std::vector<KPointHamiltonian>& hamiltonians,
    const std::vector<AtomicProjectionSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    const NSCFDiagonalizationResult& diagonalization,
    double lowdin_cutoff,
    int thread_count) {

    const int kpoint_count =
        static_cast<int>(hamiltonians.size());
    const int state_count =
        diagonalization.nspin * kpoint_count;
    const std::vector<AtomicProjectionOrbital> orbitals =
        make_atomic_projection_orbitals(species, ions);
    const int orbital_count =
        static_cast<int>(orbitals.size());
    const std::size_t weight_count =
        static_cast<std::size_t>(state_count)
        * orbital_count
        * diagonalization.nbands;
    std::vector<double> packed_weights(
        weight_count, 0.0
    );
    const parallel::KPointDistribution distribution(
        kpoint_count
    );
    double local_minimum_overlap =
        std::numeric_limits<double>::infinity();
    double local_maximum_overlap = 0.0;
    double local_maximum_orthonormality_error = 0.0;
    double local_maximum_projection_excess = 0.0;
    std::string local_error;

    try {
        for (int ik : distribution.local_kpoints()) {
            const AtomicProjectionBasis projection_basis =
                build_lowdin_atomic_projection_basis(
                    lattice,
                    hamiltonians[ik].basis,
                    species,
                    ions,
                    lowdin_cutoff,
                    thread_count
                );
            if (projection_basis.orbitals.size() !=
                orbitals.size()) {
                throw std::runtime_error(
                    "Atomic projection ordering changed between k points."
                );
            }
            local_minimum_overlap = std::min(
                local_minimum_overlap,
                projection_basis.minimum_overlap_eigenvalue
            );
            local_maximum_overlap = std::max(
                local_maximum_overlap,
                projection_basis.maximum_overlap_eigenvalue
            );
            local_maximum_orthonormality_error = std::max(
                local_maximum_orthonormality_error,
                projection_basis.orthonormality_error
            );
            for (int spin = 0;
                 spin < diagonalization.nspin;
                 ++spin) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                const NSCFElectronicState& electronic_state =
                    diagonalization.states[state];
                if (electronic_state.owner_rank !=
                        distribution.rank() ||
                    electronic_state.orbitals.rows() !=
                        hamiltonians[ik].basis.size()) {
                    throw std::runtime_error(
                        "The owning MPI rank does not retain the NSCF "
                        "eigenvectors needed for PDOS."
                    );
                }
                const Eigen::MatrixXd weights =
                    compute_lowdin_projection_weights(
                        projection_basis,
                        electronic_state.orbitals
                    );
                for (int band = 0;
                     band < diagonalization.nbands;
                     ++band) {
                    local_maximum_projection_excess =
                        std::max(
                            local_maximum_projection_excess,
                            std::max(
                                0.0,
                                weights.col(band).sum()
                                    - 1.0
                            )
                        );
                    for (int orbital = 0;
                         orbital < orbital_count;
                         ++orbital) {
                        const std::size_t index =
                            (static_cast<std::size_t>(state)
                                * orbital_count
                                + orbital)
                                * diagonalization.nbands
                            + band;
                        packed_weights[index] =
                            weights(orbital, band);
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        local_error =
            "PDOS projection failed on MPI rank "
            + std::to_string(distribution.rank())
            + ": " + error.what();
    } catch (...) {
        local_error =
            "PDOS projection failed with an unknown exception on MPI rank "
            + std::to_string(distribution.rank()) + ".";
    }
    const std::string error =
        parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
    parallel::sum_in_place(packed_weights);

    AtomicProjectionResult result;
    result.orbitals = orbitals;
    result.minimum_overlap_eigenvalue =
        -parallel::maximum(-local_minimum_overlap);
    result.maximum_overlap_eigenvalue =
        parallel::maximum(local_maximum_overlap);
    result.maximum_orthonormality_error =
        parallel::maximum(
            local_maximum_orthonormality_error
        );
    result.maximum_projection_weight_excess =
        parallel::maximum(
            local_maximum_projection_excess
        );
    result.states.resize(state_count);
    for (int spin = 0;
         spin < diagonalization.nspin;
         ++spin) {
        for (int ik = 0; ik < kpoint_count; ++ik) {
            const int state_index =
                electronic_state_index(
                    spin, ik, kpoint_count
                );
            AtomicProjectionState& state =
                result.states[state_index];
            state.spin_channel = spin;
            state.kpoint_index = ik;
            state.weights.resize(
                orbital_count,
                diagonalization.nbands
            );
            for (int orbital = 0;
                 orbital < orbital_count;
                 ++orbital) {
                for (int band = 0;
                     band < diagonalization.nbands;
                     ++band) {
                    const std::size_t index =
                        (static_cast<std::size_t>(
                            state_index
                        ) * orbital_count + orbital)
                            * diagonalization.nbands
                        + band;
                    state.weights(orbital, band) =
                        packed_weights[index];
                }
            }
        }
    }
    return result;
}

} // namespace

NSCFResult run_fixed_density_nscf(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SCFCheckpoint& checkpoint,
    std::ostream* log_stream) {

    const auto start =
        std::chrono::steady_clock::now();
    if (config.calculation != CalculationType::NSCF) {
        throw std::runtime_error(
            "run_fixed_density_nscf requires calculation = nscf."
        );
    }
    const bool band_path =
        !config.bands.path.empty();
    const bool requests_pdos =
        !config.pdos.output_path.empty();
    if (band_path &&
        (!config.dos.output_path.empty() ||
         requests_pdos)) {
        throw std::runtime_error(
            "A high-symmetry NSCF path cannot generate DOS or PDOS."
        );
    }
    if (requests_pdos &&
        config.kpoint_symmetry.enabled) {
        throw std::runtime_error(
            "PDOS currently requires kpoint_symmetry = off."
        );
    }

    const Lattice lattice(
        structure.lattice_bohr.col(0),
        structure.lattice_bohr.col(1),
        structure.lattice_bohr.col(2)
    );

    std::map<std::string, int> species_indices;
    std::vector<UPFData> upfs;
    upfs.reserve(structure.species_order.size());
    for (int species = 0;
         species < static_cast<int>(
             structure.species_order.size()
         );
         ++species) {
        const std::string& element =
            structure.species_order[species];
        const auto mapping =
            config.pseudopotential_paths.find(element);
        if (mapping ==
            config.pseudopotential_paths.end()) {
            throw std::runtime_error(
                "No pseudopotential mapping was provided for element "
                + element + "."
            );
        }
        UPFData upf =
            read_nc_upf(mapping->second);
        if (upf.header.element != element) {
            throw std::runtime_error(
                "POSCAR element " + element
                + " is mapped to a UPF for "
                + upf.header.element + "."
            );
        }
        species_indices.emplace(element, species);
        upfs.push_back(std::move(upf));
    }

    std::vector<UPFLocalSpecies> local_species;
    std::vector<UPFNonlocalSpecies> nonlocal_species;
    std::vector<AtomicProjectionSpecies>
        projection_species;
    local_species.reserve(upfs.size());
    nonlocal_species.reserve(upfs.size());
    if (requests_pdos) {
        projection_species.reserve(upfs.size());
    }
    for (const UPFData& upf : upfs) {
        local_species.push_back(
            prepare_upf_local_species(
                upf, config.ewald_width_bohr
            )
        );
        nonlocal_species.push_back(
            prepare_upf_nonlocal_species(upf)
        );
        if (requests_pdos) {
            projection_species.push_back(
                prepare_atomic_projection_species(upf)
            );
        }
    }

    std::vector<UPFLocalIon> ions;
    ions.reserve(structure.atoms.size());
    double automatic_nelect = 0.0;
    for (const StructureAtom& atom :
         structure.atoms) {
        const auto species =
            species_indices.find(atom.element);
        if (species == species_indices.end()) {
            throw std::runtime_error(
                "Atom element " + atom.element
                + " was not declared in the POSCAR species line."
            );
        }
        ions.push_back({
            species->second, atom.frac_position
        });
        automatic_nelect +=
            local_species[species->second]
                .valence_charge;
    }
    const double resolved_nelect =
        config.nelect_auto
        ? automatic_nelect
        : config.scf.nelec;
    validate_scf_checkpoint(
        checkpoint,
        structure,
        config,
        resolved_nelect
    );
    SCFOptions options = resolve_nscf_options(
        config,
        resolved_nelect,
        checkpoint,
        lattice
    );

    std::vector<BandPathSample> path;
    KPointReductionResult reduction;
    KPointSet kpoints;
    if (band_path) {
        path = interpolate_band_path(
            lattice, config.bands
        );
        kpoints.points.clear();
        kpoints.description =
            "High-symmetry path";
        kpoints.uniform_mesh = false;
        for (const BandPathSample& sample : path) {
            KPoint point;
            point.frac_position =
                sample.frac_position;
            point.weight = 0.0;
            kpoints.points.push_back(point);
        }
        reduction.irreducible_kpoints = kpoints;
        reduction.full_kpoint_count =
            static_cast<int>(kpoints.points.size());
    } else if (config.kpoint_symmetry.enabled) {
        reduction = reduce_kpoints_by_symmetry(
            structure,
            config.kpoints,
            config.kpoint_symmetry.tolerance_angstrom,
            config.kpoint_symmetry.include_time_reversal
        );
        kpoints =
            reduction.irreducible_kpoints;
    } else {
        kpoints = config.kpoints;
        reduction.irreducible_kpoints = kpoints;
        reduction.full_kpoint_count =
            static_cast<int>(kpoints.points.size());
    }

    const FFTGrid grid(
        checkpoint.fft_grid[0],
        checkpoint.fft_grid[1],
        checkpoint.fft_grid[2]
    );
    FFTWorkspace fft(
        grid, config.fft_threads
    );
    std::vector<KPointHamiltonian> hamiltonians(
        kpoints.points.size()
    );
    const parallel::KPointDistribution distribution(
        static_cast<int>(kpoints.points.size())
    );
    for (int point = 0;
         point < static_cast<int>(
             kpoints.points.size()
         );
         ++point) {
        KPointHamiltonian& hamiltonian =
            hamiltonians[point];
        hamiltonian.fractional_position =
            kpoints.points[point].frac_position;
        hamiltonian.weight =
            kpoints.points[point].weight;
        hamiltonian.basis.generate(
            lattice,
            lattice.B
                * kpoints.points[point].frac_position,
            checkpoint.ecut_hartree
        );
        if (hamiltonian.basis.size() <
            options.nbands) {
            throw std::runtime_error(
                "The NSCF plane-wave basis at k point "
                + std::to_string(point)
                + " has fewer vectors than requested bands."
            );
        }
        require_checkpoint_grid_for_basis(
            hamiltonian.basis, grid, point
        );
        if (distribution.owns(point)) {
            hamiltonian.projectors =
                build_upf_nonlocal_projectors(
                    lattice,
                    hamiltonian.basis,
                    nonlocal_species,
                    ions,
                    fft.thread_count
                );
        }
    }

    const UPFLocalReciprocalCache local_cache =
        build_upf_local_reciprocal_cache(
            lattice,
            grid,
            local_species,
            fft.thread_count
        );
    const std::vector<double> ionic_potential =
        build_upf_local_potential_real(
            lattice, fft, local_cache, ions
        );
    const std::vector<std::vector<double>>
        effective_potentials =
            build_fixed_effective_potentials(
                lattice,
                fft,
                ionic_potential,
                checkpoint.spin_densities,
                options
            );

    NSCFDiagonalizationResult diagonalization =
        solve_fixed_potential_nscf(
            hamiltonians,
            fft,
            effective_potentials,
            options,
            kpoints,
            band_path,
            checkpoint.fermi_energy_ha
        );

    const int kpoint_count =
        static_cast<int>(kpoints.points.size());
    const int state_count =
        options.nspin * kpoint_count;
    KPointOccupationResult occupations;
    if (band_path) {
        occupations.mu =
            checkpoint.fermi_energy_ha;
    } else {
        occupations =
            occupations_from_diagonalization(
                diagonalization,
                kpoints,
                options
            );
        diagonalization.fermi_energy_ha =
            occupations.mu;
    }

    KPointSCFResult electronic;
    electronic.converged =
        diagonalization.converged;
    electronic.iterations = 1;
    electronic.occupations = occupations;
    electronic.spin_densities =
        checkpoint.spin_densities;
    electronic.density =
        sum_spin_densities(
            checkpoint.spin_densities
        );
    const double dV = lattice.volume()
        / static_cast<double>(grid.ngrid);
    electronic.electron_number_from_density =
        electron_number_from_density(
            electronic.density, dV
        );
    if (std::abs(
            electronic.electron_number_from_density
            - options.nelec) > 1.0e-7) {
        throw std::runtime_error(
            "Checkpoint density integral does not match nelect."
        );
    }
    electronic.kpoints.resize(state_count);
    for (int spin = 0;
         spin < options.nspin;
         ++spin) {
        for (int point = 0;
             point < kpoint_count;
             ++point) {
            const int state_index =
                electronic_state_index(
                    spin, point, kpoint_count
                );
            KPointElectronicState& state =
                electronic.kpoints[state_index];
            state.spin_channel = spin;
            state.kpoint_index = point;
            state.fractional_position =
                kpoints.points[point].frac_position;
            state.weight =
                kpoints.points[point].weight;
            state.owner_rank =
                distribution.owner(point);
            state.eigenvalues =
                diagonalization.states[state_index]
                    .eigenvalues;
            if (!band_path) {
                state.occupations =
                    occupations.occupations[
                        state_index
                    ];
                electronic.spin_electron_counts[spin] +=
                    state.weight * std::accumulate(
                        state.occupations.begin(),
                        state.occupations.end(),
                        0.0
                    );
            }
        }
    }
    if (!band_path && options.nspin == 2) {
        electronic.magnetization =
            electronic.spin_electron_counts[0]
            - electronic.spin_electron_counts[1];
    }

    AtomicProjectionResult projection;
    if (requests_pdos) {
        projection = compute_nscf_projections(
            lattice,
            hamiltonians,
            projection_species,
            ions,
            diagonalization,
            config.pdos.lowdin_relative_cutoff,
            fft.thread_count
        );
    }

    NSCFResult result;
    result.converged =
        diagonalization.converged;
    result.full_kpoint_count =
        reduction.full_kpoint_count;
    result.irreducible_kpoint_count =
        kpoint_count;
    result.space_group_operation_count =
        static_cast<int>(
            reduction.space_group_operations.size()
        );
    result.mesh_symmetry_operation_count =
        static_cast<int>(
            reduction.mesh_compatible_operations.size()
        );
    result.kpoint_time_reversal_used =
        reduction.time_reversal_used;
    result.checkpoint_fermi_energy_ha =
        checkpoint.fermi_energy_ha;
    result.band_path = band_path;
    result.path = std::move(path);
    result.options_used = options;
    result.electronic = std::move(electronic);
    result.diagonalization =
        std::move(diagonalization);
    result.projection =
        std::move(projection);
    const double local_wall_time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
    result.wall_time_seconds =
        parallel::maximum(local_wall_time);

    if (log_stream && parallel::is_root()) {
        print_nscf_result(*log_stream, result);
    }
    return result;
}

void print_nscf_result(
    std::ostream& out,
    const NSCFResult& result) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "\n FIXED-DENSITY NSCF"
        << (result.band_path
            ? " BAND PATH"
            : " BRILLOUIN-ZONE GRID")
        << "\n"
        << " -------------------------------------------------------------------------------\n"
        << "  NKPTS(full/solved) = "
        << result.full_kpoint_count << "/"
        << result.irreducible_kpoint_count
        << "    NBANDS = "
        << result.options_used.nbands
        << "    NSPIN = "
        << result.options_used.nspin << "\n"
        << std::scientific
        << std::setprecision(12)
        << "  checkpoint E_F = "
        << result.checkpoint_fermi_energy_ha
        << " Ha";
    if (!result.band_path) {
        out << "    NSCF E_F = "
            << result.electronic.occupations.mu
            << " Ha\n"
            << "  electron count(eigenvalues/density) = "
            << result.electronic.occupations.nelec_sum
            << " / "
            << result.electronic
                .electron_number_from_density
            << "\n";
    } else {
        out << " (path reference)\n"
            << "  checkpoint density electron count = "
            << result.electronic
                .electron_number_from_density
            << "\n";
    }
    out << "  maximum Davidson residual = "
        << maximum_residual(
            result.diagonalization
        ) << "\n";
    if (!result.projection.orbitals.empty()) {
        out << "  Löwdin orbitals = "
            << result.projection.orbitals.size()
            << "    overlap eigenvalue range = ["
            << result.projection
                .minimum_overlap_eigenvalue
            << ", "
            << result.projection
                .maximum_overlap_eigenvalue
            << "]\n"
            << "  maximum Löwdin orthonormality error = "
            << result.projection
                .maximum_orthonormality_error
            << "\n";
    }
    out << std::fixed << std::setprecision(3)
        << "  NSCF wall time"
        << (parallel::size() > 1
            ? "(max-rank)"
            : "")
        << " = " << result.wall_time_seconds
        << " s\n";
    out.flags(flags);
    out.precision(precision);
}
