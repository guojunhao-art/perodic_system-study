#include "bands.hpp"

#include "parallel.hpp"
#include "scf_modules.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

int electronic_state_index(int spin, int kpoint, int kpoint_count) {
    return spin * kpoint_count + kpoint;
}

double maximum_residual(const std::vector<double>& residuals) {
    double maximum = 0.0;
    for (double residual : residuals) {
        maximum = std::max(maximum, residual);
    }
    return maximum;
}

Eigen::Vector3i maximum_required_fft_frequency(
    const PlaneWaveBasis3D& basis) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("A band-path plane-wave basis is empty.");
    }
    Eigen::Vector3i minimum_frequency = basis.gvectors.front().n;
    Eigen::Vector3i maximum_frequency = basis.gvectors.front().n;
    for (const GVector& gvector : basis.gvectors) {
        minimum_frequency = minimum_frequency.cwiseMin(gvector.n);
        maximum_frequency = maximum_frequency.cwiseMax(gvector.n);
    }
    return maximum_frequency.cwiseAbs()
        .cwiseMax(minimum_frequency.cwiseAbs())
        .cwiseMax(maximum_frequency - minimum_frequency);
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    int kpoint_index) {

    const Eigen::Vector3i required =
        maximum_required_fft_frequency(basis);
    const Eigen::Vector3i grid_sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_unaliased_frequency =
            (grid_sizes[direction] - 1) / 2;
        if (required[direction] > largest_unaliased_frequency) {
            throw std::runtime_error(
                "The SCF FFT grid is too small for band-path k point "
                + std::to_string(kpoint_index) + " in direction "
                + std::to_string(direction)
                + ". Use fft_grid = auto with calculation = bands or "
                  "relax_bands, or provide a larger explicit grid."
            );
        }
    }
}

std::vector<std::vector<double>> build_frozen_effective_potentials(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    const KPointSCFResult& scf,
    const SCFOptions& options) {

    if (static_cast<int>(scf.spin_densities.size()) != options.nspin) {
        throw std::runtime_error(
            "The converged SCF spin densities do not match nspin."
        );
    }
    const double dV = lattice.volume()
        / static_cast<double>(fft.grid.ngrid);
    const std::vector<double> hartree =
        build_hartree_potential(lattice, fft, scf.density);
    LibXCLDAFunctional xc(options.lda_functional, options.nspin);
    std::vector<std::vector<double>> potentials(options.nspin);
    if (options.nspin == 1) {
        const XCResult xc_result = xc.evaluate(
            scf.spin_densities[0], dV
        );
        potentials[0] = combine_effective_potential(
            ionic_potential, hartree, xc_result.Vxc
        );
    } else {
        const SpinXCResult xc_result = xc.evaluate_spin(
            scf.spin_densities[0], scf.spin_densities[1], dV
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

} // namespace

std::vector<BandPathSample> interpolate_band_path(
    const Lattice& lattice,
    const BandStructureOptions& options) {

    if (options.path.size() < 2) {
        throw std::runtime_error(
            "A band structure requires at least two band_point nodes."
        );
    }
    if (options.points_per_segment < 2) {
        throw std::runtime_error(
            "band_points_per_segment must be at least 2."
        );
    }

    std::vector<BandPathSample> samples;
    samples.reserve(
        (options.path.size() - 1)
            * static_cast<std::size_t>(options.points_per_segment - 1)
        + 1
    );
    double distance = 0.0;
    for (int segment = 0;
         segment + 1 < static_cast<int>(options.path.size());
         ++segment) {
        const BandPathNode& begin = options.path[segment];
        const BandPathNode& end = options.path[segment + 1];
        if (!begin.frac_position.allFinite() ||
            !end.frac_position.allFinite()) {
            throw std::runtime_error(
                "Band-path reciprocal coordinates must be finite."
            );
        }
        const Eigen::Vector3d step_fractional =
            (end.frac_position - begin.frac_position)
            / static_cast<double>(options.points_per_segment - 1);
        const double step_distance =
            (lattice.B * step_fractional).norm();
        const int first_sample = segment == 0 ? 0 : 1;
        for (int point = first_sample;
             point < options.points_per_segment;
             ++point) {
            if (!samples.empty()) {
                distance += step_distance;
            }
            BandPathSample sample;
            sample.frac_position =
                begin.frac_position
                + static_cast<double>(point) * step_fractional;
            sample.distance_bohr_inverse = distance;
            if (point == 0) {
                sample.label = begin.label;
            }
            if (point == options.points_per_segment - 1) {
                sample.label = end.label;
            }
            samples.push_back(std::move(sample));
        }
    }
    return samples;
}

BandStructureResult solve_fixed_potential_bands(
    const std::vector<BandPathSample>& path,
    const std::vector<KPointHamiltonian>& hamiltonians,
    FFTWorkspace& fft,
    const std::vector<std::vector<double>>& effective_potentials,
    const SCFOptions& options,
    double fermi_energy_ha,
    std::ostream* log_stream) {

    if (path.empty() || hamiltonians.size() != path.size()) {
        throw std::runtime_error(
            "Band-path samples and Hamiltonians must have the same nonzero size."
        );
    }
    if (options.nspin != 1 && options.nspin != 2) {
        throw std::runtime_error("Band calculation nspin must be 1 or 2.");
    }
    if (static_cast<int>(effective_potentials.size()) != options.nspin) {
        throw std::runtime_error(
            "A fixed effective potential is required for every spin channel."
        );
    }
    for (const std::vector<double>& potential : effective_potentials) {
        if (static_cast<int>(potential.size()) != fft.grid.ngrid) {
            throw std::runtime_error(
                "A band effective-potential grid has the wrong size."
            );
        }
    }
    if (options.nbands <= 0) {
        throw std::runtime_error(
            "A band calculation requires a positive nbands."
        );
    }
    if (!std::isfinite(fermi_energy_ha)) {
        throw std::runtime_error("The SCF Fermi energy is not finite.");
    }

    const auto start = std::chrono::steady_clock::now();
    const int kpoint_count = static_cast<int>(path.size());
    const int state_count = options.nspin * kpoint_count;
    const parallel::KPointDistribution distribution(kpoint_count);
    const std::size_t band_value_count =
        static_cast<std::size_t>(state_count) * options.nbands;
    std::vector<double> packed_eigenvalues(band_value_count, 0.0);
    std::vector<double> packed_residuals(band_value_count, 0.0);
    std::vector<double> packed_iterations(state_count, 0.0);
    std::string local_error;

    try {
        for (int ik : distribution.local_kpoints()) {
            const KPointHamiltonian& point = hamiltonians[ik];
            if (point.basis.size() < options.nbands) {
                throw std::runtime_error(
                    "Band-path k point " + std::to_string(ik)
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
                    point.basis.size(), 4 * options.nbands + 8
                );
            const int trial_count = std::min(
                point.basis.size(), options.nbands + 4
            );
            const Eigen::MatrixXcd trials =
                initial_low_kinetic_trials(
                    point.basis.size(), trial_count
                );
            const std::vector<NonlocalProjector>* projectors =
                point.projectors.empty() ? nullptr : &point.projectors;

            for (int spin = 0; spin < options.nspin; ++spin) {
                const int state = electronic_state_index(
                    spin, ik, kpoint_count
                );
                DavidsonResult solution = davidson_lowest_eigenstates(
                    point.basis,
                    fft,
                    effective_potentials[spin],
                    options.nbands,
                    trials,
                    options.eigensolver_max_iterations,
                    maximum_subspace,
                    options.eigensolver_tolerance,
                    options.eigensolver_denom_floor,
                    projectors,
                    false
                );
                const double residual =
                    maximum_residual(solution.residual_norms);
                if (!solution.converged) {
                    std::ostringstream message;
                    message
                        << "Band Davidson solve did not converge at spin "
                        << spin << ", path point " << ik << " ("
                        << point.fractional_position.transpose()
                        << "): max residual = " << residual
                        << ", requested tolerance = "
                        << options.eigensolver_tolerance;
                    throw std::runtime_error(message.str());
                }
                packed_iterations[state] =
                    static_cast<double>(solution.iterations);
                for (int band = 0; band < options.nbands; ++band) {
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
            "Band solve failed on MPI rank "
            + std::to_string(distribution.rank()) + ": " + error.what();
    } catch (...) {
        local_error =
            "Band solve failed with an unknown exception on MPI rank "
            + std::to_string(distribution.rank()) + ".";
    }
    const std::string error = parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }

    parallel::sum_in_place(packed_eigenvalues);
    parallel::sum_in_place(packed_residuals);
    parallel::sum_in_place(packed_iterations);

    BandStructureResult result;
    result.converged = true;
    result.nspin = options.nspin;
    result.nbands = options.nbands;
    result.fermi_energy_ha = fermi_energy_ha;
    result.path = path;
    result.states.resize(state_count);
    for (int spin = 0; spin < options.nspin; ++spin) {
        for (int ik = 0; ik < kpoint_count; ++ik) {
            const int state_index = electronic_state_index(
                spin, ik, kpoint_count
            );
            BandElectronicState& state = result.states[state_index];
            state.spin_channel = spin;
            state.kpoint_index = ik;
            state.owner_rank = distribution.owner(ik);
            state.iterations = static_cast<int>(
                std::llround(packed_iterations[state_index])
            );
            state.eigenvalues.resize(options.nbands);
            state.residual_norms.resize(options.nbands);
            for (int band = 0; band < options.nbands; ++band) {
                const std::size_t index =
                    static_cast<std::size_t>(state_index)
                        * options.nbands + band;
                state.eigenvalues[band] = packed_eigenvalues[index];
                state.residual_norms[band] = packed_residuals[index];
            }
        }
    }
    const double local_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    result.wall_time_seconds = parallel::maximum(local_wall_time);

    if (log_stream && parallel::is_root()) {
        const auto flags = log_stream->flags();
        const auto precision = log_stream->precision();
        *log_stream
            << "\n FIXED-POTENTIAL BAND STRUCTURE (NSCF)\n"
            << " -------------------------------------------------------------------------------\n"
            << "  NKPTS(path) = " << kpoint_count
            << "    NBANDS = " << options.nbands
            << "    NSPIN = " << options.nspin
            << "    E_F = " << std::scientific
            << std::setprecision(12) << fermi_energy_ha << " Ha\n";
        for (int ik = 0; ik < kpoint_count; ++ik) {
            int maximum_iterations = 0;
            double residual = 0.0;
            for (int spin = 0; spin < options.nspin; ++spin) {
                const BandElectronicState& state =
                    result.states[electronic_state_index(
                        spin, ik, kpoint_count
                    )];
                maximum_iterations = std::max(
                    maximum_iterations, state.iterations
                );
                residual = std::max(
                    residual,
                    maximum_residual(state.residual_norms)
                );
            }
            *log_stream << " BAND: " << std::setw(4) << ik
                << "  k = " << std::fixed << std::setprecision(8)
                << std::setw(11) << path[ik].frac_position[0]
                << std::setw(11) << path[ik].frac_position[1]
                << std::setw(11) << path[ik].frac_position[2]
                << "  x = " << std::setw(12)
                << path[ik].distance_bohr_inverse
                << "  iter = " << std::setw(3) << maximum_iterations
                << "  max residual = " << std::scientific
                << std::setprecision(4) << residual;
            if (!path[ik].label.empty()) {
                *log_stream << "  label = " << path[ik].label;
            }
            *log_stream << "\n";
        }
        *log_stream << "  band NSCF wall time"
            << (parallel::size() > 1 ? "(max-rank)" : "")
            << " = " << std::fixed << std::setprecision(3)
            << result.wall_time_seconds << " s\n";
        log_stream->flags(flags);
        log_stream->precision(precision);
    }
    return result;
}

BandStructureResult run_band_structure(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SinglePointResult& scf_result,
    std::ostream* log_stream) {

    if (!scf_result.converged) {
        throw std::runtime_error(
            "Band calculation requires a converged SCF result."
        );
    }
    if (scf_result.fft_grid[0] <= 0 ||
        scf_result.fft_grid[1] <= 0 ||
        scf_result.fft_grid[2] <= 0) {
        throw std::runtime_error(
            "The converged SCF result has an invalid FFT grid."
        );
    }
    const Lattice lattice(
        structure.lattice_bohr.col(0),
        structure.lattice_bohr.col(1),
        structure.lattice_bohr.col(2)
    );
    const std::vector<BandPathSample> path =
        interpolate_band_path(lattice, config.bands);

    std::map<std::string, int> species_indices;
    std::vector<UPFData> upfs;
    upfs.reserve(structure.species_order.size());
    for (int species = 0;
         species < static_cast<int>(structure.species_order.size());
         ++species) {
        const std::string& element = structure.species_order[species];
        const auto mapping = config.pseudopotential_paths.find(element);
        if (mapping == config.pseudopotential_paths.end()) {
            throw std::runtime_error(
                "No pseudopotential mapping was provided for element "
                + element + "."
            );
        }
        UPFData upf = read_nc_upf(mapping->second);
        if (upf.header.element != element) {
            throw std::runtime_error(
                "POSCAR element " + element + " is mapped to a UPF for "
                + upf.header.element + "."
            );
        }
        species_indices.emplace(element, species);
        upfs.push_back(std::move(upf));
    }

    std::vector<UPFLocalSpecies> local_species;
    std::vector<UPFNonlocalSpecies> nonlocal_species;
    local_species.reserve(upfs.size());
    nonlocal_species.reserve(upfs.size());
    for (const UPFData& upf : upfs) {
        local_species.push_back(
            prepare_upf_local_species(upf, config.ewald_width_bohr)
        );
        nonlocal_species.push_back(
            prepare_upf_nonlocal_species(upf)
        );
    }
    std::vector<UPFLocalIon> ions;
    ions.reserve(structure.atoms.size());
    for (const StructureAtom& atom : structure.atoms) {
        const auto species = species_indices.find(atom.element);
        if (species == species_indices.end()) {
            throw std::runtime_error(
                "Atom element " + atom.element
                + " was not declared in the POSCAR species line."
            );
        }
        ions.push_back({species->second, atom.frac_position});
    }

    const FFTGrid grid(
        scf_result.fft_grid[0],
        scf_result.fft_grid[1],
        scf_result.fft_grid[2]
    );
    FFTWorkspace fft(grid, config.fft_threads);
    const UPFLocalReciprocalCache local_cache =
        build_upf_local_reciprocal_cache(
            lattice, grid, local_species, fft.thread_count
        );
    const std::vector<double> ionic_potential =
        build_upf_local_potential_real(
            lattice, fft, local_cache, ions
        );

    std::vector<KPointHamiltonian> hamiltonians(path.size());
    const parallel::KPointDistribution distribution(
        static_cast<int>(path.size())
    );
    for (int ik = 0; ik < static_cast<int>(path.size()); ++ik) {
        KPointHamiltonian& point = hamiltonians[ik];
        point.fractional_position = path[ik].frac_position;
        point.weight = 0.0;
        point.basis.generate(
            lattice,
            lattice.B * path[ik].frac_position,
            scf_result.ecut_hartree
        );
        require_fft_grid_for_basis_products(point.basis, grid, ik);
        if (distribution.owns(ik)) {
            point.projectors = build_upf_nonlocal_projectors(
                lattice,
                point.basis,
                nonlocal_species,
                ions,
                fft.thread_count
            );
        }
    }
    const std::vector<std::vector<double>> effective_potentials =
        build_frozen_effective_potentials(
            lattice,
            fft,
            ionic_potential,
            scf_result.scf,
            scf_result.options_used
        );
    return solve_fixed_potential_bands(
        path,
        hamiltonians,
        fft,
        effective_potentials,
        scf_result.options_used,
        scf_result.scf.occupations.mu,
        log_stream
    );
}

void write_band_structure(
    const std::string& path,
    const BandStructureResult& result) {

    if (path.empty()) {
        throw std::runtime_error("The band output path is empty.");
    }
    if (result.path.empty() || result.nbands <= 0 ||
        (result.nspin != 1 && result.nspin != 2)) {
        throw std::runtime_error(
            "Cannot write an incomplete band-structure result."
        );
    }
    const int kpoint_count = static_cast<int>(result.path.size());
    if (static_cast<int>(result.states.size()) !=
        result.nspin * kpoint_count) {
        throw std::runtime_error(
            "Band-state count does not match nspin times path size."
        );
    }

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create band-structure output: " + path
        );
    }
    output << "# PWDFT fixed-potential band structure\n"
        << "# fermi_energy_ha = " << std::scientific
        << std::setprecision(16) << result.fermi_energy_ha << "\n"
        << "# fermi_energy_ev = "
        << result.fermi_energy_ha * HARTREE_TO_EV << "\n"
        << "# path nodes: index distance_bohr^-1 label kx ky kz\n";
    for (int ik = 0; ik < kpoint_count; ++ik) {
        if (result.path[ik].label.empty()) {
            continue;
        }
        output << "# node " << ik << " "
            << result.path[ik].distance_bohr_inverse << " "
            << result.path[ik].label << " "
            << result.path[ik].frac_position[0] << " "
            << result.path[ik].frac_position[1] << " "
            << result.path[ik].frac_position[2] << "\n";
    }
    output
        << "# columns: k_index distance_bohr^-1 kx ky kz spin band "
           "eigenvalue_ha eigenvalue_ev eigenvalue_minus_fermi_ev residual\n";

    for (int spin = 0; spin < result.nspin; ++spin) {
        for (int band = 0; band < result.nbands; ++band) {
            for (int ik = 0; ik < kpoint_count; ++ik) {
                const BandElectronicState& state =
                    result.states[electronic_state_index(
                        spin, ik, kpoint_count
                    )];
                const double eigenvalue = state.eigenvalues[band];
                output << std::setw(6) << ik << " "
                    << std::scientific << std::setprecision(12)
                    << std::setw(20)
                    << result.path[ik].distance_bohr_inverse << " "
                    << std::setw(18)
                    << result.path[ik].frac_position[0] << " "
                    << std::setw(18)
                    << result.path[ik].frac_position[1] << " "
                    << std::setw(18)
                    << result.path[ik].frac_position[2] << " "
                    << std::setw(3) << spin << " "
                    << std::setw(5) << band << " "
                    << std::setw(20) << eigenvalue << " "
                    << std::setw(20)
                    << eigenvalue * HARTREE_TO_EV << " "
                    << std::setw(20)
                    << (eigenvalue - result.fermi_energy_ha)
                        * HARTREE_TO_EV << " "
                    << std::setw(14)
                    << state.residual_norms[band] << "\n";
            }
            output << "\n";
        }
    }
}
