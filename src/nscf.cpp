#include "nscf.hpp"

#include "parallel.hpp"
#include "symmetry.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace {

Eigen::Vector3i maximum_required_fft_frequency(
    const PlaneWaveBasis3D& basis) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("The NSCF plane-wave basis is empty.");
    }
    Eigen::Vector3i minimum = basis.gvectors.front().n;
    Eigen::Vector3i maximum = basis.gvectors.front().n;
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
    const Eigen::Vector3i dimensions(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        if (required[direction] >
            (dimensions[direction] - 1) / 2) {
            throw std::runtime_error(
                "The checkpoint FFT grid is too small for NSCF k point "
                + std::to_string(kpoint_index) + " in direction "
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
    for (const std::vector<double>& density : spin_densities) {
        if (density.size() != total.size()) {
            throw std::runtime_error(
                "NSCF spin-density grids have inconsistent sizes."
            );
        }
        for (std::size_t point = 0; point < total.size(); ++point) {
            total[point] += density[point];
        }
    }
    return total;
}

std::vector<std::vector<double>> build_fixed_effective_potentials(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    const std::vector<std::vector<double>>& spin_densities,
    const SCFOptions& options) {

    if (static_cast<int>(spin_densities.size()) != options.nspin) {
        throw std::runtime_error(
            "Checkpoint spin-density channels do not match NSCF nspin."
        );
    }
    const double dV = lattice.volume()
        / static_cast<double>(fft.grid.ngrid);
    const std::vector<double> total_density =
        sum_spin_densities(spin_densities);
    const std::vector<double> hartree =
        build_hartree_potential(lattice, fft, total_density);
    LibXCLDAFunctional xc(
        options.lda_functional, options.nspin
    );
    std::vector<std::vector<double>> potentials(options.nspin);
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
        double majority_electrons = 0.5 * options.nelec;
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
            static_cast<int>(std::ceil(majority_electrons)) + 4
        );
    }
    if (options.occupation_mode == OccupationMode::Fixed &&
        static_cast<int>(options.fixed_occupations.size()) !=
            options.nbands) {
        throw std::runtime_error(
            "fixed_occupations must contain exactly nbands values."
        );
    }
    return options;
}

double maximum_residual(const BandStructureResult& result) {
    double maximum = 0.0;
    for (const BandElectronicState& state : result.states) {
        for (double residual : state.residual_norms) {
            maximum = std::max(maximum, residual);
        }
    }
    return maximum;
}

} // namespace

NSCFResult run_fixed_density_nscf(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SCFCheckpoint& checkpoint,
    std::ostream* log_stream) {

    const auto start = std::chrono::steady_clock::now();
    if (config.calculation != CalculationType::NSCF) {
        throw std::runtime_error(
            "run_fixed_density_nscf requires calculation = nscf."
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
         species < static_cast<int>(structure.species_order.size());
         ++species) {
        const std::string& element =
            structure.species_order[species];
        const auto mapping =
            config.pseudopotential_paths.find(element);
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
            prepare_upf_local_species(
                upf, config.ewald_width_bohr
            )
        );
        nonlocal_species.push_back(
            prepare_upf_nonlocal_species(upf)
        );
    }

    std::vector<UPFLocalIon> ions;
    ions.reserve(structure.atoms.size());
    double automatic_nelect = 0.0;
    for (const StructureAtom& atom : structure.atoms) {
        const auto species = species_indices.find(atom.element);
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
            local_species[species->second].valence_charge;
    }
    const double resolved_nelect = config.nelect_auto
        ? automatic_nelect
        : config.scf.nelec;
    validate_scf_checkpoint(
        checkpoint, structure, config, resolved_nelect
    );
    SCFOptions options = resolve_nscf_options(
        config, resolved_nelect, checkpoint, lattice
    );

    KPointReductionResult reduction;
    if (config.kpoint_symmetry.enabled) {
        reduction = reduce_kpoints_by_symmetry(
            structure,
            config.kpoints,
            config.kpoint_symmetry.tolerance_angstrom,
            config.kpoint_symmetry.include_time_reversal
        );
    } else {
        reduction.irreducible_kpoints = config.kpoints;
        reduction.full_kpoint_count =
            static_cast<int>(config.kpoints.points.size());
    }
    const KPointSet& kpoints =
        reduction.irreducible_kpoints;
    const FFTGrid grid(
        checkpoint.fft_grid[0],
        checkpoint.fft_grid[1],
        checkpoint.fft_grid[2]
    );
    FFTWorkspace fft(grid, config.fft_threads);

    std::vector<KPointHamiltonian> hamiltonians(
        kpoints.points.size()
    );
    const parallel::KPointDistribution distribution(
        static_cast<int>(kpoints.points.size())
    );
    for (int point = 0;
         point < static_cast<int>(kpoints.points.size());
         ++point) {
        hamiltonians[point].fractional_position =
            kpoints.points[point].frac_position;
        hamiltonians[point].weight =
            kpoints.points[point].weight;
        hamiltonians[point].basis.generate(
            lattice,
            lattice.B * kpoints.points[point].frac_position,
            checkpoint.ecut_hartree
        );
        if (hamiltonians[point].basis.size() < options.nbands) {
            throw std::runtime_error(
                "The NSCF plane-wave basis at k point "
                + std::to_string(point)
                + " has fewer vectors than requested bands."
            );
        }
        require_checkpoint_grid_for_basis(
            hamiltonians[point].basis, grid, point
        );
        if (distribution.owns(point)) {
            hamiltonians[point].projectors =
                build_upf_nonlocal_projectors(
                    lattice,
                    hamiltonians[point].basis,
                    nonlocal_species,
                    ions,
                    fft.thread_count
                );
        }
    }

    const UPFLocalReciprocalCache local_cache =
        build_upf_local_reciprocal_cache(
            lattice, grid, local_species, fft.thread_count
        );
    const std::vector<double> ionic_potential =
        build_upf_local_potential_real(
            lattice, fft, local_cache, ions
        );
    const std::vector<std::vector<double>> effective_potentials =
        build_fixed_effective_potentials(
            lattice,
            fft,
            ionic_potential,
            checkpoint.spin_densities,
            options
        );

    std::vector<BandPathSample> samples(kpoints.points.size());
    for (int point = 0;
         point < static_cast<int>(samples.size());
         ++point) {
        samples[point].frac_position =
            kpoints.points[point].frac_position;
    }
    BandStructureResult diagonalization =
        solve_fixed_potential_bands(
            samples,
            hamiltonians,
            fft,
            effective_potentials,
            options,
            checkpoint.fermi_energy_ha,
            nullptr
        );

    const int kpoint_count =
        static_cast<int>(kpoints.points.size());
    const int state_count = options.nspin * kpoint_count;
    std::vector<Eigen::VectorXd> eigenvalues(state_count);
    std::vector<double> weights(state_count, 0.0);
    for (int spin = 0; spin < options.nspin; ++spin) {
        for (int point = 0; point < kpoint_count; ++point) {
            const int state =
                electronic_state_index(spin, point, kpoint_count);
            eigenvalues[state] =
                diagonalization.states[state].eigenvalues;
            weights[state] = kpoints.points[point].weight;
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

    KPointSCFResult electronic;
    electronic.converged = diagonalization.converged;
    electronic.iterations = 1;
    electronic.occupations = occupations;
    electronic.spin_densities = checkpoint.spin_densities;
    electronic.density =
        sum_spin_densities(checkpoint.spin_densities);
    const double dV = lattice.volume()
        / static_cast<double>(grid.ngrid);
    electronic.electron_number_from_density =
        electron_number_from_density(electronic.density, dV);
    if (std::abs(
            electronic.electron_number_from_density
            - options.nelec) > 1.0e-7) {
        throw std::runtime_error(
            "Checkpoint density integral does not match nelect."
        );
    }
    electronic.kpoints.resize(state_count);
    for (int spin = 0; spin < options.nspin; ++spin) {
        for (int point = 0; point < kpoint_count; ++point) {
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
            state.weight = kpoints.points[point].weight;
            state.owner_rank = distribution.owner(point);
            state.eigenvalues = eigenvalues[state_index];
            state.occupations =
                occupations.occupations[state_index];
            electronic.spin_electron_counts[spin] +=
                state.weight * std::accumulate(
                    state.occupations.begin(),
                    state.occupations.end(),
                    0.0
                );
        }
    }
    if (options.nspin == 2) {
        electronic.magnetization =
            electronic.spin_electron_counts[0]
            - electronic.spin_electron_counts[1];
    }

    NSCFResult result;
    result.converged = diagonalization.converged;
    result.full_kpoint_count = reduction.full_kpoint_count;
    result.irreducible_kpoint_count = kpoint_count;
    result.space_group_operation_count = static_cast<int>(
        reduction.space_group_operations.size()
    );
    result.mesh_symmetry_operation_count = static_cast<int>(
        reduction.mesh_compatible_operations.size()
    );
    result.kpoint_time_reversal_used =
        reduction.time_reversal_used;
    result.checkpoint_fermi_energy_ha =
        checkpoint.fermi_energy_ha;
    result.options_used = options;
    result.electronic = std::move(electronic);
    result.diagonalization = std::move(diagonalization);
    const double local_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    result.wall_time_seconds = parallel::maximum(local_wall_time);

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
    out << "\n FIXED-DENSITY NSCF\n"
        << " -------------------------------------------------------------------------------\n"
        << "  NKPTS(full/irreducible) = "
        << result.full_kpoint_count << "/"
        << result.irreducible_kpoint_count
        << "    NBANDS = " << result.options_used.nbands
        << "    NSPIN = " << result.options_used.nspin << "\n"
        << std::scientific << std::setprecision(12)
        << "  checkpoint E_F = "
        << result.checkpoint_fermi_energy_ha
        << " Ha    NSCF E_F = "
        << result.electronic.occupations.mu << " Ha\n"
        << "  electron count(eigenvalues/density) = "
        << result.electronic.occupations.nelec_sum << " / "
        << result.electronic.electron_number_from_density << "\n"
        << "  maximum Davidson residual = "
        << maximum_residual(result.diagonalization) << "\n"
        << std::fixed << std::setprecision(3)
        << "  NSCF wall time"
        << (parallel::size() > 1 ? "(max-rank)" : "")
        << " = " << result.wall_time_seconds << " s\n";
    out.flags(flags);
    out.precision(precision);
}
