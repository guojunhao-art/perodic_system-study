#include "calculation.hpp"

#include "ewald.hpp"
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
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace {

using PerformanceClock = std::chrono::steady_clock;

double elapsed_seconds(PerformanceClock::time_point start) {
    return std::chrono::duration<double>(
        PerformanceClock::now() - start
    ).count();
}

Eigen::Vector3i maximum_required_fft_frequency(
    const PlaneWaveBasis3D& basis) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("The plane-wave basis is empty.");
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

bool has_only_small_prime_factors(int value) {
    for (int factor : {2, 3, 5}) {
        while (value % factor == 0) {
            value /= factor;
        }
    }
    return value == 1;
}

int next_fft_friendly_even_size(int minimum_size) {
    int candidate = std::max(4, minimum_size);
    if (candidate % 2 != 0) {
        ++candidate;
    }
    while (!has_only_small_prime_factors(candidate)) {
        if (candidate > std::numeric_limits<int>::max() - 2) {
            throw std::runtime_error(
                "Automatic FFT grid dimension exceeds integer range."
            );
        }
        candidate += 2;
    }
    return candidate;
}

std::array<int, 3> fft_grid_dimensions_from_required_frequency(
    const Eigen::Vector3i& required_frequency) {

    std::array<int, 3> dimensions{};
    for (int direction = 0; direction < 3; ++direction) {
        if (required_frequency[direction] >
            (std::numeric_limits<int>::max() - 2) / 2) {
            throw std::runtime_error(
                "Automatic FFT grid dimension exceeds integer range."
            );
        }
        const int minimum_size =
            2 * required_frequency[direction] + 2;
        dimensions[direction] =
            next_fft_friendly_even_size(minimum_size);
    }
    return dimensions;
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid) {

    const Eigen::Vector3i required_frequency =
        maximum_required_fft_frequency(basis);
    const Eigen::Vector3i grid_sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_unaliased_frequency =
            (grid_sizes[direction] - 1) / 2;
        if (required_frequency[direction] >
            largest_unaliased_frequency) {
            throw std::runtime_error(
                "fft_grid is too small for products of cutoff wavefunctions "
                "in direction " + std::to_string(direction) + "."
            );
        }
    }
}

double maximum_force_component(
    const std::vector<Eigen::Vector3d>& forces) {

    double maximum = 0.0;
    for (const Eigen::Vector3d& force : forces) {
        maximum = std::max(maximum, force.cwiseAbs().maxCoeff());
    }
    return maximum;
}

bool equivalent_fractional_kpoint(
    const Eigen::Vector3d& first,
    const Eigen::Vector3d& second) {

    const Eigen::Vector3d difference = first - second;
    return (
        difference.array() - difference.array().round()
    ).abs().maxCoeff() <= 1.0e-10;
}

KPointSCFInitialGuess compatible_initial_guess(
    const KPointSCFInitialGuess& initial_guess,
    const KPointSet& kpoints,
    int nspin) {

    KPointSCFInitialGuess result = initial_guess;
    if (result.orbitals.empty()) {
        return result;
    }
    bool compatible =
        result.orbitals.size()
        == static_cast<std::size_t>(nspin) * kpoints.points.size();
    if (compatible && !result.orbital_kpoints.empty()) {
        compatible =
            result.orbital_kpoints.size() == kpoints.points.size();
        for (int point = 0;
             compatible &&
             point < static_cast<int>(kpoints.points.size());
             ++point) {
            compatible = equivalent_fractional_kpoint(
                result.orbital_kpoints[point],
                kpoints.points[point].frac_position
            );
        }
    }
    if (!compatible) {
        result.orbitals.clear();
        result.orbital_kpoints.clear();
    }
    return result;
}

void print_setup_performance(
    std::ostream& out,
    const SetupPerformanceBreakdown& timing) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << std::fixed << std::setprecision(3)
        << "  setup wall time = " << timing.total_seconds << " s\n"
        << "  setup breakdown: UPF/ions = "
        << timing.upf_and_ions_seconds
        << "  basis/FFT = " << timing.basis_and_fft_seconds
        << "  V_NL(projectors) = "
        << timing.nonlocal_projector_seconds
        << "  V_loc(cache+FFT) = "
        << timing.local_potential_seconds
        << "  Ewald = " << timing.ewald_energy_seconds
        << " s\n";
    out.flags(flags);
    out.precision(precision);
    out << std::flush;
}

void print_force_performance(
    std::ostream& out,
    const ForcePerformanceBreakdown& timing) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << std::fixed << std::setprecision(3)
        << "  post-SCF force wall time = "
        << timing.total_seconds << " s\n"
        << "  force breakdown: density FFT = "
        << timing.density_fft_seconds
        << "  local = " << timing.local_seconds
        << "  ion-ion = " << timing.ion_ion_seconds
        << "  nonlocal = " << timing.nonlocal_seconds
        << "  MPI reduction = " << timing.mpi_reduction_seconds
        << " s\n";
    out.flags(flags);
    out.precision(precision);
    out << std::flush;
}

} // namespace

std::array<int, 3> automatic_fft_grid_dimensions(
    const std::vector<KPointHamiltonian>& kpoint_hamiltonians) {

    if (kpoint_hamiltonians.empty()) {
        throw std::runtime_error(
            "Cannot determine an FFT grid without a plane-wave basis."
        );
    }

    Eigen::Vector3i required_frequency = Eigen::Vector3i::Zero();
    for (const KPointHamiltonian& point : kpoint_hamiltonians) {
        required_frequency = required_frequency.cwiseMax(
            maximum_required_fft_frequency(point.basis)
        );
    }

    return fft_grid_dimensions_from_required_frequency(
        required_frequency
    );
}

SinglePointResult run_single_point(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    std::ostream* log_stream,
    const KPointSCFInitialGuess& initial_guess,
    bool print_setup) {

    if (structure.atoms.empty()) {
        throw std::runtime_error("The structure contains no atoms.");
    }
    if (!structure.lattice_bohr.allFinite() ||
        std::abs(structure.lattice_bohr.determinant()) < 1.0e-14) {
        throw std::runtime_error("The structure lattice is invalid.");
    }
    const Lattice lattice(
        structure.lattice_bohr.col(0),
        structure.lattice_bohr.col(1),
        structure.lattice_bohr.col(2)
    );

    KPointReductionResult kpoint_reduction;
    if (config.kpoint_symmetry.enabled) {
        kpoint_reduction = reduce_kpoints_by_symmetry(
            structure,
            config.kpoints,
            config.kpoint_symmetry.tolerance_angstrom,
            config.kpoint_symmetry.include_time_reversal
        );
    } else {
        kpoint_reduction.irreducible_kpoints = config.kpoints;
        kpoint_reduction.full_kpoint_count =
            static_cast<int>(config.kpoints.points.size());
    }
    const KPointSet& scf_kpoints =
        kpoint_reduction.irreducible_kpoints;

    SetupPerformanceBreakdown setup_performance;
    const auto setup_start = PerformanceClock::now();
    auto phase_start = PerformanceClock::now();
    std::vector<UPFData> upfs;
    upfs.reserve(structure.species_order.size());
    std::map<std::string, int> species_indices;
    double recommended_cutoff_hartree = 0.0;
    for (int ispecies = 0;
         ispecies < static_cast<int>(structure.species_order.size());
         ++ispecies) {
        const std::string& element = structure.species_order[ispecies];
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
        if (!pseudopotential_functional_matches_xc(
                upf.header.functional,
                config.scf.xc_functional)) {
            throw std::runtime_error(
                "The " + element + " UPF functional label '"
                + upf.header.functional + "' is incompatible with "
                + xc_functional_name(config.scf.xc_functional) + "."
            );
        }
        recommended_cutoff_hartree = std::max(
            recommended_cutoff_hartree,
            RYDBERG_TO_HARTREE * upf.header.wavefunction_cutoff_ry
        );
        species_indices.emplace(element, ispecies);
        upfs.push_back(std::move(upf));
    }

    const double ecut_hartree = config.ecut_hartree > 0.0
        ? config.ecut_hartree
        : (recommended_cutoff_hartree > 0.0
            ? recommended_cutoff_hartree
            : 10.0);

    std::vector<UPFLocalSpecies> local_species;
    std::vector<UPFNonlocalSpecies> nonlocal_species;
    local_species.reserve(upfs.size());
    nonlocal_species.reserve(upfs.size());
    for (const UPFData& upf : upfs) {
        local_species.push_back(
            prepare_upf_local_species(upf, config.ewald_width_bohr)
        );
        nonlocal_species.push_back(prepare_upf_nonlocal_species(upf));
    }

    std::vector<UPFLocalIon> upf_ions;
    std::vector<Ion> ewald_ions;
    upf_ions.reserve(structure.atoms.size());
    ewald_ions.reserve(structure.atoms.size());
    double automatic_nelect = 0.0;
    int radial_projector_count = 0;
    for (const StructureAtom& atom : structure.atoms) {
        const auto found = species_indices.find(atom.element);
        if (found == species_indices.end()) {
            throw std::runtime_error(
                "Atom element " + atom.element
                + " was not declared in the POSCAR species line."
            );
        }
        const int species_index = found->second;
        upf_ions.push_back({species_index, atom.frac_position});
        radial_projector_count += static_cast<int>(
            upfs[species_index].projectors.size()
        );

        Ion ewald_ion;
        ewald_ion.frac_position = atom.frac_position;
        ewald_ion.Z = local_species[species_index].valence_charge;
        ewald_ion.sigma = config.ewald_width_bohr;
        ewald_ions.push_back(ewald_ion);
        automatic_nelect += ewald_ion.Z;
    }

    SCFOptions options = config.scf;
    if (config.nelect_auto) {
        options.nelec = automatic_nelect;
    }
    if (options.nspin == 2 &&
        std::abs(options.starting_magnetization) >
            options.nelec + 1.0e-12) {
        throw std::runtime_error(
            "starting_magnetization must satisfy |M| <= nelect."
        );
    }
    if (config.nbands_auto) {
        const double majority_electrons = options.nspin == 1
            ? 0.5 * options.nelec
            : 0.5 * (
                options.nelec +
                std::abs(options.starting_magnetization)
            );
        options.nbands = std::max(
            8,
            static_cast<int>(std::ceil(majority_electrons)) + 4
        );
    }
    if (options.occupation_mode == OccupationMode::Fixed &&
        static_cast<int>(options.fixed_occupations.size()) != options.nbands) {
        throw std::runtime_error(
            "fixed_occupations must contain exactly nbands values."
        );
    }
    if (options.occupation_mode == OccupationMode::Fixed) {
        double occupation_sum = 0.0;
        for (double occupation : options.fixed_occupations) {
            occupation_sum += occupation;
        }
        if (std::abs(occupation_sum - options.nelec) > 1.0e-10) {
            throw std::runtime_error(
                "The fixed occupations do not sum to nelect."
            );
        }
    }
    setup_performance.upf_and_ions_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    std::vector<KPointHamiltonian> kpoint_hamiltonians;
    kpoint_hamiltonians.reserve(scf_kpoints.points.size());
    int minimum_plane_wave_count = 0;
    int maximum_plane_wave_count = 0;
    int expanded_projector_count = 0;
    for (int ik = 0;
         ik < static_cast<int>(scf_kpoints.points.size());
         ++ik) {
        const KPoint& input_point = scf_kpoints.points[ik];
        KPointHamiltonian point;
        point.fractional_position = input_point.frac_position;
        point.weight = input_point.weight;
        const Eigen::Vector3d k_cart =
            lattice.B * input_point.frac_position;
        point.basis.generate(lattice, k_cart, ecut_hartree);
        if (point.basis.size() < options.nbands) {
            throw std::runtime_error(
                "The plane-wave basis at k point " + std::to_string(ik)
                + " has fewer vectors than requested bands."
            );
        }
        if (ik == 0) {
            minimum_plane_wave_count = point.basis.size();
            maximum_plane_wave_count = point.basis.size();
        } else {
            minimum_plane_wave_count = std::min(
                minimum_plane_wave_count, point.basis.size()
            );
            maximum_plane_wave_count = std::max(
                maximum_plane_wave_count, point.basis.size()
            );
        }
        kpoint_hamiltonians.push_back(std::move(point));
    }

    const bool automatic_fft_grid =
        config.fft_grid[0] == 0 &&
        config.fft_grid[1] == 0 &&
        config.fft_grid[2] == 0;
    const bool explicit_fft_grid =
        config.fft_grid[0] > 0 &&
        config.fft_grid[1] > 0 &&
        config.fft_grid[2] > 0;
    if (!automatic_fft_grid && !explicit_fft_grid) {
        throw std::runtime_error(
            "fft_grid must be either auto or three positive dimensions."
        );
    }

    std::array<int, 3> fft_dimensions = config.fft_grid;
    if (automatic_fft_grid) {
        Eigen::Vector3i required_frequency = Eigen::Vector3i::Zero();
        for (const KPointHamiltonian& point : kpoint_hamiltonians) {
            required_frequency = required_frequency.cwiseMax(
                maximum_required_fft_frequency(point.basis)
            );
        }
        /*
         * The irreducible representatives are the only Hamiltonians solved,
         * but the FFT box must still cover plane-wave products at every
         * member of every reconstructed star.
         */
        if (kpoint_reduction.reduction_applied) {
            PlaneWaveBasis3D full_mesh_basis;
            for (const KPoint& point : config.kpoints.points) {
                full_mesh_basis.generate(
                    lattice,
                    lattice.B * point.frac_position,
                    ecut_hartree
                );
                required_frequency = required_frequency.cwiseMax(
                    maximum_required_fft_frequency(full_mesh_basis)
                );
            }
        }
        fft_dimensions =
            fft_grid_dimensions_from_required_frequency(
                required_frequency
            );
        fft_dimensions = symmetry_compatible_fft_dimensions(
            fft_dimensions,
            kpoint_reduction.mesh_compatible_operations
        );
    }
    const FFTGrid grid(
        fft_dimensions[0],
        fft_dimensions[1],
        fft_dimensions[2]
    );
    for (const KPointHamiltonian& point : kpoint_hamiltonians) {
        require_fft_grid_for_basis_products(point.basis, grid);
    }
    FFTWorkspace fft(grid, config.fft_threads);
    setup_performance.basis_and_fft_seconds =
        elapsed_seconds(phase_start);

    if (log_stream && print_setup) {
        *log_stream
            << "\n"
            << " PWDFT: POSCAR multi-k-point NC-UPF calculation\n"
            << " -------------------------------------------------------------------------------\n"
            << "  SYSTEM = " << structure.comment << "\n"
            << "  NIONS  = " << structure.atoms.size()
            << "    NTYPES = " << structure.species_order.size()
            << "    VOLUME = " << lattice.volume() << " Bohr^3\n"
            << "  NPLWV(min/max) = " << minimum_plane_wave_count
            << "/" << maximum_plane_wave_count
            << "    NGX = " << grid.n1
            << "    NGY = " << grid.n2
            << "    NGZ = " << grid.n3 << "\n"
            << "  NELECT = " << options.nelec
            << "    NBANDS = " << options.nbands
            << "    NSPIN = " << options.nspin
            << "    ENCUT = " << ecut_hartree << " Ha\n"
            << "  FFT grid = "
            << (automatic_fft_grid ? "automatic" : "explicit")
            << "    FFT threads = " << fft.thread_count << "\n"
            << "  KPOINTS = " << config.kpoints.description
            << "    NKPTS(full/irreducible) = "
            << kpoint_reduction.full_kpoint_count << "/"
            << scf_kpoints.points.size()
            << "    NPROJ(radial) = " << radial_projector_count << "\n"
            << "  SYMMETRY operations(total/mesh) = "
            << kpoint_reduction.space_group_operations.size()
            << "/"
            << kpoint_reduction.mesh_compatible_operations.size()
            << "    time reversal = "
            << (kpoint_reduction.time_reversal_used ? "on" : "off")
            << "\n"
            << "  EDIFF  = " << options.energy_tolerance
            << " Ha (applied to |dE| and |d eps|)\n"
            << "  DAV density reference = "
            << options.density_tolerance
            << "    empty-band tolerance = "
            << options.eigensolver_empty_tolerance << " Ha\n"
            << "  XC     = " << xc_functional_name(options.xc_functional)
            << "    LibXC " << libxc_runtime_version();
        if (options.nspin == 2) {
            *log_stream << "    MAGMOM(start) = "
                << options.starting_magnetization << " mu_B";
        }
        *log_stream << "\n"
            << "\n  k-point list (reciprocal fractional coordinates):\n";
        for (int ik = 0;
             ik < static_cast<int>(kpoint_hamiltonians.size());
             ++ik) {
            const KPointHamiltonian& point = kpoint_hamiltonians[ik];
            *log_stream << "    " << std::setw(4) << ik
                << "  " << std::fixed << std::setprecision(8)
                << std::setw(12) << point.fractional_position[0]
                << std::setw(12) << point.fractional_position[1]
                << std::setw(12) << point.fractional_position[2]
                << "  weight = " << std::setw(12) << point.weight
                << "  NPLWV = " << point.basis.size() << "\n";
        }
        *log_stream << std::flush;
    }

    const parallel::KPointDistribution distribution(
        static_cast<int>(kpoint_hamiltonians.size())
    );
    phase_start = PerformanceClock::now();
    for (int ik : distribution.local_kpoints()) {
        KPointHamiltonian& point = kpoint_hamiltonians[ik];
        point.projectors = build_upf_nonlocal_projectors(
            lattice,
            point.basis,
            nonlocal_species,
            upf_ions,
            fft.thread_count
        );
        if (ik == 0) {
            expanded_projector_count = static_cast<int>(
                point.projectors.size()
            );
        }
    }
    setup_performance.nonlocal_projector_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    const UPFLocalReciprocalCache local_reciprocal_cache =
        build_upf_local_reciprocal_cache(
            lattice,
            grid,
            local_species,
            fft.thread_count
        );
    const std::vector<double> local_potential =
        build_upf_local_potential_real(
            lattice,
            fft,
            local_reciprocal_cache,
            upf_ions
        );
    setup_performance.local_potential_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    const EwaldEnergyComponents ewald =
        compute_point_ion_ion_ewald_energy(
            lattice,
            grid,
            ewald_ions,
            config.ewald_width_bohr
        );
    setup_performance.ewald_energy_seconds =
        elapsed_seconds(phase_start);
    setup_performance.total_seconds =
        elapsed_seconds(setup_start);
    if (log_stream && parallel::is_root()) {
        print_setup_performance(*log_stream, setup_performance);
    }

    const KPointSCFInitialGuess scf_initial_guess =
        compatible_initial_guess(
            initial_guess, scf_kpoints, options.nspin
        );
    KPointSCFResult scf = run_kpoint_scf(
        lattice,
        kpoint_hamiltonians,
        fft,
        local_potential,
        ewald.total,
        options,
        scf_initial_guess,
        log_stream,
        kpoint_reduction.mesh_compatible_operations
    );

    ForcePerformanceBreakdown force_performance;
    const auto force_start = PerformanceClock::now();
    phase_start = PerformanceClock::now();
    const auto density_G = build_density_G(fft, scf.density);
    force_performance.density_fft_seconds =
        elapsed_seconds(phase_start);

    IonicForceComponents forces;
    phase_start = PerformanceClock::now();
    forces.local = compute_upf_local_ionic_forces(
        lattice,
        local_reciprocal_cache,
        upf_ions,
        density_G,
        fft.thread_count
    );
    force_performance.local_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    forces.ion_ion = compute_point_ion_ion_ewald_forces(
        lattice,
        grid,
        ewald_ions,
        config.ewald_width_bohr
    );
    force_performance.ion_ion_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    forces.nonlocal.assign(
        structure.atoms.size(), Eigen::Vector3d::Zero()
    );
    std::string local_force_error;
    try {
        for (int ik : distribution.local_kpoints()) {
            for (int spin = 0; spin < options.nspin; ++spin) {
                const int state =
                    spin * static_cast<int>(
                        kpoint_hamiltonians.size()
                    ) + ik;
                if (scf.kpoints[state].owner_rank !=
                        distribution.rank() ||
                    scf.kpoints[state].orbitals.size() == 0) {
                    throw std::runtime_error(
                        "The owning rank does not hold orbitals for spin "
                        + std::to_string(spin) + ", k point "
                        + std::to_string(ik) + "."
                    );
                }
                const auto force_at_k =
                    compute_nonlocal_ionic_forces(
                        kpoint_hamiltonians[ik].basis,
                        kpoint_hamiltonians[ik].projectors,
                        scf.kpoints[state].orbitals,
                        scf.kpoints[state].occupations,
                        static_cast<int>(structure.atoms.size()),
                        fft.thread_count
                    );
                for (int iatom = 0;
                     iatom < static_cast<int>(structure.atoms.size());
                     ++iatom) {
                    forces.nonlocal[iatom] +=
                        kpoint_hamiltonians[ik].weight *
                        force_at_k[iatom];
                }
            }
        }
    } catch (const std::exception& error) {
        local_force_error =
            "Nonlocal-force assembly failed on MPI rank "
            + std::to_string(distribution.rank()) + ": " + error.what();
    } catch (...) {
        local_force_error =
            "Nonlocal-force assembly failed with an unknown exception on "
            "MPI rank " + std::to_string(distribution.rank()) + ".";
    }
    const std::string force_error = parallel::first_error(local_force_error);
    if (!force_error.empty()) {
        throw std::runtime_error(force_error);
    }
    force_performance.nonlocal_seconds =
        elapsed_seconds(phase_start);

    phase_start = PerformanceClock::now();
    std::vector<double> packed_nonlocal_forces(
        3 * structure.atoms.size(), 0.0
    );
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
            packed_nonlocal_forces[3 * iatom + direction] =
                forces.nonlocal[iatom][direction];
        }
    }
    parallel::sum_in_place(packed_nonlocal_forces);
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
            forces.nonlocal[iatom][direction] =
                packed_nonlocal_forces[3 * iatom + direction];
        }
    }
    force_performance.mpi_reduction_seconds =
        elapsed_seconds(phase_start);

    symmetrize_atomic_vectors(
        structure,
        kpoint_reduction.mesh_compatible_operations,
        forces.local
    );
    symmetrize_atomic_vectors(
        structure,
        kpoint_reduction.mesh_compatible_operations,
        forces.ion_ion
    );
    symmetrize_atomic_vectors(
        structure,
        kpoint_reduction.mesh_compatible_operations,
        forces.nonlocal
    );

    forces.total.resize(structure.atoms.size(), Eigen::Vector3d::Zero());
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        forces.total[iatom] =
            forces.local[iatom]
            + forces.ion_ion[iatom]
            + forces.nonlocal[iatom];
    }
    force_performance.total_seconds =
        elapsed_seconds(force_start);
    if (log_stream && parallel::is_root()) {
        print_force_performance(*log_stream, force_performance);
    }

    SinglePointResult result;
    result.converged = scf.converged;
    result.ecut_hartree = ecut_hartree;
    result.fft_grid = fft_dimensions;
    result.plane_wave_count = maximum_plane_wave_count;
    result.radial_projector_count = radial_projector_count;
    result.expanded_projector_count = expanded_projector_count;
    result.full_kpoint_count = kpoint_reduction.full_kpoint_count;
    result.irreducible_kpoint_count =
        static_cast<int>(scf_kpoints.points.size());
    result.space_group_operation_count =
        static_cast<int>(
            kpoint_reduction.space_group_operations.size()
        );
    result.mesh_symmetry_operation_count =
        static_cast<int>(
            kpoint_reduction.mesh_compatible_operations.size()
        );
    result.kpoint_time_reversal_used =
        kpoint_reduction.time_reversal_used;
    result.ion_ion_energy = ewald.total;
    result.total_valence_charge = automatic_nelect;
    result.options_used = options;
    result.scf = std::move(scf);
    result.forces = std::move(forces);
    result.setup_performance = setup_performance;
    result.force_performance = force_performance;
    return result;
}

void print_single_point_result(
    std::ostream& out,
    const AtomicStructure& structure,
    const SinglePointResult& result) {

    const double energy_without_entropy =
        result.scf.energy.total + result.scf.energy.ion_smooth;
    const double sigma0_energy =
        result.scf.energy.sigma0_estimate + result.scf.energy.ion_smooth;

    out << "\n"
        << "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (Ha)\n"
        << "  ---------------------------------------------------\n"
        << std::setprecision(12) << std::scientific
        << "  free  energy   TOTEN  = " << std::setw(20)
        << result.scf.variational_energy << "\n"
        << "  energy  without entropy = " << std::setw(18)
        << energy_without_entropy
        << "  energy(sigma->0) = " << std::setw(18)
        << sigma0_energy << "\n"
        << "  E_NL = " << std::setw(20)
        << result.scf.energy.nonlocal
        << "  E_II(Ewald) = " << std::setw(20)
        << result.scf.energy.ion_smooth << "\n";
    if (result.options_used.nspin == 2) {
        out << "  number of electron  up/down = "
            << result.scf.spin_electron_counts[0] << " / "
            << result.scf.spin_electron_counts[1] << "\n"
            << "  total magnetization = "
            << result.scf.magnetization << " mu_B\n";
    }
    out << "\n";

    out << "  POSITION (Bohr)" << std::setw(45)
        << "TOTAL-FORCE (Ha/Bohr)\n"
        << "  -----------------------------------------------------------------------------------\n";
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        const Eigen::Vector3d position =
            structure.lattice_bohr * structure.atoms[iatom].frac_position;
        out << "  " << std::fixed << std::setprecision(8)
            << std::setw(14) << position[0]
            << std::setw(14) << position[1]
            << std::setw(14) << position[2]
            << "    "
            << std::setw(14) << result.forces.total[iatom][0]
            << std::setw(14) << result.forces.total[iatom][1]
            << std::setw(14) << result.forces.total[iatom][2]
            << "\n";
    }
    out << "  -----------------------------------------------------------------------------------\n"
        << "  force split max |F_loc| / |F_II| / |F_NL| = "
        << std::scientific << std::setprecision(5)
        << maximum_force_component(result.forces.local) << " / "
        << maximum_force_component(result.forces.ion_ion) << " / "
        << maximum_force_component(result.forces.nonlocal)
        << " Ha/Bohr\n"
        << "  reached required accuracy: "
        << (result.converged
            ? "electronic convergence achieved"
            : "no")
        << "\n";
}
