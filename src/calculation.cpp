#include "calculation.hpp"

#include "ewald.hpp"
#include "parallel.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace {

bool is_perdew_zunger_lda(const std::string& functional) {
    std::string uppercase = functional;
    std::transform(
        uppercase.begin(), uppercase.end(), uppercase.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        }
    );
    return uppercase.find("PZ") != std::string::npos;
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("The plane-wave basis is empty.");
    }

    Eigen::Vector3i minimum_frequency = basis.gvectors.front().n;
    Eigen::Vector3i maximum_frequency = basis.gvectors.front().n;
    for (const GVector& gvector : basis.gvectors) {
        minimum_frequency = minimum_frequency.cwiseMin(gvector.n);
        maximum_frequency = maximum_frequency.cwiseMax(gvector.n);
    }

    const Eigen::Vector3i grid_sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_product_frequency =
            maximum_frequency[direction] - minimum_frequency[direction];
        const int largest_unaliased_frequency =
            (grid_sizes[direction] - 1) / 2;
        if (largest_product_frequency > largest_unaliased_frequency) {
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

} // namespace

SinglePointResult run_single_point(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    std::ostream* log_stream,
    const KPointSCFInitialGuess& initial_guess) {

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
    const FFTGrid grid(
        config.fft_grid[0],
        config.fft_grid[1],
        config.fft_grid[2]
    );

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
        if (!is_perdew_zunger_lda(upf.header.functional)) {
            throw std::runtime_error(
                "The " + element + " UPF functional is not PZ-LDA, while "
                "this driver uses LibXC LDA_X + LDA_C_PZ."
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
    FFTWorkspace fft(grid);

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
    if (config.nbands_auto) {
        options.nbands = std::max(
            8,
            static_cast<int>(std::ceil(0.5 * options.nelec)) + 4
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

    std::vector<KPointHamiltonian> kpoint_hamiltonians;
    kpoint_hamiltonians.reserve(config.kpoints.points.size());
    int minimum_plane_wave_count = 0;
    int maximum_plane_wave_count = 0;
    int expanded_projector_count = 0;
    for (int ik = 0;
         ik < static_cast<int>(config.kpoints.points.size());
         ++ik) {
        const KPoint& input_point = config.kpoints.points[ik];
        KPointHamiltonian point;
        point.fractional_position = input_point.frac_position;
        point.weight = input_point.weight;
        const Eigen::Vector3d k_cart =
            lattice.B * input_point.frac_position;
        point.basis.generate(lattice, k_cart, ecut_hartree);
        require_fft_grid_for_basis_products(point.basis, grid);
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

    if (log_stream) {
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
            << "    ENCUT = " << ecut_hartree << " Ha\n"
            << "  KPOINTS = " << config.kpoints.description
            << "    NKPTS = " << config.kpoints.points.size()
            << "    NPROJ(radial) = " << radial_projector_count << "\n"
            << "  EDIFF  = " << options.energy_tolerance
            << "    EDIFFRHO = " << options.density_tolerance << "\n"
            << "  XC     = " << lda_functional_name(options.lda_functional)
            << "    LibXC " << libxc_runtime_version() << "\n"
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

    for (int ik = 0;
         ik < static_cast<int>(kpoint_hamiltonians.size());
         ++ik) {
        KPointHamiltonian& point = kpoint_hamiltonians[ik];
        point.projectors = build_upf_nonlocal_projectors(
            lattice, point.basis, nonlocal_species, upf_ions
        );
        if (ik == 0) {
            expanded_projector_count = static_cast<int>(
                point.projectors.size()
            );
        }
    }

    const std::vector<double> local_potential =
        build_upf_local_potential_real(
            lattice, fft, local_species, upf_ions
        );
    const EwaldEnergyComponents ewald =
        compute_point_ion_ion_ewald_energy(
            lattice,
            grid,
            ewald_ions,
            config.ewald_width_bohr
        );

    KPointSCFResult scf = run_kpoint_scf(
        lattice,
        kpoint_hamiltonians,
        fft,
        local_potential,
        ewald.total,
        options,
        initial_guess,
        log_stream
    );

    const auto density_G = build_density_G(fft, scf.density);
    IonicForceComponents forces;
    forces.local = compute_upf_local_ionic_forces(
        lattice, grid, local_species, upf_ions, density_G
    );
    forces.ion_ion = compute_point_ion_ion_ewald_forces(
        lattice,
        grid,
        ewald_ions,
        config.ewald_width_bohr
    );
    forces.nonlocal.assign(
        structure.atoms.size(), Eigen::Vector3d::Zero()
    );
    const parallel::KPointDistribution distribution(
        static_cast<int>(kpoint_hamiltonians.size())
    );
    std::string local_force_error;
    try {
        for (int ik : distribution.local_kpoints()) {
            if (scf.kpoints[ik].owner_rank != distribution.rank() ||
                scf.kpoints[ik].orbitals.size() == 0) {
                throw std::runtime_error(
                    "The owning rank does not hold orbitals for k point "
                    + std::to_string(ik) + "."
                );
            }
            const auto force_at_k = compute_nonlocal_ionic_forces(
                kpoint_hamiltonians[ik].basis,
                kpoint_hamiltonians[ik].projectors,
                scf.kpoints[ik].orbitals,
                scf.kpoints[ik].occupations,
                static_cast<int>(structure.atoms.size())
            );
            for (int iatom = 0;
                 iatom < static_cast<int>(structure.atoms.size());
                 ++iatom) {
                forces.nonlocal[iatom] +=
                    kpoint_hamiltonians[ik].weight * force_at_k[iatom];
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
    forces.total.resize(structure.atoms.size(), Eigen::Vector3d::Zero());
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        forces.total[iatom] =
            forces.local[iatom]
            + forces.ion_ion[iatom]
            + forces.nonlocal[iatom];
    }

    SinglePointResult result;
    result.converged = scf.converged;
    result.ecut_hartree = ecut_hartree;
    result.plane_wave_count = maximum_plane_wave_count;
    result.radial_projector_count = radial_projector_count;
    result.expanded_projector_count = expanded_projector_count;
    result.ion_ion_energy = ewald.total;
    result.total_valence_charge = automatic_nelect;
    result.options_used = options;
    result.scf = std::move(scf);
    result.forces = std::move(forces);
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
        << result.scf.energy.ion_smooth << "\n\n";

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
