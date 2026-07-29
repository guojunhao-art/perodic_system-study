#include "dos.hpp"
#include "nscf.hpp"
#include "parallel.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

AtomicStructure hydrogen_structure() {
    AtomicStructure structure;
    structure.comment = "fixed-density NSCF integration test";
    structure.lattice_bohr =
        10.0 * Eigen::Matrix3d::Identity();
    structure.species_order = {"H"};
    StructureAtom atom;
    atom.element = "H";
    atom.frac_position = Eigen::Vector3d::Zero();
    structure.atoms.push_back(atom);
    return structure;
}

} // namespace

int main(int argc, char** argv) {
    try {
        parallel::Environment environment(argc, argv);
        const AtomicStructure structure = hydrogen_structure();
        CalculationConfig config;
        config.calculation = CalculationType::NSCF;
        config.structure_path = "unused";
        config.pseudopotential_paths["H"] =
            (std::filesystem::path(TEST_DATA_DIR)
             / "minimal_local_nc.upf").string();
        config.ecut_hartree = 1.0;
        config.fft_grid = {{0, 0, 0}};
        config.ewald_width_bohr = 0.70;
        config.nelect_auto = true;
        config.nbands_auto = false;
        config.scf.nbands = 1;
        config.scf.nspin = 1;
        config.scf.occupation_mode =
            OccupationMode::DegeneracyAwareZeroT;
        config.scf.eigensolver_max_iterations = 100;
        config.scf.eigensolver_tolerance = 1.0e-8;
        config.kpoints =
            make_uniform_kpoint_mesh({{2, 1, 1}}, true);
        config.kpoint_symmetry.enabled = false;
        config.pdos.output_path = "unused-pdos.dat";

        SCFCheckpoint checkpoint;
        checkpoint.structure = structure;
        checkpoint.ecut_hartree = 1.0;
        checkpoint.fft_grid = {{12, 12, 12}};
        checkpoint.nspin = 1;
        checkpoint.nelect = 1.0;
        checkpoint.lda_functional =
            LDAFunctional::PerdewZunger;
        checkpoint.ewald_width_bohr = 0.70;
        checkpoint.fermi_energy_ha = -0.10;
        checkpoint.pseudopotentials["H"] =
            fingerprint_file(
                config.pseudopotential_paths["H"]
            );
        const double uniform_density =
            checkpoint.nelect
            / std::abs(structure.lattice_bohr.determinant());
        checkpoint.spin_densities.assign(
            1,
            std::vector<double>(
                12 * 12 * 12, uniform_density
            )
        );

        const NSCFResult result = run_fixed_density_nscf(
            structure, config, checkpoint
        );
        require(result.converged,
                "Fixed-density NSCF did not converge");
        require(
            result.full_kpoint_count == 2 &&
            result.irreducible_kpoint_count == 2 &&
            result.electronic.kpoints.size() == 2,
            "Fixed-density NSCF returned the wrong k-point count"
        );
        require(
            std::abs(
                result.electronic.occupations.nelec_sum - 1.0
            ) < 1.0e-12,
            "NSCF occupations do not integrate to one electron"
        );
        require(
            std::abs(
                result.electronic.electron_number_from_density
                - 1.0
            ) < 1.0e-12,
            "NSCF changed the checkpoint density integral"
        );
        require(
            std::isfinite(
                result.electronic.occupations.mu
            ),
            "NSCF Fermi energy is not finite"
        );
        require(
            result.projection.orbitals.size() == 1 &&
            result.projection.states.size() == 2,
            "NSCF did not retain the Löwdin PDOS projection weights"
        );

        DensityOfStatesOptions dos_options;
        dos_options.points = 101;
        dos_options.smearing_ev = 0.20;
        const DensityOfStatesResult dos =
            compute_density_of_states(
                result.electronic,
                result.options_used,
                dos_options
            );
        require(
            dos.kpoint_count == 2 &&
            dos.samples.back().integrated_total > 1.99,
            "NSCF eigenvalues were not usable by the DOS module"
        );
        const ProjectedDensityOfStatesResult pdos =
            compute_projected_density_of_states(
                result.electronic,
                result.options_used,
                dos,
                result.projection
            );
        require(
            pdos.channels.size() == 1 &&
            pdos.occupied_spilling >= 0.0 &&
            pdos.occupied_spilling <= 1.0,
            "NSCF projection weights were not usable by the PDOS module"
        );

        if (parallel::is_root()) {
            std::cout
                << "Fixed-density NSCF tests passed with "
                << parallel::size() << " rank(s).\n";
        }
        return 0;
    } catch (const std::exception& error) {
        if (parallel::is_root()) {
            std::cerr << "Fixed-density NSCF test failed: "
                      << error.what() << "\n";
        }
        return 1;
    }
}
