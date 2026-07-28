#include "input.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {

    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual = " + std::to_string(actual)
            + ", expected = " + std::to_string(expected)
        );
    }
}

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string data_path(const std::string& filename) {
    return (std::filesystem::path(TEST_DATA_DIR) / filename).string();
}

} // namespace

int main() {
    try {
        const AtomicStructure direct = read_poscar(
            data_path("POSCAR_direct_selective")
        );
        require_true(direct.comment == "Si H direct/selective parser test",
                     "POSCAR comment was not preserved");
        require_true(direct.species_order.size() == 2,
                     "POSCAR species count mismatch");
        require_true(direct.species_order[0] == "Si" &&
                     direct.species_order[1] == "H",
                     "POSCAR species order mismatch");
        require_true(direct.atoms.size() == 2,
                     "POSCAR atom count mismatch");
        require_true(
            direct.selective_dynamics &&
            !direct.selective_dynamics_cartesian,
            "Direct selective-dynamics coordinate mode mismatch"
        );
        require_close(
            direct.lattice_bohr(0, 0),
            ANGSTROM_TO_BOHR,
            1.0e-14,
            "POSCAR Angstrom-to-Bohr conversion"
        );
        require_close(
            direct.lattice_bohr(1, 1),
            2.0 * ANGSTROM_TO_BOHR,
            1.0e-14,
            "POSCAR second lattice vector"
        );
        require_close(
            direct.atoms[0].frac_position[2],
            0.30,
            1.0e-14,
            "Direct coordinate"
        );
        require_true(
            direct.atoms[0].movable[0] &&
            !direct.atoms[0].movable[1] &&
            direct.atoms[0].movable[2],
            "Selective-dynamics flags mismatch"
        );
        require_true(
            !direct.atoms[1].movable[0] &&
            !direct.atoms[1].movable[1] &&
            !direct.atoms[1].movable[2],
            "Second atom selective-dynamics flags mismatch"
        );

        const AtomicStructure cartesian = read_poscar(
            data_path("POSCAR_cartesian_scaled")
        );
        require_close(
            cartesian.lattice_bohr(0, 0),
            2.0 * ANGSTROM_TO_BOHR,
            1.0e-14,
            "Cartesian POSCAR lattice scale"
        );
        require_close(
            cartesian.lattice_bohr(0, 1),
            0.4 * ANGSTROM_TO_BOHR,
            1.0e-14,
            "POSCAR row-vector to internal column-vector conversion"
        );
        require_close(
            cartesian.atoms[0].frac_position[0],
            0.25,
            1.0e-14,
            "Cartesian POSCAR x conversion"
        );
        require_close(
            cartesian.atoms[0].frac_position[2],
            0.75,
            1.0e-14,
            "Cartesian POSCAR z conversion"
        );

        const AtomicStructure negative = read_poscar(
            data_path("POSCAR_negative_volume")
        );
        const double volume_angstrom3 =
            std::abs(negative.lattice_bohr.determinant())
            / std::pow(ANGSTROM_TO_BOHR, 3);
        require_close(
            volume_angstrom3,
            8.0,
            1.0e-12,
            "Negative POSCAR scale target volume"
        );

        const CalculationConfig config = read_calculation_config(
            data_path("general_scf.in")
        );
        require_true(config.structure_path.find("POSCAR_direct_selective")
                         != std::string::npos,
                     "Relative structure path was not resolved");
        require_true(config.pseudopotential_paths.size() == 2,
                     "Pseudopotential mapping count mismatch");
        require_close(config.ecut_hartree, 10.0, 1.0e-14,
                      "ecut_ha parsing");
        require_true(config.fft_grid[0] == 24 &&
                     config.fft_grid[1] == 26 &&
                     config.fft_grid[2] == 28,
                     "FFT grid parsing mismatch");
        require_true(config.fft_threads == 3,
                     "FFT thread-count parsing mismatch");
        require_true(!config.nbands_auto && config.scf.nbands == 9,
                     "nbands parsing mismatch");
        require_true(
            config.scf.nspin == 2 &&
            std::abs(config.scf.starting_magnetization - 1.0) < 1.0e-14,
            "spin-control parsing mismatch"
        );
        require_true(config.nelect_auto,
                     "automatic nelect parsing mismatch");
        require_true(config.scf.occupation_mode == OccupationMode::FermiDirac,
                     "occupation mode parsing mismatch");
        require_close(
            config.scf.smearing_sigma,
            0.10 / HARTREE_TO_EV,
            1.0e-14,
            "smearing eV-to-Hartree conversion"
        );
        require_true(config.scf.max_iterations == 123,
                     "SCF iteration parsing mismatch");
        require_true(config.scf.eigensolver_max_iterations == 77,
                     "eigensolver iteration parsing mismatch");
        require_close(config.scf.eigensolver_initial_tolerance, 5.0e-8,
                      1.0e-20, "initial eigensolver tolerance parsing");
        require_close(config.scf.eigensolver_tolerance, 4.0e-10,
                      1.0e-22, "final eigensolver tolerance parsing");
        require_true(config.scf.verbosity == SCFVerbosity::Detailed,
                     "verbosity parsing mismatch");

        const CalculationConfig relax_config = read_calculation_config(
            data_path("general_relax.in")
        );
        require_true(
            relax_config.fft_grid == std::array<int, 3>{{0, 0, 0}},
            "fft_grid = auto parsing mismatch"
        );
        require_true(
            relax_config.fft_threads == 0,
            "fft_threads = auto parsing mismatch"
        );
        require_true(
            relax_config.calculation == CalculationType::Relax,
            "relax calculation parsing mismatch"
        );
        require_true(
            relax_config.scf.verbosity == SCFVerbosity::Compact,
            "VASP-style DAV diagnostics should be compact by default"
        );
        require_true(relax_config.relaxation.max_ionic_steps == 37,
                     "max_ionic_steps parsing mismatch");
        require_true(relax_config.relaxation.max_backtracks == 4,
                     "max_backtracks parsing mismatch");
        require_close(
            relax_config.relaxation.force_tolerance_ha_bohr,
            3.0e-4,
            1.0e-16,
            "force tolerance parsing"
        );
        require_close(
            relax_config.relaxation.max_step_angstrom,
            0.075,
            1.0e-16,
            "maximum ionic step parsing"
        );
        require_close(
            relax_config.relaxation.initial_curvature_ha_bohr2,
            0.20,
            1.0e-16,
            "initial BFGS curvature parsing"
        );
        require_close(
            relax_config.scf.energy_tolerance,
            1.0e-6,
            0.0,
            "default VASP-style EDIFF"
        );
        require_true(
            relax_config.relaxation.contcar_path == "test.CONTCAR" &&
            relax_config.relaxation.trajectory_path == "test-relax.xyz",
            "relaxation output-path parsing mismatch"
        );

        const CalculationConfig bands_config = read_calculation_config(
            data_path("general_bands.in")
        );
        require_true(
            bands_config.calculation == CalculationType::Bands,
            "bands calculation parsing mismatch"
        );
        require_true(
            bands_config.bands.path.size() == 3,
            "band-path node count mismatch"
        );
        require_true(
            bands_config.bands.path[0].label == "G" &&
            bands_config.bands.path[1].label == "X" &&
            bands_config.bands.path[2].label == "L",
            "band-path label parsing mismatch"
        );
        require_close(
            bands_config.bands.path[1].frac_position[2],
            0.5,
            1.0e-14,
            "band-path coordinate parsing"
        );
        require_true(
            bands_config.bands.points_per_segment == 7 &&
            bands_config.bands.output_path == "test-bands.dat",
            "band output-control parsing mismatch"
        );

        const CalculationConfig relax_bands_config =
            read_calculation_config(
                data_path("general_relax_bands.in")
            );
        require_true(
            relax_bands_config.calculation ==
                CalculationType::RelaxBands,
            "relax_bands calculation parsing mismatch"
        );
        require_true(
            relax_bands_config.bands.path.size() == 2 &&
            relax_bands_config.bands.points_per_segment == 3,
            "relax_bands path parsing mismatch"
        );

        const KPointSet gamma_mesh = make_uniform_kpoint_mesh(
            {{2, 1, 1}}, true
        );
        require_true(gamma_mesh.points.size() == 2,
                     "Gamma-centered mesh size mismatch");
        require_close(gamma_mesh.points[0].frac_position[0], 0.0, 1.0e-14,
                      "Gamma-centered first point");
        require_close(gamma_mesh.points[1].frac_position[0], -0.5, 1.0e-14,
                      "Gamma-centered boundary point");
        require_close(gamma_mesh.points[0].weight, 0.5, 1.0e-14,
                      "Gamma-centered mesh weight");

        const CalculationConfig mesh_config = read_calculation_config(
            data_path("kpoints_mesh_scf.in")
        );
        require_true(
            mesh_config.fft_threads == 1,
            "omitted fft_threads should preserve the serial default"
        );
        require_true(mesh_config.kpoints.points.size() == 6,
                     "Monkhorst-Pack mesh size mismatch");
        require_close(
            mesh_config.kpoints.points.front().frac_position[0],
            -0.25,
            1.0e-14,
            "Monkhorst-Pack x shift"
        );
        require_close(
            mesh_config.kpoints.points.front().frac_position[1],
            -1.0 / 3.0,
            1.0e-14,
            "Monkhorst-Pack y shift"
        );

        const CalculationConfig explicit_config = read_calculation_config(
            data_path("kpoints_explicit_scf.in")
        );
        require_true(explicit_config.kpoints.points.size() == 2,
                     "Explicit k-point count mismatch");
        require_close(explicit_config.kpoints.points[0].weight, 0.25, 1.0e-14,
                      "First normalized explicit weight");
        require_close(explicit_config.kpoints.points[1].weight, 0.75, 1.0e-14,
                      "Second normalized explicit weight");

        std::cout << "POSCAR and calculation-input tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Input parser test failed: " << error.what() << "\n";
        return 1;
    }
}
