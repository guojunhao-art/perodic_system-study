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
        require_true(!config.nbands_auto && config.scf.nbands == 9,
                     "nbands parsing mismatch");
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
        require_true(config.scf.verbosity == SCFVerbosity::Detailed,
                     "verbosity parsing mismatch");

        std::cout << "POSCAR and calculation-input tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Input parser test failed: " << error.what() << "\n";
        return 1;
    }
}
