#include "checkpoint.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {

    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

std::string data_path(const std::string& filename) {
    return (
        std::filesystem::path(TEST_DATA_DIR) / filename
    ).string();
}

void expect_validation_failure(
    const SCFCheckpoint& checkpoint,
    const AtomicStructure& structure,
    const CalculationConfig& config,
    double nelect,
    const std::string& message) {

    bool failed = false;
    try {
        validate_scf_checkpoint(
            checkpoint, structure, config, nelect
        );
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed, message);
}

} // namespace

int main() {
    try {
        const AtomicStructure structure = read_poscar(
            data_path("POSCAR_direct_selective")
        );
        CalculationConfig config;
        config.ewald_width_bohr = 0.75;
        config.scf.nspin = 2;
        config.scf.xc_functional =
            XCFunctional::PerdewZunger;
        config.pseudopotential_paths["Si"] =
            data_path("minimal_sp_nc.upf");
        config.pseudopotential_paths["H"] =
            data_path("minimal_local_nc.upf");

        SinglePointResult result;
        result.converged = true;
        result.ecut_hartree = 10.0;
        result.fft_grid = {{4, 4, 4}};
        result.options_used = config.scf;
        result.options_used.nelec = 3.0;
        result.scf.occupations.mu = -0.21;
        result.scf.spin_densities.assign(
            2, std::vector<double>(64, 0.0)
        );
        for (int point = 0; point < 64; ++point) {
            result.scf.spin_densities[0][point] =
                0.010 + 1.0e-5 * point;
            result.scf.spin_densities[1][point] =
                0.007 + 2.0e-5 * point;
        }

        const SCFCheckpoint checkpoint =
            make_scf_checkpoint(structure, config, result);
        const std::filesystem::path output =
            std::filesystem::temp_directory_path()
            / "pwdft-checkpoint-test.dat";
        write_scf_checkpoint(output.string(), checkpoint);
        const SCFCheckpoint restored =
            read_scf_checkpoint(output.string());

        std::ifstream current_input(output);
        std::ostringstream current_buffer;
        current_buffer << current_input.rdbuf();
        std::string legacy_contents = current_buffer.str();
        const std::string version_two = "PWDFT_SCF_CHECKPOINT 2";
        const std::string version_one = "PWDFT_SCF_CHECKPOINT 1";
        const std::size_t version_position =
            legacy_contents.find(version_two);
        const std::size_t functional_position =
            legacy_contents.find("xc_functional");
        require(
            version_position != std::string::npos &&
            functional_position != std::string::npos,
            "Checkpoint v2 fields were not written"
        );
        legacy_contents.replace(
            version_position, version_two.size(), version_one
        );
        legacy_contents.replace(
            functional_position,
            std::string("xc_functional").size(),
            "lda_functional"
        );
        const std::filesystem::path legacy_output =
            std::filesystem::temp_directory_path()
            / "pwdft-checkpoint-test-v1.dat";
        {
            std::ofstream legacy_stream(legacy_output);
            legacy_stream << legacy_contents;
        }
        const SCFCheckpoint restored_legacy =
            read_scf_checkpoint(legacy_output.string());
        std::filesystem::remove(output);
        std::filesystem::remove(legacy_output);

        require(
            restored.structure.species_order ==
                structure.species_order &&
            restored.structure.atoms.size() ==
                structure.atoms.size(),
            "Checkpoint structure metadata did not round-trip"
        );
        require(
            restored.fft_grid ==
                std::array<int, 3>{{4, 4, 4}} &&
            restored.nspin == 2 &&
            restored.spin_densities.size() == 2,
            "Checkpoint electronic dimensions did not round-trip"
        );
        require(
            restored.format_version == 2 &&
            restored.xc_functional == XCFunctional::PerdewZunger,
            "Checkpoint XC metadata did not round-trip"
        );
        require(
            restored_legacy.format_version == 1 &&
            restored_legacy.xc_functional ==
                XCFunctional::PerdewZunger,
            "Legacy PZ-LDA checkpoint was not read"
        );
        require_close(
            restored.fermi_energy_ha,
            -0.21,
            1.0e-16,
            "Checkpoint Fermi energy did not round-trip"
        );
        require_close(
            restored.spin_densities[1][37],
            result.scf.spin_densities[1][37],
            1.0e-17,
            "Checkpoint density did not round-trip"
        );
        validate_scf_checkpoint(
            restored, structure, config, 3.0
        );
        validate_scf_checkpoint(
            restored_legacy, structure, config, 3.0
        );

        AtomicStructure displaced = structure;
        displaced.atoms[0].frac_position[0] += 1.0e-3;
        expect_validation_failure(
            restored,
            displaced,
            config,
            3.0,
            "A changed atomic position was accepted"
        );

        CalculationConfig changed_pseudo = config;
        changed_pseudo.pseudopotential_paths["Si"] =
            data_path("minimal_local_nc.upf");
        expect_validation_failure(
            restored,
            structure,
            changed_pseudo,
            3.0,
            "Changed pseudopotential content was accepted"
        );

        std::cout << "SCF checkpoint tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SCF checkpoint test failed: "
                  << error.what() << "\n";
        return 1;
    }
}
