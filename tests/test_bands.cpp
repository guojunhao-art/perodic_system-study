#include "bands.hpp"
#include "nscf.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double value,
    double expected,
    double tolerance,
    const std::string& message) {

    if (std::abs(value - expected) > tolerance) {
        throw std::runtime_error(
            message + ": value = " + std::to_string(value)
            + ", expected = " + std::to_string(expected)
        );
    }
}

} // namespace

int main() {
    try {
        /*
         * A = 2 pi I gives B = I, so reciprocal fractional and Cartesian
         * coordinates have the same numerical values and path distances are
         * especially transparent.
         */
        const Eigen::Vector3d a1(2.0 * M_PI, 0.0, 0.0);
        const Eigen::Vector3d a2(0.0, 2.0 * M_PI, 0.0);
        const Eigen::Vector3d a3(0.0, 0.0, 2.0 * M_PI);
        const Lattice lattice(a1, a2, a3);

        BandStructureOptions path_options;
        path_options.points_per_segment = 3;
        path_options.path = {
            {"G", Eigen::Vector3d(0.0, 0.0, 0.0)},
            {"X", Eigen::Vector3d(0.25, 0.0, 0.0)},
            {"L", Eigen::Vector3d(0.25, 0.25, 0.0)}
        };
        const std::vector<BandPathSample> path =
            interpolate_band_path(lattice, path_options);
        require_true(path.size() == 5, "Interpolated path size");
        require_close(
            path[0].distance_bohr_inverse, 0.0, 1.0e-14,
            "Gamma path distance"
        );
        require_close(
            path[2].distance_bohr_inverse, 0.25, 1.0e-14,
            "X path distance"
        );
        require_close(
            path[4].distance_bohr_inverse, 0.50, 1.0e-14,
            "L path distance"
        );
        require_true(
            path[0].label == "G" &&
            path[2].label == "X" &&
            path[4].label == "L",
            "Band-path node labels"
        );
        require_true(
            path[1].label.empty() && path[3].label.empty(),
            "Interior band-path labels"
        );

        NSCFResult bands;
        bands.converged = true;
        bands.band_path = true;
        bands.path = path;
        bands.diagonalization.converged = true;
        bands.diagonalization.nspin = 1;
        bands.diagonalization.nbands = 2;
        bands.diagonalization.fermi_energy_ha = 0.02;
        bands.diagonalization.states.resize(path.size());
        for (int ik = 0;
             ik < static_cast<int>(path.size());
             ++ik) {
            NSCFElectronicState& state =
                bands.diagonalization.states[ik];
            state.spin_channel = 0;
            state.kpoint_index = ik;
            state.eigenvalues.resize(2);
            state.eigenvalues[0] =
                0.5 * path[ik].frac_position.squaredNorm();
            state.eigenvalues[1] =
                state.eigenvalues[0] + 0.5;
            state.residual_norms = {1.0e-10, 2.0e-7};
        }

        const std::filesystem::path output_path =
            std::filesystem::temp_directory_path()
            / "pwdft-test-bands.dat";
        write_band_structure(
            output_path.string(), bands
        );
        std::ifstream output(output_path);
        std::ostringstream text;
        text << output.rdbuf();
        require_true(
            text.str().find(
                "fixed-density NSCF band structure"
            ) != std::string::npos &&
            text.str().find("# node 2") !=
                std::string::npos,
            "NSCF band output header or nodes"
        );
        std::filesystem::remove(output_path);

        bands.projection.orbitals = {
            {0, "Si", 0, "3S", 0, 0},
            {0, "Si", 1, "3P", 1, 1}
        };
        bands.projection.minimum_overlap_eigenvalue = 0.8;
        bands.projection.maximum_overlap_eigenvalue = 1.2;
        bands.projection.maximum_orthonormality_error = 1.0e-13;
        bands.projection.states.resize(path.size());
        for (int ik = 0;
             ik < static_cast<int>(path.size());
             ++ik) {
            AtomicProjectionState& projected =
                bands.projection.states[ik];
            projected.spin_channel = 0;
            projected.kpoint_index = ik;
            projected.weights.resize(2, 2);
            projected.weights <<
                0.7, 0.2,
                0.1, 0.6;
        }
        const std::filesystem::path fat_output_path =
            std::filesystem::temp_directory_path()
            / "pwdft-test-fatbands.dat";
        write_fat_band_structure(
            fat_output_path.string(), bands
        );
        std::ifstream fat_output(fat_output_path);
        std::ostringstream fat_text;
        fat_text << fat_output.rdbuf();
        require_true(
            fat_text.str().find(
                "Löwdin projected fixed-density NSCF bands"
            ) != std::string::npos &&
            fat_text.str().find("\"px\"") !=
                std::string::npos &&
            fat_text.str().find("\"cos\"") !=
                std::string::npos &&
            fat_text.str().find("state_projection_sum") !=
                std::string::npos,
            "Fat-band output header or real-harmonic labels"
        );
        std::filesystem::remove(fat_output_path);

        std::cout
            << "Band-path, NSCF, and fat-band output tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Band-structure test failed: "
            << error.what() << "\n";
        return 1;
    }
}
