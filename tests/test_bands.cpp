#include "bands.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
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

        std::vector<KPointHamiltonian> hamiltonians(path.size());
        for (int ik = 0; ik < static_cast<int>(path.size()); ++ik) {
            hamiltonians[ik].fractional_position =
                path[ik].frac_position;
            hamiltonians[ik].weight = 0.0;
            hamiltonians[ik].basis.generate(
                lattice,
                lattice.B * path[ik].frac_position,
                1.0
            );
        }

        const FFTGrid grid(8, 8, 8);
        FFTWorkspace fft(grid, 1);
        const std::vector<std::vector<double>> effective_potentials(
            1, std::vector<double>(grid.ngrid, 0.0)
        );
        SCFOptions options;
        options.nbands = 2;
        options.nspin = 1;
        options.eigensolver_max_iterations = 20;
        options.eigensolver_max_subspace = 8;
        options.eigensolver_tolerance = 1.0e-11;
        options.eigensolver_denom_floor = 1.0e-8;

        const double fermi_energy = 0.02;
        const BandStructureResult bands =
            solve_fixed_potential_bands(
                path,
                hamiltonians,
                fft,
                effective_potentials,
                options,
                fermi_energy
            );
        require_true(bands.converged, "Free-electron band solve");
        require_true(
            bands.states.size() == path.size(),
            "Free-electron band-state count"
        );
        require_close(
            bands.fermi_energy_ha,
            fermi_energy,
            0.0,
            "SCF Fermi reference propagation"
        );
        for (int ik = 0; ik < static_cast<int>(path.size()); ++ik) {
            const double expected =
                0.5 * path[ik].frac_position.squaredNorm();
            require_close(
                bands.states[ik].eigenvalues[0],
                expected,
                1.0e-11,
                "Lowest free-electron band"
            );
            require_true(
                bands.states[ik].residual_norms[0]
                    <= options.eigensolver_tolerance,
                "Free-electron Davidson residual"
            );
        }

        std::cout
            << "Band-path interpolation and fixed-potential NSCF tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Band-structure test failed: "
            << error.what() << "\n";
        return 1;
    }
}
