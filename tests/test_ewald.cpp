#include "ewald.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_less(double value, double tolerance, const std::string& label) {
    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

Ion make_ion(
    const Eigen::Vector3d& fractional_position,
    double charge,
    double sigma) {

    Ion ion;
    ion.frac_position = fractional_position;
    ion.Z = charge;
    ion.sigma = sigma;
    return ion;
}

void wrap_fractional(Eigen::Vector3d& position) {
    for (int a = 0; a < 3; ++a) {
        position[a] -= std::floor(position[a]);
    }
}

void test_width_invariance_and_force_finite_difference() {
    const double cell_length = 9.0;
    const Lattice lattice(
        Eigen::Vector3d(cell_length, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell_length, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell_length)
    );
    const FFTGrid grid(41, 41, 41);
    const std::vector<Ion> reference{
        make_ion(Eigen::Vector3d(0.21, 0.34, 0.43), 1.0, 0.55),
        make_ion(Eigen::Vector3d(0.67, 0.59, 0.76), 1.7, 0.78)
    };

    const EwaldEnergyComponents reference_energy =
        compute_point_ion_ion_ewald_energy(
            lattice, grid, reference, 0.55
        );
    const auto reference_force =
        compute_point_ion_ion_ewald_forces(
            lattice, grid, reference, 0.55
        );

    double max_energy_width_error = 0.0;
    double max_force_width_error = 0.0;
    for (double width : std::vector<double>{0.42, 0.70}) {
        const EwaldEnergyComponents energy =
            compute_point_ion_ion_ewald_energy(
                lattice, grid, reference, width
            );
        const auto force =
            compute_point_ion_ion_ewald_forces(
                lattice, grid, reference, width
            );

        max_energy_width_error = std::max(
            max_energy_width_error,
            std::abs(energy.total - reference_energy.total)
        );
        for (int iion = 0; iion < static_cast<int>(force.size()); ++iion) {
            max_force_width_error = std::max(
                max_force_width_error,
                (force[iion] - reference_force[iion])
                    .cwiseAbs().maxCoeff()
            );
        }
    }

    require_less(
        max_energy_width_error,
        2.0e-11,
        "point-Ewald screening-width energy error"
    );
    require_less(
        max_force_width_error,
        2.0e-11,
        "point-Ewald screening-width force error"
    );
    require_less(
        (reference_force[0] + reference_force[1])
            .cwiseAbs().maxCoeff(),
        2.0e-12,
        "point-Ewald total-force error"
    );

    const double displacement = 1.0e-5;
    Eigen::Vector3d finite_difference = Eigen::Vector3d::Zero();
    for (int direction = 0; direction < 3; ++direction) {
        std::vector<Ion> plus = reference;
        std::vector<Ion> minus = reference;
        Eigen::Vector3d cartesian_step = Eigen::Vector3d::Zero();
        cartesian_step[direction] = displacement;
        plus[0].frac_position += lattice.A_inv * cartesian_step;
        minus[0].frac_position -= lattice.A_inv * cartesian_step;
        wrap_fractional(plus[0].frac_position);
        wrap_fractional(minus[0].frac_position);

        const double plus_energy =
            compute_point_ion_ion_ewald_energy(
                lattice, grid, plus, 0.55
            ).total;
        const double minus_energy =
            compute_point_ion_ion_ewald_energy(
                lattice, grid, minus, 0.55
            ).total;
        finite_difference[direction] =
            -(plus_energy - minus_energy) / (2.0 * displacement);
    }

    require_less(
        (reference_force[0] - finite_difference)
            .cwiseAbs().maxCoeff(),
        3.0e-9,
        "point-Ewald force finite-difference error"
    );
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_width_invariance_and_force_finite_difference();
    } catch (const std::exception& error) {
        std::cerr << "Point-ion Ewald test failed: "
                  << error.what() << "\n";
        return 1;
    }

    std::cout << "All point-ion Ewald tests passed.\n";
    return 0;
}
