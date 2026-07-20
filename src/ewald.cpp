#include "ewald.hpp"

#include "forces.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

void validate_ewald_inputs(
    const std::vector<Ion>& ions,
    double gaussian_width_bohr,
    double real_space_cutoff_factor) {

    if (!std::isfinite(gaussian_width_bohr) ||
        gaussian_width_bohr <= 0.0) {
        throw std::runtime_error(
            "Ewald Gaussian width must be positive and finite."
        );
    }
    if (!std::isfinite(real_space_cutoff_factor) ||
        real_space_cutoff_factor <= 0.0) {
        throw std::runtime_error(
            "Ewald real-space cutoff factor must be positive and finite."
        );
    }

    for (const Ion& ion : ions) {
        if (!std::isfinite(ion.Z) || ion.Z <= 0.0) {
            throw std::runtime_error(
                "Ewald ion charge must be positive and finite."
            );
        }
        if (!ion.frac_position.allFinite()) {
            throw std::runtime_error(
                "Ewald ion position must be finite."
            );
        }
    }
}

int lattice_image_bound(
    const Lattice& lattice,
    double gaussian_width_bohr,
    double real_space_cutoff_factor) {

    const double real_space_cutoff =
        2.0 * real_space_cutoff_factor * gaussian_width_bohr;
    const double cell_diameter_bound =
        lattice.A.col(0).norm()
        + lattice.A.col(1).norm()
        + lattice.A.col(2).norm();
    const double smallest_stretch =
        lattice.A.jacobiSvd().singularValues().minCoeff();

    if (!std::isfinite(smallest_stretch) || smallest_stretch <= 0.0) {
        throw std::runtime_error(
            "Ewald lattice has an invalid smallest singular value."
        );
    }

    return static_cast<int>(std::ceil(
        (real_space_cutoff + cell_diameter_bound) / smallest_stretch
    ));
}

bool is_zero_image(int n1, int n2, int n3) {
    return n1 == 0 && n2 == 0 && n3 == 0;
}

} // namespace

EwaldEnergyComponents compute_point_ion_ion_ewald_energy(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    double gaussian_width_bohr,
    double real_space_cutoff_factor) {

    validate_ewald_inputs(
        ions,
        gaussian_width_bohr,
        real_space_cutoff_factor
    );

    EwaldEnergyComponents result;
    if (ions.empty()) {
        return result;
    }

    std::vector<Ion> gaussian_ions = ions;
    for (Ion& ion : gaussian_ions) {
        ion.sigma = gaussian_width_bohr;
    }
    result.reciprocal_gaussian = compute_smooth_ion_ion_energy(
        lattice,
        grid,
        gaussian_ions
    );

    double total_charge = 0.0;
    for (const Ion& ion : ions) {
        result.gaussian_self +=
            ion.Z * ion.Z
            / (2.0 * std::sqrt(M_PI) * gaussian_width_bohr);
        total_charge += ion.Z;
    }
    result.neutralizing_background =
        -2.0 * M_PI * gaussian_width_bohr * gaussian_width_bohr
        * total_charge * total_charge / lattice.volume();

    const int image_bound = lattice_image_bound(
        lattice,
        gaussian_width_bohr,
        real_space_cutoff_factor
    );
    const double width = 2.0 * gaussian_width_bohr;

    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        const Eigen::Vector3d first_position =
            lattice.cart_from_frac(ions[iion].frac_position);

        for (int jion = 0; jion < static_cast<int>(ions.size()); ++jion) {
            const Eigen::Vector3d second_position =
                lattice.cart_from_frac(ions[jion].frac_position);
            for (int n1 = -image_bound; n1 <= image_bound; ++n1) {
                for (int n2 = -image_bound; n2 <= image_bound; ++n2) {
                    for (int n3 = -image_bound; n3 <= image_bound; ++n3) {
                        if (iion == jion && is_zero_image(n1, n2, n3)) {
                            continue;
                        }

                        const Eigen::Vector3d translation =
                            lattice.A * Eigen::Vector3d(n1, n2, n3);
                        const double distance =
                            (first_position - second_position + translation)
                                .norm();
                        const double scaled_distance = distance / width;

                        if (distance < 1.0e-14 ||
                            scaled_distance > real_space_cutoff_factor) {
                            continue;
                        }

                        result.real_space_correction +=
                            0.5 * ions[iion].Z * ions[jion].Z
                            * std::erfc(scaled_distance)
                            / distance;
                    }
                }
            }
        }
    }

    result.total =
        result.reciprocal_gaussian
        + result.real_space_correction
        - result.gaussian_self
        + result.neutralizing_background;
    return result;
}

std::vector<Eigen::Vector3d> compute_point_ion_ion_ewald_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    double gaussian_width_bohr,
    double real_space_cutoff_factor) {

    validate_ewald_inputs(
        ions,
        gaussian_width_bohr,
        real_space_cutoff_factor
    );
    std::vector<Ion> gaussian_ions = ions;
    for (Ion& ion : gaussian_ions) {
        ion.sigma = gaussian_width_bohr;
    }
    std::vector<Eigen::Vector3d> forces =
        compute_smooth_ion_ion_forces(lattice, grid, gaussian_ions);

    if (ions.empty()) {
        return forces;
    }

    const int image_bound = lattice_image_bound(
        lattice,
        gaussian_width_bohr,
        real_space_cutoff_factor
    );
    const double width = 2.0 * gaussian_width_bohr;

    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        const Eigen::Vector3d first_position =
            lattice.cart_from_frac(ions[iion].frac_position);

        for (int jion = 0; jion < static_cast<int>(ions.size()); ++jion) {
            const Eigen::Vector3d second_position =
                lattice.cart_from_frac(ions[jion].frac_position);
            for (int n1 = -image_bound; n1 <= image_bound; ++n1) {
                for (int n2 = -image_bound; n2 <= image_bound; ++n2) {
                    for (int n3 = -image_bound; n3 <= image_bound; ++n3) {
                        if (iion == jion && is_zero_image(n1, n2, n3)) {
                            continue;
                        }

                        const Eigen::Vector3d translation =
                            lattice.A * Eigen::Vector3d(n1, n2, n3);
                        const Eigen::Vector3d displacement =
                            first_position - second_position + translation;
                        const double distance = displacement.norm();
                        const double scaled_distance = distance / width;

                        if (distance < 1.0e-14 ||
                            scaled_distance > real_space_cutoff_factor) {
                            continue;
                        }

                        const double distance2 = distance * distance;
                        const double radial_prefactor =
                            std::erfc(scaled_distance)
                                / (distance2 * distance)
                            + 2.0 * std::exp(
                                -scaled_distance * scaled_distance
                              )
                                / (std::sqrt(M_PI) * width * distance2);
                        const Eigen::Vector3d pair_force =
                            ions[iion].Z * ions[jion].Z
                            * radial_prefactor
                            * displacement;

                        forces[iion] += 0.5 * pair_force;
                        forces[jion] -= 0.5 * pair_force;
                    }
                }
            }
        }
    }

    return forces;
}
