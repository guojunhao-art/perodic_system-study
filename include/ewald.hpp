#pragma once

#include "core.hpp"
#include "potentials.hpp"

#include <Eigen/Dense>

#include <vector>

/*
 * Point-ion Ewald energy evaluated with the same Gaussian charge splitting
 * used elsewhere in the code:
 *
 *   E_point = E_gaussian,G + E_erfc,real
 *             - E_gaussian,self + E_background.
 *
 * All ions use one auxiliary Gaussian width. The individual pieces depend on
 * that width, while total is width independent once both sums are converged.
 */
struct EwaldEnergyComponents {
    double reciprocal_gaussian = 0.0;
    double real_space_correction = 0.0;
    double gaussian_self = 0.0;
    double neutralizing_background = 0.0;
    double total = 0.0;
};

EwaldEnergyComponents compute_point_ion_ion_ewald_energy(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    double gaussian_width_bohr,
    double real_space_cutoff_factor = 7.0);

std::vector<Eigen::Vector3d> compute_point_ion_ion_ewald_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    double gaussian_width_bohr,
    double real_space_cutoff_factor = 7.0);
