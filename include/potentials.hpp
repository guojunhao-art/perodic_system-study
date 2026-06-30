#pragma once

#include "core.hpp"

#include <Eigen/Dense>
#include <vector>

struct Ion {
    /*
     * Fractional coordinate in [0, 1)^3.
     */
    Eigen::Vector3d frac_position;

    /*
     * Valence charge.
     * For a one-pseudo-atom toy system, choose Z = nelec.
     */
    double Z = 1.0;

    /*
     * Gaussian smoothing length, in the same length unit as lattice vectors.
     */
    double sigma = 0.5;

    double short_amp = 0.0;
    double short_rc = 0.5;
    
    // toy nonlocal projector
    double beta_rc = 0.5;
    double beta_D = 0.0;
};

struct LocalPotentialComponents {
    std::vector<double> Vcoul;
    std::vector<double> Vshort;
    std::vector<double> Vtotal;

    double Eion_smooth = 0.0;
};

std::vector<double> build_gaussian_local_pseudopotential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions);

std::vector<std::complex<double>> build_gaussian_ion_density_G(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions);

double compute_smooth_ion_ion_energy(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions);

double compute_total_electrostatic_energy_direct(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& rhoI_G,
    const std::vector<std::complex<double>>& n_G);

std::vector<double> build_short_range_local_potential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions);

LocalPotentialComponents build_local_pseudopotential_components(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions);

std::vector<NonlocalProjector> build_s_gaussian_projectors(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<Ion>& ions);

Eigen::VectorXcd apply_nonlocal_projectors(
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::VectorXcd& c);