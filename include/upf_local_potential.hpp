#pragma once

#include "core.hpp"
#include "upf_reader.hpp"

#include <Eigen/Dense>

#include <complex>
#include <string>
#include <vector>

/*
 * One NC-UPF species after splitting its local potential into
 *
 *   V_loc(r) = -Z erf[r/(sqrt(2) sigma)] / r + Delta V(r).
 *
 * The Gaussian Coulomb part uses the same width as the smooth ionic charge
 * employed by the periodic electrostatics. Delta V is finite at the origin
 * and short ranged.
 */
struct UPFLocalSpecies {
    std::string element;
    double valence_charge = 0.0;
    double gaussian_width_bohr = 0.0;

    std::vector<double> radial_grid_bohr;
    std::vector<double> quadrature_weights;
    std::vector<double> correction_hartree;

    /*
     * Standard finite G=0 ("alpha Z") term:
     *
     *   4 pi integral r^2 [V_loc(r) + Z/r] dr.
     *
     * It is independent of the auxiliary Gaussian screening width.
     */
    double g0_correction_hartree_bohr3 = 0.0;
};

struct UPFLocalIon {
    int species_index = 0;
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
};

/*
 * Position-independent reciprocal-space data shared by local-potential and
 * local-force assembly.  The expensive radial Fourier--Bessel transform is
 * evaluated once per species and FFT-grid vector; moving ions changes only
 * the translation phases.
 */
struct UPFLocalReciprocalCache {
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int species_count = 0;
    std::vector<Eigen::Vector3d> g_cart;
    std::vector<double> species_kernels;

    int grid_size() const;
    double kernel(int species_index, int grid_index) const;
};

UPFLocalSpecies prepare_upf_local_species(
    const UPFData& upf,
    double gaussian_width_bohr);

/*
 * erf[r/(sqrt(2) sigma)] / r, with its finite r -> 0 limit.
 * Units are inverse Bohr.
 */
double gaussian_coulomb_shape(
    double radius_bohr,
    double gaussian_width_bohr);

/*
 * Isolated radial Fourier transform of Delta V:
 *
 *   4 pi integral r^2 j_0(G r) Delta V(r) dr.
 *
 * Units are Ha * Bohr^3.
 */
double upf_local_correction_transform(
    const UPFLocalSpecies& species,
    double g_bohr_inverse);

/*
 * Full isolated local-potential kernel for G > 0:
 *
 *   -4 pi Z exp(-sigma^2 G^2 / 2) / G^2 + Delta V(G).
 *
 * At G = 0 the divergent Coulomb term is omitted and the standard finite
 * integral 4 pi int r^2 [V_loc(r) + Z/r] dr is returned.
 */
double upf_local_kernel_G(
    const UPFLocalSpecies& species,
    double g_bohr_inverse);

UPFLocalReciprocalCache build_upf_local_reciprocal_cache(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    int thread_count = 1);

/*
 * Periodic Fourier-series coefficients, including ionic translation phases.
 * Coefficients have units of Hartree and include the 1/Omega factor.
 */
std::vector<std::complex<double>> build_upf_local_potential_G(
    const Lattice& lattice,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions,
    int thread_count = 1);

std::vector<std::complex<double>> build_upf_local_potential_G(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions);

std::vector<double> build_upf_local_potential_real(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions);

std::vector<double> build_upf_local_potential_real(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions);
