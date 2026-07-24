#pragma once

#include "core.hpp"
#include "potentials.hpp"
#include "upf_local_potential.hpp"

#include <Eigen/Dense>
#include <complex>
#include <vector>

struct IonicForceComponents {
    std::vector<Eigen::Vector3d> local;
    std::vector<Eigen::Vector3d> ion_ion;
    std::vector<Eigen::Vector3d> nonlocal;
    std::vector<Eigen::Vector3d> total;
};

/*
 * Hellmann--Feynman force from the local electron--ion potential,
 * including both the smooth Coulomb and short-range Gaussian parts.
 * n_G uses the convention n_G = Omega^{-1} int n(r) exp(-i G.r) dr.
 */
std::vector<Eigen::Vector3d> compute_local_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    const std::vector<std::complex<double>>& n_G);

/*
 * The same local Hellmann--Feynman force for screened NC-UPF local
 * potentials. The species/ion arrays use the same convention as
 * build_upf_local_potential_G().
 */
std::vector<Eigen::Vector3d> compute_upf_local_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    const std::vector<std::complex<double>>& n_G);

std::vector<Eigen::Vector3d> compute_upf_local_ionic_forces(
    const Lattice& lattice,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions,
    const std::vector<std::complex<double>>& n_G,
    int thread_count = 1);

/* Force obtained by differentiating E_II^smooth. */
std::vector<Eigen::Vector3d> compute_smooth_ion_ion_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions);

/* Force obtained by differentiating sum_n f_n <psi_n|V_NL|psi_n>. */
std::vector<Eigen::Vector3d> compute_nonlocal_ionic_forces(
    const PlaneWaveBasis3D& basis,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    int nions,
    int thread_count = 1);

IonicForceComponents compute_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<Ion>& ions,
    const std::vector<std::complex<double>>& n_G,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations);
