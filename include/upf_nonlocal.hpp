#pragma once

#include "core.hpp"
#include "upf_local_potential.hpp"
#include "upf_reader.hpp"

#include <Eigen/Dense>

#include <string>
#include <vector>

/*
 * One NC-UPF species prepared for reciprocal-space nonlocal projectors.
 * PP_BETA stores r beta_l(r); PP_DIJ is converted from Ry to Hartree but
 * otherwise keeps the file's joint beta/D normalization convention.
 */
struct UPFNonlocalSpecies {
    std::string element;
    std::vector<double> radial_grid_bohr;
    std::vector<double> quadrature_weights;
    std::vector<UPFProjector> radial_projectors;
    Eigen::MatrixXd dij_hartree;
};

UPFNonlocalSpecies prepare_upf_nonlocal_species(const UPFData& upf);

/*
 * Quantum ESPRESSO ordering for the 2l+1 real spherical harmonics:
 *
 *   m = 0, cos(phi), sin(phi), cos(2 phi), sin(2 phi), ...
 *
 * The Condon--Shortley phase is included. At G=0, l>0 returns zeros because
 * the accompanying radial transform vanishes as G^l.
 */
std::vector<double> qe_real_spherical_harmonics(
    int l,
    const Eigen::Vector3d& G);

/*
 * Build plane-wave matrix elements
 *
 *   <G|beta_Iilm> = (-i)^l / sqrt(Omega)
 *                    Y_lm(Ghat) beta_il(G) exp(-i G.R_I),
 *
 * where beta_il(G) = 4 pi int r j_l(Gr) [r beta_il(r)] dr.
 *
 * Each same-l PP_DIJ block is diagonalized exactly. The resulting eigen-
 * projectors have scalar D values and can therefore use the existing Hpsi,
 * E_NL, and analytic-force implementations without discarding off-diagonal
 * D_ij elements. Real UPF projectors are never normalized independently.
 */
std::vector<NonlocalProjector> build_upf_nonlocal_projectors(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFNonlocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions);
