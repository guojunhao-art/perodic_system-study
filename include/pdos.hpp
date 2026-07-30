#pragma once

#include "core.hpp"
#include "dos.hpp"
#include "input.hpp"
#include "scf.hpp"
#include "upf_local_potential.hpp"
#include "upf_reader.hpp"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct AtomicProjectionOrbital {
    int atom_index = -1;
    std::string element;
    int wavefunction_index = -1;
    std::string label;
    int angular_momentum = -1;

    /*
     * Quantum ESPRESSO real-harmonic ordering within one l shell:
     * 0, cos(phi), sin(phi), cos(2 phi), sin(2 phi), ...
     */
    int real_harmonic_index = -1;
};

struct RealHarmonicDescriptor {
    int absolute_m = -1;
    std::string branch;
    std::string orbital_name;
};

/*
 * Describe the QE ylmr2 real-harmonic ordering.  For |m| > 0 the real
 * cos/sin functions are linear combinations of complex +/-m eigenfunctions,
 * so absolute_m is reported rather than a signed magnetic quantum number.
 */
RealHarmonicDescriptor describe_real_harmonic(
    int angular_momentum,
    int real_harmonic_index);

struct AtomicProjectionSpecies {
    std::string element;
    std::vector<double> radial_grid_bohr;
    std::vector<double> quadrature_weights;
    std::vector<UPFAtomicWavefunction> wavefunctions;
};

AtomicProjectionSpecies prepare_atomic_projection_species(
    const UPFData& upf);

std::vector<AtomicProjectionOrbital> make_atomic_projection_orbitals(
    const std::vector<AtomicProjectionSpecies>& species,
    const std::vector<UPFLocalIon>& ions);

struct AtomicProjectionBasis {
    std::vector<AtomicProjectionOrbital> orbitals;

    /*
     * Columns are the Löwdin-orthonormalized pseudo-atomic Bloch orbitals in
     * the plane-wave basis of one k point.
     */
    Eigen::MatrixXcd coefficients;
    double minimum_overlap_eigenvalue = 0.0;
    double maximum_overlap_eigenvalue = 0.0;
    double orthonormality_error = 0.0;
};

AtomicProjectionBasis build_lowdin_atomic_projection_basis(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<AtomicProjectionSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    double relative_eigenvalue_cutoff,
    int thread_count = 1);

Eigen::MatrixXd compute_lowdin_projection_weights(
    const AtomicProjectionBasis& projection_basis,
    const Eigen::MatrixXcd& eigenvectors);

struct AtomicProjectionState {
    int spin_channel = 0;
    int kpoint_index = 0;
    Eigen::MatrixXd weights;
};

struct AtomicProjectionResult {
    std::vector<AtomicProjectionOrbital> orbitals;
    std::vector<AtomicProjectionState> states;
    double minimum_overlap_eigenvalue = 0.0;
    double maximum_overlap_eigenvalue = 0.0;
    double maximum_orthonormality_error = 0.0;
    double maximum_projection_weight_excess = 0.0;
};

struct ProjectedDensityOfStatesChannel {
    AtomicProjectionOrbital orbital;
    std::vector<std::vector<double>> spin_per_ha;
    std::vector<std::vector<double>> integrated_spin;
};

struct ProjectedDensityOfStatesResult {
    int nspin = 1;
    int nbands = 0;
    int kpoint_count = 0;
    double fermi_energy_ha = 0.0;
    double gaussian_sigma_ha = 0.0;
    double occupied_spilling = 0.0;
    double full_projected_state_weight = 0.0;
    double analytic_projected_states_in_window = 0.0;
    double numerical_projected_states_in_window = 0.0;
    double numerical_minus_analytic_projected_states = 0.0;
    AtomicProjectionResult projection;
    std::vector<double> energies_ha;
    std::vector<ProjectedDensityOfStatesChannel> channels;
};

ProjectedDensityOfStatesResult compute_projected_density_of_states(
    const KPointSCFResult& electronic,
    const SCFOptions& scf_options,
    const DensityOfStatesResult& total_dos,
    const AtomicProjectionResult& projection);

void write_projected_density_of_states(
    const std::string& path,
    const ProjectedDensityOfStatesResult& result);
