#pragma once

#include "calculation.hpp"
#include "input.hpp"
#include "scf.hpp"

#include <Eigen/Dense>

#include <iosfwd>
#include <string>
#include <vector>

struct BandPathSample {
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
    double distance_bohr_inverse = 0.0;
    std::string label;
};

struct BandElectronicState {
    int spin_channel = 0;
    int kpoint_index = 0;
    int owner_rank = 0;
    Eigen::VectorXd eigenvalues;
    std::vector<double> residual_norms;
    int iterations = 0;
};

struct BandStructureResult {
    bool converged = false;
    int nspin = 1;
    int nbands = 0;
    double fermi_energy_ha = 0.0;
    double wall_time_seconds = 0.0;
    std::vector<BandPathSample> path;
    std::vector<BandElectronicState> states;
};

/*
 * Interpolate the labeled high-symmetry vertices using reciprocal fractional
 * coordinates. Distances are accumulated in Cartesian reciprocal space.
 */
std::vector<BandPathSample> interpolate_band_path(
    const Lattice& lattice,
    const BandStructureOptions& options);

/*
 * Diagonalize a fixed spin-dependent effective potential independently at
 * each path point. No occupations, density update, or mixing are performed.
 * In MPI mode each k point is solved by exactly one rank and the eigenvalues
 * and residual diagnostics are replicated on all ranks.
 */
BandStructureResult solve_fixed_potential_bands(
    const std::vector<BandPathSample>& path,
    const std::vector<KPointHamiltonian>& hamiltonians,
    FFTWorkspace& fft,
    const std::vector<std::vector<double>>& effective_potentials,
    const SCFOptions& options,
    double fermi_energy_ha,
    std::ostream* log_stream = nullptr);

/*
 * Rebuild the final ionic pseudopotential for the requested path, construct
 * V_eff from the converged SCF density, and run the fixed-potential NSCF
 * diagonalizations.
 */
BandStructureResult run_band_structure(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SinglePointResult& scf_result,
    std::ostream* log_stream = nullptr);

void write_band_structure(
    const std::string& path,
    const BandStructureResult& result);
