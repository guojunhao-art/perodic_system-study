#pragma once

#include "bands.hpp"
#include "checkpoint.hpp"
#include "input.hpp"
#include "pdos.hpp"
#include "scf.hpp"

#include <Eigen/Dense>

#include <iosfwd>
#include <vector>

struct NSCFElectronicState {
    int spin_channel = 0;
    int kpoint_index = 0;
    int owner_rank = 0;
    Eigen::VectorXd eigenvalues;
    std::vector<double> residual_norms;
    int iterations = 0;

    /* Stored only on the rank that owns this k point. */
    Eigen::MatrixXcd orbitals;
};

struct NSCFDiagonalizationResult {
    bool converged = false;
    int nspin = 1;
    int nbands = 0;
    double fermi_energy_ha = 0.0;
    double wall_time_seconds = 0.0;
    std::vector<NSCFElectronicState> states;
};

struct NSCFResult {
    bool converged = false;
    int full_kpoint_count = 0;
    int irreducible_kpoint_count = 0;
    int space_group_operation_count = 0;
    int mesh_symmetry_operation_count = 0;
    bool kpoint_time_reversal_used = false;
    double checkpoint_fermi_energy_ha = 0.0;
    double wall_time_seconds = 0.0;
    bool band_path = false;
    std::vector<BandPathSample> path;
    SCFOptions options_used;
    KPointSCFResult electronic;
    NSCFDiagonalizationResult diagonalization;
    AtomicProjectionResult projection;
};

/*
 * Rebuild the Kohn--Sham potential from a persisted self-consistent density
 * and diagonalize it on the k-point mesh in config.  The density is never
 * updated or mixed.
 */
NSCFResult run_fixed_density_nscf(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SCFCheckpoint& checkpoint,
    std::ostream* log_stream = nullptr);

void print_nscf_result(
    std::ostream& out,
    const NSCFResult& result);
