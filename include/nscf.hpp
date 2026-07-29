#pragma once

#include "bands.hpp"
#include "checkpoint.hpp"
#include "input.hpp"
#include "scf.hpp"

#include <iosfwd>

struct NSCFResult {
    bool converged = false;
    int full_kpoint_count = 0;
    int irreducible_kpoint_count = 0;
    int space_group_operation_count = 0;
    int mesh_symmetry_operation_count = 0;
    bool kpoint_time_reversal_used = false;
    double checkpoint_fermi_energy_ha = 0.0;
    double wall_time_seconds = 0.0;
    SCFOptions options_used;
    KPointSCFResult electronic;
    BandStructureResult diagonalization;
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
