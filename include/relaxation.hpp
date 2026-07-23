#pragma once

#include "calculation.hpp"
#include "input.hpp"

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

enum class IonicSearchDirection {
    None,
    InitialHessian,
    BFGS,
    SteepestDescentReset
};

struct IonicStepSummary {
    int step = 0;
    double free_energy_ha = 0.0;
    double energy_change_ha = 0.0;
    double maximum_force_ha_bohr = 0.0;
    double maximum_displacement_angstrom = 0.0;
    double linear_energy_change_ha = 0.0;
    int scf_iterations = 0;
    int backtracks = 0;
    IonicSearchDirection search_direction = IonicSearchDirection::None;
};

struct RelaxationResult {
    bool converged = false;
    int ionic_steps = 0;
    std::string stop_reason;
    AtomicStructure structure;
    SinglePointResult electronic;
    std::vector<IonicStepSummary> history;
};

using FixedCellEvaluator = std::function<SinglePointResult(
    const AtomicStructure&,
    const KPointSCFInitialGuess&)>;

/*
 * Minimize the finite-temperature variational energy at fixed lattice
 * vectors. Cartesian ionic coordinates are the optimization variables and
 * StructureAtom::movable selects their active components.
 */
RelaxationResult run_fixed_cell_relaxation(
    const AtomicStructure& initial_structure,
    const CalculationConfig& config,
    const FixedCellEvaluator& evaluator,
    std::ostream* log_stream = nullptr);

RelaxationResult run_fixed_cell_relaxation(
    const AtomicStructure& initial_structure,
    const CalculationConfig& config,
    std::ostream* log_stream = nullptr);

void write_poscar(
    const std::string& path,
    const AtomicStructure& structure,
    const std::string& comment);
