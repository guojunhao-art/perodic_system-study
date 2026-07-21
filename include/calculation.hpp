#pragma once

#include "forces.hpp"
#include "input.hpp"
#include "scf.hpp"

#include <iosfwd>
#include <string>
#include <vector>

struct SinglePointResult {
    bool converged = false;
    double ecut_hartree = 0.0;
    int plane_wave_count = 0;
    int radial_projector_count = 0;
    int expanded_projector_count = 0;
    double ion_ion_energy = 0.0;
    double total_valence_charge = 0.0;

    SCFOptions options_used;
    SCFResult scf;
    IonicForceComponents forces;
};

SinglePointResult run_single_point(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    std::ostream* log_stream = nullptr);

void print_single_point_result(
    std::ostream& out,
    const AtomicStructure& structure,
    const SinglePointResult& result);
