#pragma once

#include "forces.hpp"
#include "input.hpp"
#include "scf.hpp"

#include <iosfwd>
#include <string>
#include <vector>

struct SetupPerformanceBreakdown {
    double upf_and_ions_seconds = 0.0;
    double basis_and_fft_seconds = 0.0;
    double nonlocal_projector_seconds = 0.0;
    double local_potential_seconds = 0.0;
    double ewald_energy_seconds = 0.0;
    double total_seconds = 0.0;
};

struct ForcePerformanceBreakdown {
    double density_fft_seconds = 0.0;
    double local_seconds = 0.0;
    double ion_ion_seconds = 0.0;
    double nonlocal_seconds = 0.0;
    double mpi_reduction_seconds = 0.0;
    double total_seconds = 0.0;
};

struct SinglePointResult {
    bool converged = false;
    double ecut_hartree = 0.0;
    std::array<int, 3> fft_grid{{0, 0, 0}};
    int plane_wave_count = 0;
    int radial_projector_count = 0;
    int expanded_projector_count = 0;
    int full_kpoint_count = 0;
    int irreducible_kpoint_count = 0;
    int space_group_operation_count = 0;
    int mesh_symmetry_operation_count = 0;
    bool kpoint_time_reversal_used = false;
    double ion_ion_energy = 0.0;
    double total_valence_charge = 0.0;

    SCFOptions options_used;
    KPointSCFResult scf;
    IonicForceComponents forces;
    SetupPerformanceBreakdown setup_performance;
    ForcePerformanceBreakdown force_performance;
};

std::array<int, 3> automatic_fft_grid_dimensions(
    const std::vector<KPointHamiltonian>& kpoint_hamiltonians);

SinglePointResult run_single_point(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    std::ostream* log_stream = nullptr,
    const KPointSCFInitialGuess& initial_guess = {},
    bool print_setup = true);

void print_single_point_result(
    std::ostream& out,
    const AtomicStructure& structure,
    const SinglePointResult& result);
