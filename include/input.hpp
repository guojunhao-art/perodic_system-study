#pragma once

#include "scf.hpp"

#include <Eigen/Dense>

#include <array>
#include <map>
#include <string>
#include <vector>

constexpr double ANGSTROM_TO_BOHR = 1.8897261254578281;
constexpr double HARTREE_TO_EV = 27.211386245988;

struct StructureAtom {
    std::string element;
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
    std::array<bool, 3> movable{{true, true, true}};
};

/*
 * POSCAR lattice vectors are stored as columns after conversion from
 * Angstrom to Bohr. Atomic positions are always stored fractionally.
 */
struct AtomicStructure {
    std::string comment;
    Eigen::Matrix3d lattice_bohr = Eigen::Matrix3d::Zero();
    std::vector<std::string> species_order;
    std::vector<StructureAtom> atoms;
    bool selective_dynamics = false;
    bool selective_dynamics_cartesian = false;
};

struct KPoint {
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
    double weight = 1.0;
};

struct KPointSet {
    std::vector<KPoint> points{{}};
    std::string description = "Gamma";
    bool uniform_mesh = true;
    std::array<int, 3> mesh{{1, 1, 1}};
    bool gamma_centered = true;
};

KPointSet make_uniform_kpoint_mesh(
    const std::array<int, 3>& mesh,
    bool gamma_centered);

void normalize_kpoint_weights(KPointSet& kpoints);

enum class CalculationType {
    SCF,
    Relax,
    NSCF
};

enum class IonAlgorithm {
    BFGS
};

struct RelaxationOptions {
    IonAlgorithm algorithm = IonAlgorithm::BFGS;
    int max_ionic_steps = 50;
    int max_backtracks = 5;

    double force_tolerance_ha_bohr = 2.0e-4;
    double max_step_angstrom = 0.10;
    double initial_curvature_ha_bohr2 = 0.10;

    std::string contcar_path = "CONTCAR";
    std::string trajectory_path = "relaxation.xyz";
};

struct BandPathNode {
    std::string label;
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
};

struct BandStructureOptions {
    std::vector<BandPathNode> path;

    /*
     * Number of samples on each segment including both end points. Shared
     * vertices are emitted once, so S segments produce
     * S * (points_per_segment - 1) + 1 samples.
     */
    int points_per_segment = 20;
    std::string output_path;
    std::string projection_output_path;
};

struct DensityOfStatesOptions {
    double smearing_ev = 0.10;
    int points = 2001;

    /*
     * Explicit limits are relative to the converged Fermi energy.  The auto
     * limits span all computed bands plus five Gaussian widths.
     */
    bool energy_min_auto = true;
    bool energy_max_auto = true;
    double energy_min_ev = 0.0;
    double energy_max_ev = 0.0;

    std::string output_path;
};

struct ProjectedDensityOfStatesOptions {
    std::string output_path;
    double lowdin_relative_cutoff = 1.0e-10;
};

struct KPointSymmetryOptions {
    bool enabled = true;
    bool include_time_reversal = true;
    double tolerance_angstrom = 1.0e-5;
};

struct CalculationConfig {
    std::string source_path;
    std::string structure_path;
    std::map<std::string, std::string> pseudopotential_paths;

    CalculationType calculation = CalculationType::SCF;
    RelaxationOptions relaxation;
    BandStructureOptions bands;
    DensityOfStatesOptions dos;
    ProjectedDensityOfStatesOptions pdos;
    std::string checkpoint_input_path;
    std::string checkpoint_output_path;

    double ecut_hartree = 0.0;
    std::array<int, 3> fft_grid{{0, 0, 0}};
    /* 0 means auto; the omitted-key default remains one thread. */
    int fft_threads = 1;
    double ewald_width_bohr = 0.70;

    bool nelect_auto = true;
    bool nbands_auto = true;
    KPointSet kpoints;
    KPointSymmetryOptions kpoint_symmetry;
    SCFOptions scf;
};

AtomicStructure read_poscar(const std::string& path);

CalculationConfig read_calculation_config(const std::string& path);
