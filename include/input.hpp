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
};

struct KPoint {
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
    double weight = 1.0;
};

struct KPointSet {
    std::vector<KPoint> points{{}};
    std::string description = "Gamma";
};

KPointSet make_uniform_kpoint_mesh(
    const std::array<int, 3>& mesh,
    bool gamma_centered);

void normalize_kpoint_weights(KPointSet& kpoints);

struct CalculationConfig {
    std::string source_path;
    std::string structure_path;
    std::map<std::string, std::string> pseudopotential_paths;

    double ecut_hartree = 0.0;
    std::array<int, 3> fft_grid{{0, 0, 0}};
    double ewald_width_bohr = 0.70;

    bool nelect_auto = true;
    bool nbands_auto = true;
    KPointSet kpoints;
    SCFOptions scf;
};

AtomicStructure read_poscar(const std::string& path);

CalculationConfig read_calculation_config(const std::string& path);
