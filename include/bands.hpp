#pragma once

#include "input.hpp"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct BandPathSample {
    Eigen::Vector3d frac_position = Eigen::Vector3d::Zero();
    double distance_bohr_inverse = 0.0;
    std::string label;
};

/*
 * Interpolate the labeled high-symmetry vertices using reciprocal fractional
 * coordinates. Distances are accumulated in Cartesian reciprocal space.
 */
std::vector<BandPathSample> interpolate_band_path(
    const Lattice& lattice,
    const BandStructureOptions& options);

struct NSCFResult;

void write_band_structure(
    const std::string& path,
    const NSCFResult& result);

void write_fat_band_structure(
    const std::string& path,
    const NSCFResult& result);
