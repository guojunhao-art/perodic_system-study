#pragma once

#include "input.hpp"
#include "symmetry_types.hpp"

#include <array>
#include <vector>

struct KPointReductionResult {
    KPointSet irreducible_kpoints;
    std::vector<SpaceGroupOperation> space_group_operations;
    std::vector<SpaceGroupOperation> mesh_compatible_operations;

    /*
     * Maps each point in the original full mesh to an irreducible-point
     * index.  star_members[ikir] stores the corresponding full-mesh indices.
     */
    std::vector<int> full_to_irreducible;
    std::vector<std::vector<int>> star_members;

    int full_kpoint_count = 0;
    bool time_reversal_used = false;
    bool reduction_applied = false;
};

/*
 * Discover all space-group operations that leave the lattice and the
 * species-labelled atomic motif invariant within a Cartesian tolerance.
 */
std::vector<SpaceGroupOperation> find_space_group_operations(
    const AtomicStructure& structure,
    double tolerance_angstrom);

/*
 * Reduce a parser-generated uniform k mesh to irreducible representatives.
 * Explicit user-weighted point sets are returned unchanged.
 */
KPointReductionResult reduce_kpoints_by_symmetry(
    const AtomicStructure& structure,
    const KPointSet& full_kpoints,
    double tolerance_angstrom,
    bool include_time_reversal);

/*
 * Make an automatically selected FFT grid compatible with the fractional
 * rotations used for density symmetrization.  Directions mixed by any
 * rotation receive the same FFT dimension.
 */
std::array<int, 3> symmetry_compatible_fft_dimensions(
    const std::array<int, 3>& minimum_dimensions,
    const std::vector<SpaceGroupOperation>& operations);

/*
 * Apply the space-group average to a scalar periodic real-space field using
 * reciprocal-space phases.  This supports nonsymmorphic translations
 * without requiring them to fall exactly on the FFT grid.
 */
void symmetrize_scalar_field(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<double>& field);

/*
 * Rotate and permute one Cartesian vector per atom by a space-group average.
 */
void symmetrize_atomic_vectors(
    const AtomicStructure& structure,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<Eigen::Vector3d>& vectors);
