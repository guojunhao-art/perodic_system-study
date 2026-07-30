#pragma once

#include <Eigen/Dense>

#include <vector>

/*
 * A crystal-space-group operation in fractional direct coordinates:
 *
 *     s' = rotation * s + translation  (modulo lattice vectors).
 *
 * atom_mapping[i] is the atom index reached from atom i.  The mapping is
 * retained because vector quantities such as ionic forces must be rotated
 * and permuted when an irreducible k-point star is reconstructed.
 */
struct SpaceGroupOperation {
    Eigen::Matrix3i rotation = Eigen::Matrix3i::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::vector<int> atom_mapping;
};
