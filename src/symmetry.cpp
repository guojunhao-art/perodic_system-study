#include "symmetry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

constexpr double two_pi = 2.0 * M_PI;
constexpr long long openmp_minimum_work = 32768;

double wrap_unit(double value) {
    value -= std::floor(value);
    if (value >= 1.0 - 1.0e-12) {
        value = 0.0;
    }
    return value;
}

Eigen::Vector3d wrap_unit(const Eigen::Vector3d& value) {
    return Eigen::Vector3d(
        wrap_unit(value[0]),
        wrap_unit(value[1]),
        wrap_unit(value[2])
    );
}

double minimum_periodic_distance(
    const Eigen::Matrix3d& lattice,
    const Eigen::Vector3d& first,
    const Eigen::Vector3d& second) {

    const Eigen::Vector3d difference = first - second;
    const Eigen::Vector3d nearest =
        difference.array().round().matrix();
    double minimum = std::numeric_limits<double>::infinity();
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                const Eigen::Vector3d shift =
                    nearest + Eigen::Vector3d(i, j, k);
                minimum = std::min(
                    minimum,
                    (lattice * (difference - shift)).norm()
                );
            }
        }
    }
    return minimum;
}

bool same_translation(
    const Eigen::Matrix3d& lattice,
    const Eigen::Vector3d& first,
    const Eigen::Vector3d& second,
    double tolerance_bohr) {

    return minimum_periodic_distance(
        lattice, first, second
    ) <= tolerance_bohr;
}

std::vector<Eigen::Vector3i> lattice_vector_candidates(
    const Eigen::Matrix3d& metric,
    int column,
    double metric_tolerance) {

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(metric);
    if (eigensolver.info() != Eigen::Success ||
        eigensolver.eigenvalues().minCoeff() <= 0.0) {
        throw std::runtime_error(
            "Cannot determine crystal symmetry for an invalid lattice metric."
        );
    }

    const double target = metric(column, column);
    const double minimum_eigenvalue =
        eigensolver.eigenvalues().minCoeff();
    const int bound = static_cast<int>(std::ceil(
        std::sqrt(
            std::max(0.0, target + metric_tolerance)
            / minimum_eigenvalue
        )
    )) + 1;
    if (bound > 32) {
        throw std::runtime_error(
            "The lattice basis is too skewed for the built-in symmetry "
            "search; use a reduced cell or disable kpoint_symmetry."
        );
    }

    std::vector<Eigen::Vector3i> candidates;
    for (int i = -bound; i <= bound; ++i) {
        for (int j = -bound; j <= bound; ++j) {
            for (int k = -bound; k <= bound; ++k) {
                const Eigen::Vector3i integer_vector(i, j, k);
                if (integer_vector.isZero()) {
                    continue;
                }
                const Eigen::Vector3d vector =
                    integer_vector.cast<double>();
                const double length_squared =
                    vector.dot(metric * vector);
                if (std::abs(length_squared - target)
                    <= metric_tolerance) {
                    candidates.push_back(integer_vector);
                }
            }
        }
    }
    return candidates;
}

std::vector<Eigen::Matrix3i> lattice_rotations(
    const AtomicStructure& structure,
    double tolerance_bohr) {

    const Eigen::Matrix3d metric =
        structure.lattice_bohr.transpose() * structure.lattice_bohr;
    const double maximum_length =
        std::sqrt(metric.diagonal().maxCoeff());
    const double metric_scale =
        std::max(1.0, metric.cwiseAbs().maxCoeff());
    const double metric_tolerance = std::max(
        1.0e-11 * metric_scale,
        2.0 * maximum_length * tolerance_bohr
            + tolerance_bohr * tolerance_bohr
    );

    std::array<std::vector<Eigen::Vector3i>, 3> columns;
    for (int column = 0; column < 3; ++column) {
        columns[column] = lattice_vector_candidates(
            metric, column, metric_tolerance
        );
    }

    std::vector<Eigen::Matrix3i> rotations;
    for (const Eigen::Vector3i& first : columns[0]) {
        const Eigen::Vector3d first_d = first.cast<double>();
        for (const Eigen::Vector3i& second : columns[1]) {
            const Eigen::Vector3d second_d = second.cast<double>();
            if (std::abs(
                    first_d.dot(metric * second_d) - metric(0, 1)
                ) > metric_tolerance) {
                continue;
            }
            for (const Eigen::Vector3i& third : columns[2]) {
                const Eigen::Vector3d third_d = third.cast<double>();
                if (std::abs(
                        first_d.dot(metric * third_d) - metric(0, 2)
                    ) > metric_tolerance ||
                    std::abs(
                        second_d.dot(metric * third_d) - metric(1, 2)
                    ) > metric_tolerance) {
                    continue;
                }

                Eigen::Matrix3i rotation;
                rotation.col(0) = first;
                rotation.col(1) = second;
                rotation.col(2) = third;
                if (std::abs(rotation.determinant()) != 1) {
                    continue;
                }
                const Eigen::Matrix3d metric_error =
                    rotation.cast<double>().transpose()
                    * metric
                    * rotation.cast<double>()
                    - metric;
                if (metric_error.cwiseAbs().maxCoeff()
                    <= metric_tolerance) {
                    rotations.push_back(rotation);
                }
            }
        }
    }
    return rotations;
}

bool operation_maps_atoms(
    const AtomicStructure& structure,
    const Eigen::Matrix3i& rotation,
    const Eigen::Vector3d& translation,
    double tolerance_bohr,
    std::vector<int>& mapping) {

    const int atom_count = static_cast<int>(structure.atoms.size());
    mapping.assign(atom_count, -1);
    std::vector<bool> used(atom_count, false);
    for (int source = 0; source < atom_count; ++source) {
        const StructureAtom& atom = structure.atoms[source];
        const Eigen::Vector3d transformed =
            rotation.cast<double>() * atom.frac_position
            + translation;
        int best_target = -1;
        double best_distance = std::numeric_limits<double>::infinity();
        for (int target = 0; target < atom_count; ++target) {
            if (used[target] ||
                structure.atoms[target].element != atom.element) {
                continue;
            }
            const double distance = minimum_periodic_distance(
                structure.lattice_bohr,
                transformed,
                structure.atoms[target].frac_position
            );
            if (distance < best_distance) {
                best_distance = distance;
                best_target = target;
            }
        }
        if (best_target < 0 || best_distance > tolerance_bohr) {
            return false;
        }
        mapping[source] = best_target;
        used[best_target] = true;
    }
    return true;
}

Eigen::Matrix3i inverse_transpose(
    const Eigen::Matrix3i& rotation) {

    const Eigen::Matrix3d inverse_transpose_double =
        rotation.cast<double>().inverse().transpose();
    const Eigen::Matrix3i result =
        inverse_transpose_double.array().round().cast<int>();
    if ((rotation.cast<double>().transpose()
         * result.cast<double>()
         - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff()
        > 1.0e-12) {
        throw std::runtime_error(
            "A crystal rotation does not have an integer reciprocal inverse."
        );
    }
    return result;
}

struct KPointKey {
    std::array<std::int64_t, 3> values{{0, 0, 0}};

    bool operator==(const KPointKey& other) const {
        return values == other.values;
    }
};

struct KPointKeyHash {
    std::size_t operator()(const KPointKey& key) const {
        std::size_t seed = 0;
        for (std::int64_t value : key.values) {
            const std::size_t part = std::hash<std::int64_t>{}(value);
            seed ^= part + 0x9e3779b97f4a7c15ULL
                + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

KPointKey kpoint_key(const Eigen::Vector3d& point) {
    constexpr std::int64_t scale = 1000000000000LL;
    KPointKey key;
    for (int direction = 0; direction < 3; ++direction) {
        const double wrapped = wrap_unit(point[direction]);
        std::int64_t value = static_cast<std::int64_t>(
            std::llround(wrapped * static_cast<double>(scale))
        );
        value %= scale;
        if (value < 0) {
            value += scale;
        }
        key.values[direction] = value;
    }
    return key;
}

using KPointLookup =
    std::unordered_map<KPointKey, int, KPointKeyHash>;

int transformed_kpoint_index(
    const KPointLookup& lookup,
    const Eigen::Matrix3i& reciprocal_rotation,
    int sign,
    const Eigen::Vector3d& point) {

    const Eigen::Vector3d transformed =
        static_cast<double>(sign)
        * reciprocal_rotation.cast<double>() * point;
    const auto found = lookup.find(kpoint_key(transformed));
    return found == lookup.end() ? -1 : found->second;
}

class DisjointSet {
public:
    explicit DisjointSet(int size)
        : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int root(int value) {
        if (parent_[value] != value) {
            parent_[value] = root(parent_[value]);
        }
        return parent_[value];
    }

    void unite(int first, int second) {
        first = root(first);
        second = root(second);
        if (first == second) {
            return;
        }
        if (rank_[first] < rank_[second]) {
            std::swap(first, second);
        }
        parent_[second] = first;
        if (rank_[first] == rank_[second]) {
            ++rank_[first];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

int positive_mod(long long value, int modulus) {
    long long result = value % modulus;
    if (result < 0) {
        result += modulus;
    }
    return static_cast<int>(result);
}

bool grid_rotation_compatible(
    const FFTGrid& grid,
    const Eigen::Matrix3i& rotation) {

    const std::array<int, 3> sizes{{grid.n1, grid.n2, grid.n3}};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (rotation(row, column) == 0) {
                continue;
            }
            const long long numerator =
                static_cast<long long>(sizes[row])
                * rotation(row, column);
            if (numerator % sizes[column] != 0) {
                return false;
            }
        }
    }
    return true;
}

bool operation_maps_real_grid(
    const FFTGrid& grid,
    const SpaceGroupOperation& operation) {

    if (!grid_rotation_compatible(grid, operation.rotation)) {
        return false;
    }
    const Eigen::Matrix3i inverse =
        operation.rotation.cast<double>().inverse()
            .array().round().cast<int>();
    const Eigen::Vector3d inverse_translation =
        -inverse.cast<double>() * operation.translation;
    const std::array<int, 3> sizes{{grid.n1, grid.n2, grid.n3}};
    for (int direction = 0; direction < 3; ++direction) {
        const double grid_shift =
            static_cast<double>(sizes[direction])
            * inverse_translation[direction];
        if (std::abs(grid_shift - std::round(grid_shift)) > 1.0e-8) {
            return false;
        }
    }
    return true;
}

void symmetrize_on_real_grid(
    FFTWorkspace& fft,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<double>& field) {

    const FFTGrid& grid = fft.grid;
    const std::array<int, 3> sizes{{grid.n1, grid.n2, grid.n3}};
    struct InverseGridOperation {
        Eigen::Matrix3i rotation;
        Eigen::Vector3i translation;
    };
    std::vector<InverseGridOperation> inverse_operations;
    inverse_operations.reserve(operations.size());
    for (const SpaceGroupOperation& operation : operations) {
        const Eigen::Matrix3i inverse =
            operation.rotation.cast<double>().inverse()
                .array().round().cast<int>();
        const Eigen::Vector3d inverse_translation =
            -inverse.cast<double>() * operation.translation;
        InverseGridOperation grid_operation;
        for (int row = 0; row < 3; ++row) {
            grid_operation.translation[row] = static_cast<int>(
                std::llround(
                    static_cast<double>(sizes[row])
                    * inverse_translation[row]
                )
            );
            for (int column = 0; column < 3; ++column) {
                grid_operation.rotation(row, column) = static_cast<int>(
                    static_cast<long long>(sizes[row])
                    * inverse(row, column)
                    / sizes[column]
                );
            }
        }
        inverse_operations.push_back(grid_operation);
    }

    std::vector<double> symmetrized(grid.ngrid, 0.0);
    const double inverse_count =
        1.0 / static_cast<double>(operations.size());
#pragma omp parallel for collapse(3) schedule(static) \
    if(fft.thread_count > 1 && grid.ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const Eigen::Vector3i target(i, j, k);
                double sum = 0.0;
                for (const InverseGridOperation& operation :
                     inverse_operations) {
                    const Eigen::Vector3i source =
                        operation.rotation * target
                        + operation.translation;
                    sum += field[grid.index(
                        positive_mod(source[0], grid.n1),
                        positive_mod(source[1], grid.n2),
                        positive_mod(source[2], grid.n3)
                    )];
                }
                symmetrized[grid.index(i, j, k)] =
                    inverse_count * sum;
            }
        }
    }
    field = std::move(symmetrized);
}

void symmetrize_in_reciprocal_space(
    FFTWorkspace& fft,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<double>& field) {

    const FFTGrid& grid = fft.grid;
    for (int point = 0; point < grid.ngrid; ++point) {
        fft.real_grid[point] = {field[point], 0.0};
    }
    fftw_execute(fft.forward_plan);
    const double inverse_grid_size =
        1.0 / static_cast<double>(grid.ngrid);
    std::vector<std::complex<double>> coefficients(grid.ngrid);
    for (int point = 0; point < grid.ngrid; ++point) {
        coefficients[point] =
            inverse_grid_size * fft.forward_raw[point];
    }

    std::vector<std::complex<double>> symmetrized(
        grid.ngrid, {0.0, 0.0}
    );
    const double inverse_count =
        1.0 / static_cast<double>(operations.size());
#pragma omp parallel for collapse(3) schedule(static) \
    if(fft.thread_count > 1 && grid.ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const Eigen::Vector3i frequency =
                    grid.freq_from_indices(i, j, k);
                std::complex<double> sum = 0.0;
                for (const SpaceGroupOperation& operation : operations) {
                    const Eigen::Vector3i source_frequency =
                        operation.rotation.transpose() * frequency;
                    const int source_index = grid.index(
                        positive_mod(source_frequency[0], grid.n1),
                        positive_mod(source_frequency[1], grid.n2),
                        positive_mod(source_frequency[2], grid.n3)
                    );
                    const double phase_angle =
                        -two_pi * frequency.cast<double>().dot(
                            operation.translation
                        );
                    sum += std::polar(1.0, phase_angle)
                        * coefficients[source_index];
                }
                symmetrized[grid.index(i, j, k)] =
                    inverse_count * sum;
            }
        }
    }

    /*
     * FFTW plans retain the data pointers supplied when the workspace is
     * constructed.  Keep reciprocal_grid's allocation in place: move
     * assignment would replace its buffer while backward_plan still points
     * to the old, freed storage.
     */
    std::copy(
        symmetrized.begin(),
        symmetrized.end(),
        fft.reciprocal_grid.begin()
    );
    fftw_execute(fft.backward_plan);
    for (int point = 0; point < grid.ngrid; ++point) {
        field[point] = fft.real_grid[point].real();
    }
}

} // namespace

std::vector<SpaceGroupOperation> find_space_group_operations(
    const AtomicStructure& structure,
    double tolerance_angstrom) {

    if (structure.atoms.empty()) {
        throw std::runtime_error(
            "Cannot determine symmetry for a structure without atoms."
        );
    }
    if (!structure.lattice_bohr.allFinite() ||
        std::abs(structure.lattice_bohr.determinant()) < 1.0e-14) {
        throw std::runtime_error(
            "Cannot determine symmetry for an invalid lattice."
        );
    }
    if (!std::isfinite(tolerance_angstrom) ||
        tolerance_angstrom <= 0.0) {
        throw std::runtime_error(
            "The symmetry tolerance must be positive and finite."
        );
    }
    const double tolerance_bohr =
        tolerance_angstrom * ANGSTROM_TO_BOHR;

    std::map<std::string, int> species_counts;
    for (const StructureAtom& atom : structure.atoms) {
        ++species_counts[atom.element];
    }
    std::string anchor_species = structure.atoms.front().element;
    for (const auto& count : species_counts) {
        if (count.second < species_counts.at(anchor_species)) {
            anchor_species = count.first;
        }
    }
    int anchor = 0;
    while (structure.atoms[anchor].element != anchor_species) {
        ++anchor;
    }

    std::vector<SpaceGroupOperation> operations;
    const std::vector<Eigen::Matrix3i> rotations =
        lattice_rotations(structure, tolerance_bohr);
    for (const Eigen::Matrix3i& rotation : rotations) {
        std::vector<Eigen::Vector3d> accepted_translations;
        for (int target = 0;
             target < static_cast<int>(structure.atoms.size());
             ++target) {
            if (structure.atoms[target].element != anchor_species) {
                continue;
            }
            const Eigen::Vector3d translation = wrap_unit(
                structure.atoms[target].frac_position
                - rotation.cast<double>()
                    * structure.atoms[anchor].frac_position
            );
            bool duplicate = false;
            for (const Eigen::Vector3d& earlier :
                 accepted_translations) {
                if (same_translation(
                        structure.lattice_bohr,
                        translation,
                        earlier,
                        tolerance_bohr
                    )) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            std::vector<int> mapping;
            if (operation_maps_atoms(
                    structure,
                    rotation,
                    translation,
                    tolerance_bohr,
                    mapping
                )) {
                SpaceGroupOperation operation;
                operation.rotation = rotation;
                operation.translation = translation;
                operation.atom_mapping = std::move(mapping);
                operations.push_back(std::move(operation));
                accepted_translations.push_back(translation);
            }
        }
    }
    if (operations.empty()) {
        throw std::runtime_error(
            "Crystal symmetry search did not find the identity operation."
        );
    }
    return operations;
}

KPointReductionResult reduce_kpoints_by_symmetry(
    const AtomicStructure& structure,
    const KPointSet& full_kpoints,
    double tolerance_angstrom,
    bool include_time_reversal) {

    KPointReductionResult result;
    result.irreducible_kpoints = full_kpoints;
    result.full_kpoint_count =
        static_cast<int>(full_kpoints.points.size());
    if (!full_kpoints.uniform_mesh) {
        return result;
    }
    if (full_kpoints.points.empty()) {
        throw std::runtime_error(
            "Cannot reduce an empty uniform k-point mesh."
        );
    }

    result.space_group_operations =
        find_space_group_operations(structure, tolerance_angstrom);
    KPointLookup lookup;
    lookup.reserve(full_kpoints.points.size());
    for (int point = 0;
         point < static_cast<int>(full_kpoints.points.size());
         ++point) {
        if (!lookup.emplace(
                kpoint_key(full_kpoints.points[point].frac_position),
                point
            ).second) {
            throw std::runtime_error(
                "The uniform k-point mesh contains duplicate points."
            );
        }
    }

    std::vector<Eigen::Matrix3i> compatible_reciprocal_rotations;
    for (const SpaceGroupOperation& operation :
         result.space_group_operations) {
        const Eigen::Matrix3i reciprocal_rotation =
            inverse_transpose(operation.rotation);
        bool compatible = true;
        for (const KPoint& point : full_kpoints.points) {
            if (transformed_kpoint_index(
                    lookup,
                    reciprocal_rotation,
                    1,
                    point.frac_position
                ) < 0) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            result.mesh_compatible_operations.push_back(operation);
            bool new_rotation = true;
            for (const Eigen::Matrix3i& earlier :
                 compatible_reciprocal_rotations) {
                if (earlier == reciprocal_rotation) {
                    new_rotation = false;
                    break;
                }
            }
            if (new_rotation) {
                compatible_reciprocal_rotations.push_back(
                    reciprocal_rotation
                );
            }
        }
    }
    if (compatible_reciprocal_rotations.empty()) {
        throw std::runtime_error(
            "No crystal rotation, including identity, preserves the "
            "uniform k-point mesh."
        );
    }

    DisjointSet stars(result.full_kpoint_count);
    for (const Eigen::Matrix3i& reciprocal_rotation :
         compatible_reciprocal_rotations) {
        for (int point = 0; point < result.full_kpoint_count; ++point) {
            const int equivalent = transformed_kpoint_index(
                lookup,
                reciprocal_rotation,
                1,
                full_kpoints.points[point].frac_position
            );
            if (equivalent < 0) {
                throw std::runtime_error(
                    "An allegedly compatible crystal rotation left the "
                    "k-point mesh."
                );
            }
            stars.unite(point, equivalent);
        }
    }
    if (include_time_reversal) {
        const Eigen::Matrix3i identity = Eigen::Matrix3i::Identity();
        for (int point = 0; point < result.full_kpoint_count; ++point) {
            const int reversed = transformed_kpoint_index(
                lookup,
                identity,
                -1,
                full_kpoints.points[point].frac_position
            );
            if (reversed < 0) {
                throw std::runtime_error(
                    "Time reversal does not preserve the uniform k-point mesh."
                );
            }
            stars.unite(point, reversed);
        }
        result.time_reversal_used = true;
    }

    std::map<int, int> root_to_irreducible;
    result.full_to_irreducible.resize(result.full_kpoint_count, -1);
    for (int point = 0; point < result.full_kpoint_count; ++point) {
        const int root = stars.root(point);
        auto found = root_to_irreducible.find(root);
        if (found == root_to_irreducible.end()) {
            const int irreducible =
                static_cast<int>(result.star_members.size());
            found = root_to_irreducible.emplace(
                root, irreducible
            ).first;
            result.star_members.push_back({});
        }
        result.full_to_irreducible[point] = found->second;
        result.star_members[found->second].push_back(point);
    }

    result.irreducible_kpoints.points.clear();
    result.irreducible_kpoints.points.reserve(
        result.star_members.size()
    );
    for (const std::vector<int>& members : result.star_members) {
        KPoint representative =
            full_kpoints.points[members.front()];
        representative.weight = 0.0;
        for (int point : members) {
            representative.weight +=
                full_kpoints.points[point].weight;
        }
        result.irreducible_kpoints.points.push_back(
            representative
        );
    }
    normalize_kpoint_weights(result.irreducible_kpoints);
    result.reduction_applied =
        result.irreducible_kpoints.points.size()
        < full_kpoints.points.size();
    if (result.reduction_applied) {
        result.irreducible_kpoints.description +=
            " (symmetry-reduced)";
    }
    return result;
}

std::array<int, 3> symmetry_compatible_fft_dimensions(
    const std::array<int, 3>& minimum_dimensions,
    const std::vector<SpaceGroupOperation>& operations) {

    std::array<int, 3> parent{{0, 1, 2}};
    auto root = [&parent](int value) {
        while (parent[value] != value) {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        return value;
    };
    auto unite = [&parent, &root](int first, int second) {
        first = root(first);
        second = root(second);
        if (first != second) {
            parent[second] = first;
        }
    };
    for (const SpaceGroupOperation& operation : operations) {
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                if (row != column &&
                    operation.rotation(row, column) != 0) {
                    unite(row, column);
                }
            }
        }
    }

    std::array<int, 3> result = minimum_dimensions;
    for (int direction = 0; direction < 3; ++direction) {
        const int component = root(direction);
        int maximum = result[direction];
        for (int other = 0; other < 3; ++other) {
            if (root(other) == component) {
                maximum = std::max(maximum, result[other]);
            }
        }
        for (int other = 0; other < 3; ++other) {
            if (root(other) == component) {
                result[other] = maximum;
            }
        }
    }
    return result;
}

void symmetrize_scalar_field(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<double>& field) {

    (void)lattice;
    if (operations.empty()) {
        return;
    }
    if (static_cast<int>(field.size()) != fft.grid.ngrid) {
        throw std::runtime_error(
            "A scalar field to symmetrize has the wrong FFT-grid size."
        );
    }
    bool real_grid_compatible = true;
    for (const SpaceGroupOperation& operation : operations) {
        if (!grid_rotation_compatible(
                fft.grid, operation.rotation
            )) {
            throw std::runtime_error(
                "The FFT grid is incompatible with a crystal rotation "
                "used for k-point reduction. Use fft_grid = auto, choose "
                "symmetry-compatible dimensions, or disable "
                "kpoint_symmetry."
            );
        }
        if (!operation_maps_real_grid(fft.grid, operation)) {
            real_grid_compatible = false;
        }
    }
    if (real_grid_compatible) {
        symmetrize_on_real_grid(fft, operations, field);
    } else {
        symmetrize_in_reciprocal_space(fft, operations, field);
    }
}

void symmetrize_atomic_vectors(
    const AtomicStructure& structure,
    const std::vector<SpaceGroupOperation>& operations,
    std::vector<Eigen::Vector3d>& vectors) {

    if (operations.empty()) {
        return;
    }
    if (vectors.size() != structure.atoms.size()) {
        throw std::runtime_error(
            "Atomic vectors and structure have different atom counts."
        );
    }
    std::vector<Eigen::Vector3d> symmetrized(
        vectors.size(), Eigen::Vector3d::Zero()
    );
    const Eigen::Matrix3d inverse_lattice =
        structure.lattice_bohr.inverse();
    for (const SpaceGroupOperation& operation : operations) {
        if (operation.atom_mapping.size() != vectors.size()) {
            throw std::runtime_error(
                "A space-group operation has an invalid atom mapping."
            );
        }
        const Eigen::Matrix3d cartesian_rotation =
            structure.lattice_bohr
            * operation.rotation.cast<double>()
            * inverse_lattice;
        for (int atom = 0;
             atom < static_cast<int>(vectors.size());
             ++atom) {
            const int target = operation.atom_mapping[atom];
            if (target < 0 ||
                target >= static_cast<int>(vectors.size())) {
                throw std::runtime_error(
                    "A space-group atom mapping is out of range."
                );
            }
            symmetrized[target] +=
                cartesian_rotation * vectors[atom];
        }
    }
    const double inverse_count =
        1.0 / static_cast<double>(operations.size());
    for (Eigen::Vector3d& vector : symmetrized) {
        vector *= inverse_count;
    }
    vectors = std::move(symmetrized);
}
