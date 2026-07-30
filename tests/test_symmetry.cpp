#include "calculation.hpp"
#include "symmetry.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double value,
    double reference,
    double tolerance,
    const std::string& message) {

    if (std::abs(value - reference) > tolerance) {
        throw std::runtime_error(
            message + ": value=" + std::to_string(value)
            + ", reference=" + std::to_string(reference)
        );
    }
}

AtomicStructure simple_cubic_structure() {
    AtomicStructure structure;
    structure.comment = "simple cubic";
    structure.lattice_bohr = 8.0 * Eigen::Matrix3d::Identity();
    structure.species_order = {"X"};
    StructureAtom atom;
    atom.element = "X";
    atom.frac_position = Eigen::Vector3d(0.17, 0.23, 0.31);
    structure.atoms.push_back(atom);
    return structure;
}

AtomicStructure generic_triclinic_structure() {
    AtomicStructure structure;
    structure.comment = "generic triclinic binary";
    structure.lattice_bohr <<
        7.1, 1.2, 0.4,
        0.0, 8.3, 1.1,
        0.0, 0.0, 9.7;
    structure.species_order = {"X", "Y"};
    StructureAtom first;
    first.element = "X";
    first.frac_position = Eigen::Vector3d(0.13, 0.21, 0.34);
    StructureAtom second;
    second.element = "Y";
    second.frac_position = Eigen::Vector3d(0.46, 0.57, 0.72);
    structure.atoms = {first, second};
    return structure;
}

AtomicStructure diamond_primitive_structure() {
    AtomicStructure structure;
    structure.comment = "diamond primitive";
    const double half_lattice = 5.43 * ANGSTROM_TO_BOHR / 2.0;
    structure.lattice_bohr <<
        0.0,          half_lattice, half_lattice,
        half_lattice, 0.0,          half_lattice,
        half_lattice, half_lattice, 0.0;
    structure.species_order = {"Si"};
    StructureAtom first;
    first.element = "Si";
    first.frac_position = Eigen::Vector3d::Zero();
    StructureAtom second = first;
    second.frac_position = Eigen::Vector3d(0.25, 0.25, 0.25);
    structure.atoms = {first, second};
    return structure;
}

void test_cubic_space_group_and_mesh_reduction() {
    const AtomicStructure structure = simple_cubic_structure();
    const auto operations =
        find_space_group_operations(structure, 1.0e-6);
    require(
        operations.size() == 48,
        "A primitive simple-cubic lattice should have 48 point operations"
    );
    for (const SpaceGroupOperation& operation : operations) {
        require(
            operation.atom_mapping.size() == 1 &&
            operation.atom_mapping[0] == 0,
            "Single-atom cubic symmetry produced a bad atom mapping"
        );
    }

    const KPointSet gamma =
        make_uniform_kpoint_mesh({{4, 4, 4}}, true);
    const KPointReductionResult gamma_reduced =
        reduce_kpoints_by_symmetry(
            structure, gamma, 1.0e-6, true
        );
    require(
        gamma_reduced.full_kpoint_count == 64 &&
        gamma_reduced.irreducible_kpoints.points.size() == 10,
        "Cubic Gamma 4x4x4 mesh should reduce from 64 to 10 points"
    );
    require(
        gamma_reduced.mesh_compatible_operations.size() == 48,
        "All cubic operations should preserve an isotropic Gamma mesh"
    );

    double weight_sum = 0.0;
    int member_count = 0;
    std::vector<int> multiplicities;
    for (int point = 0;
         point < static_cast<int>(
             gamma_reduced.irreducible_kpoints.points.size()
         );
         ++point) {
        weight_sum +=
            gamma_reduced.irreducible_kpoints.points[point].weight;
        member_count += static_cast<int>(
            gamma_reduced.star_members[point].size()
        );
        multiplicities.push_back(static_cast<int>(
            gamma_reduced.star_members[point].size()
        ));
    }
    std::sort(multiplicities.begin(), multiplicities.end());
    require_close(weight_sum, 1.0, 1.0e-14,
                  "Irreducible weights do not sum to one");
    require(member_count == 64,
            "Irreducible stars do not cover the full mesh");
    require(
        multiplicities ==
            std::vector<int>({1, 1, 3, 3, 6, 6, 8, 12, 12, 12}),
        "Unexpected cubic Gamma-mesh star multiplicities"
    );
    auto cubic_invariant = [](const Eigen::Vector3d& kpoint) {
        return std::cos(2.0 * M_PI * kpoint[0])
            + std::cos(2.0 * M_PI * kpoint[1])
            + std::cos(2.0 * M_PI * kpoint[2])
            + 0.3 * (
                std::cos(4.0 * M_PI * kpoint[0])
                + std::cos(4.0 * M_PI * kpoint[1])
                + std::cos(4.0 * M_PI * kpoint[2])
            );
    };
    double full_integral = 0.0;
    for (const KPoint& point : gamma.points) {
        full_integral += point.weight
            * cubic_invariant(point.frac_position);
    }
    double reduced_integral = 0.0;
    for (const KPoint& point :
         gamma_reduced.irreducible_kpoints.points) {
        reduced_integral += point.weight
            * cubic_invariant(point.frac_position);
    }
    require_close(
        reduced_integral, full_integral, 1.0e-14,
        "Irreducible weights did not reproduce a full-mesh integral"
    );

    const KPointSet monkhorst_pack =
        make_uniform_kpoint_mesh({{4, 4, 4}}, false);
    const KPointReductionResult mp_reduced =
        reduce_kpoints_by_symmetry(
            structure, monkhorst_pack, 1.0e-6, true
        );
    require(
        mp_reduced.irreducible_kpoints.points.size() == 4,
        "Cubic MP 4x4x4 mesh should reduce from 64 to 4 points"
    );

    const KPointSet anisotropic =
        make_uniform_kpoint_mesh({{2, 3, 4}}, true);
    const KPointReductionResult anisotropic_reduced =
        reduce_kpoints_by_symmetry(
            structure, anisotropic, 1.0e-6, true
        );
    require(
        anisotropic_reduced.irreducible_kpoints.points.size() == 12,
        "An anisotropic 2x3x4 mesh should retain only mesh-compatible symmetry"
    );
    require(
        anisotropic_reduced.mesh_compatible_operations.size() == 8,
        "An anisotropic cubic mesh should retain eight diagonal sign rotations"
    );
}

void test_atomic_displacement_lowers_symmetry() {
    const AtomicStructure diamond = diamond_primitive_structure();
    const auto symmetric_operations =
        find_space_group_operations(diamond, 1.0e-6);
    require(
        symmetric_operations.size() == 48,
        "Primitive diamond Si should expose 48 space-group operations"
    );
    const KPointReductionResult diamond_mesh =
        reduce_kpoints_by_symmetry(
            diamond,
            make_uniform_kpoint_mesh({{6, 6, 6}}, true),
            1.0e-6,
            true
        );
    require(
        diamond_mesh.irreducible_kpoints.points.size() == 16,
        "Diamond Si Gamma 6x6x6 mesh should reduce from 216 to 16 points"
    );

    AtomicStructure displaced = diamond;
    displaced.atoms[1].frac_position +=
        Eigen::Vector3d(0.013, -0.007, 0.011);
    const auto displaced_operations =
        find_space_group_operations(displaced, 1.0e-6);
    require(
        displaced_operations.size() < symmetric_operations.size(),
        "A generic atomic displacement did not lower crystal symmetry"
    );
}

void test_time_reversal_without_spatial_inversion() {
    const AtomicStructure structure = generic_triclinic_structure();
    const auto operations =
        find_space_group_operations(structure, 1.0e-7);
    require(
        operations.size() == 1,
        "The generic triclinic binary test structure should have identity only"
    );

    const KPointSet mesh =
        make_uniform_kpoint_mesh({{4, 4, 4}}, true);
    const KPointReductionResult without_time_reversal =
        reduce_kpoints_by_symmetry(
            structure, mesh, 1.0e-7, false
        );
    const KPointReductionResult with_time_reversal =
        reduce_kpoints_by_symmetry(
            structure, mesh, 1.0e-7, true
        );
    require(
        without_time_reversal.irreducible_kpoints.points.size() == 64,
        "Identity-only symmetry should not spatially reduce the mesh"
    );
    require(
        with_time_reversal.irreducible_kpoints.points.size() == 36,
        "Time reversal should reduce Gamma 4x4x4 from 64 to 36 points"
    );
    require(
        with_time_reversal.time_reversal_used,
        "Time-reversal use was not recorded"
    );
}

void test_explicit_points_are_not_reinterpreted() {
    KPointSet explicit_points;
    explicit_points.uniform_mesh = false;
    explicit_points.description = "Explicit";
    explicit_points.points = {
        {Eigen::Vector3d(0.0, 0.0, 0.0), 0.2},
        {Eigen::Vector3d(0.25, 0.0, 0.0), 0.8}
    };
    const KPointReductionResult result =
        reduce_kpoints_by_symmetry(
            simple_cubic_structure(),
            explicit_points,
            1.0e-6,
            true
        );
    require(
        result.irreducible_kpoints.points.size() == 2 &&
        !result.reduction_applied &&
        result.space_group_operations.empty(),
        "Explicit weighted k points must remain untouched"
    );
}

void test_fft_dimensions_and_scalar_symmetrization() {
    SpaceGroupOperation swap_xy;
    swap_xy.rotation <<
        0, 1, 0,
        1, 0, 0,
        0, 0, 1;
    const std::array<int, 3> compatible =
        symmetry_compatible_fft_dimensions(
            {{48, 60, 72}}, {swap_xy}
        );
    require(
        compatible == std::array<int, 3>{{60, 60, 72}},
        "FFT dimensions mixed by symmetry were not equalized"
    );

    AtomicStructure structure;
    structure.lattice_bohr = 8.0 * Eigen::Matrix3d::Identity();
    structure.species_order = {"X"};
    StructureAtom first;
    first.element = "X";
    first.frac_position = Eigen::Vector3d::Zero();
    StructureAtom second = first;
    second.frac_position = Eigen::Vector3d(0.5, 0.0, 0.0);
    structure.atoms = {first, second};
    const Lattice lattice(
        structure.lattice_bohr.col(0),
        structure.lattice_bohr.col(1),
        structure.lattice_bohr.col(2)
    );
    FFTWorkspace fft(FFTGrid(4, 4, 4), 1);

    SpaceGroupOperation identity;
    identity.atom_mapping = {0, 1};
    SpaceGroupOperation half_translation;
    half_translation.translation = Eigen::Vector3d(0.5, 0.0, 0.0);
    half_translation.atom_mapping = {1, 0};
    const std::vector<SpaceGroupOperation> operations{
        identity, half_translation
    };

    std::vector<double> field(fft.grid.ngrid, 0.0);
    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                field[fft.grid.index(i, j, k)] =
                    1.0 + std::cos(
                        2.0 * M_PI * static_cast<double>(i)
                        / static_cast<double>(fft.grid.n1)
                    );
            }
        }
    }
    symmetrize_scalar_field(
        lattice, fft, operations, field
    );
    for (double value : field) {
        require_close(
            value, 1.0, 1.0e-13,
            "Half-translation did not remove an odd Fourier component"
        );
    }

    std::vector<Eigen::Vector3d> vectors{
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(3.0, 0.0, 0.0)
    };
    symmetrize_atomic_vectors(
        structure, operations, vectors
    );
    require_close(vectors[0][0], 2.0, 1.0e-14,
                  "Force symmetrization failed on atom 0");
    require_close(vectors[1][0], 2.0, 1.0e-14,
                  "Force symmetrization failed on atom 1");
}

void test_reciprocal_symmetrization_preserves_fft_buffers() {
    const Lattice lattice(
        Eigen::Vector3d(8.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 8.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 8.0)
    );
    FFTWorkspace fft(FFTGrid(6, 6, 6), 1);
    const std::complex<double>* reciprocal_buffer =
        fft.reciprocal_grid.data();

    std::vector<SpaceGroupOperation> operations(4);
    for (int operation = 0; operation < 4; ++operation) {
        operations[operation].translation =
            Eigen::Vector3d(
                0.25 * static_cast<double>(operation),
                0.0,
                0.0
            );
    }

    std::vector<double> field(fft.grid.ngrid, 0.0);
    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                field[fft.grid.index(i, j, k)] =
                    1.0 + std::cos(
                        2.0 * M_PI * static_cast<double>(i)
                        / static_cast<double>(fft.grid.n1)
                    );
            }
        }
    }

    /*
     * A quarter-cell translation is incompatible with a six-point real
     * grid, so this call must use reciprocal-space symmetrization.
     */
    symmetrize_scalar_field(lattice, fft, operations, field);

    require(
        fft.reciprocal_grid.data() == reciprocal_buffer,
        "Reciprocal symmetrization changed an FFTW-bound buffer address"
    );
    for (double value : field) {
        require_close(
            value, 1.0, 1.0e-12,
            "Quarter translations did not remove a Fourier component"
        );
    }
}

void test_full_and_irreducible_scf_agree() {
    AtomicStructure structure;
    structure.comment = "cubic H symmetry integration test";
    structure.lattice_bohr =
        10.0 * Eigen::Matrix3d::Identity();
    structure.species_order = {"H"};
    StructureAtom atom;
    atom.element = "H";
    atom.frac_position = Eigen::Vector3d::Zero();
    structure.atoms.push_back(atom);

    CalculationConfig full_config;
    full_config.pseudopotential_paths["H"] =
        std::string(TEST_DATA_DIR) + "/minimal_local_nc.upf";
    full_config.ecut_hartree = 2.0;
    full_config.kpoints =
        make_uniform_kpoint_mesh({{2, 2, 2}}, true);
    full_config.kpoint_symmetry.enabled = false;
    full_config.scf.occupation_mode =
        OccupationMode::DegeneracyAwareZeroT;
    full_config.scf.max_iterations = 100;
    full_config.scf.energy_tolerance = 1.0e-8;
    full_config.scf.eigensolver_initial_tolerance = 1.0e-7;
    full_config.scf.eigensolver_tolerance = 1.0e-8;
    full_config.scf.density_tolerance = 1.0e-7;
    full_config.scf.verbosity = SCFVerbosity::Silent;
    full_config.scf.mixing_alpha = 0.25;

    CalculationConfig reduced_config = full_config;
    reduced_config.kpoint_symmetry.enabled = true;
    const SinglePointResult full =
        run_single_point(structure, full_config);
    const SinglePointResult reduced =
        run_single_point(structure, reduced_config);
    require(
        full.converged && reduced.converged,
        "Full or irreducible integration-test SCF did not converge"
    );
    require(
        full.full_kpoint_count == 8 &&
        full.irreducible_kpoint_count == 8 &&
        reduced.full_kpoint_count == 8 &&
        reduced.irreducible_kpoint_count == 4,
        "Unexpected full/irreducible k-point counts in SCF integration test"
    );
    require_close(
        reduced.scf.variational_energy,
        full.scf.variational_energy,
        2.0e-8,
        "Full and irreducible SCF energies disagree"
    );
    require(
        reduced.scf.density.size() == full.scf.density.size(),
        "Full and irreducible density grids have different sizes"
    );
    double maximum_density_difference = 0.0;
    for (int point = 0;
         point < static_cast<int>(full.scf.density.size());
         ++point) {
        maximum_density_difference = std::max(
            maximum_density_difference,
            std::abs(
                reduced.scf.density[point]
                - full.scf.density[point]
            )
        );
    }
    require(
        maximum_density_difference < 2.0e-8,
        "Full and irreducible SCF densities disagree"
    );
    require(
        full.forces.total.size() == 1 &&
        reduced.forces.total.size() == 1 &&
        (full.forces.total[0]
            - reduced.forces.total[0]).norm() < 1.0e-8,
        "Full and irreducible SCF forces disagree"
    );
}

} // namespace

int main() {
    try {
        test_cubic_space_group_and_mesh_reduction();
        test_atomic_displacement_lowers_symmetry();
        test_time_reversal_without_spatial_inversion();
        test_explicit_points_are_not_reinterpreted();
        test_fft_dimensions_and_scalar_symmetrization();
        test_reciprocal_symmetrization_preserves_fft_buffers();
        test_full_and_irreducible_scf_agree();
        std::cout << "K-point symmetry tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "K-point symmetry test failed: "
                  << error.what() << "\n";
        return 1;
    }
}
