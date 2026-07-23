#include "relaxation.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {

    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual = " + std::to_string(actual)
            + ", expected = " + std::to_string(expected)
        );
    }
}

AtomicStructure make_two_atom_structure() {
    AtomicStructure structure;
    structure.comment = "BFGS harmonic test";
    structure.lattice_bohr = 20.0 * Eigen::Matrix3d::Identity();
    structure.species_order = {"X"};

    StructureAtom fixed;
    fixed.element = "X";
    fixed.frac_position = Eigen::Vector3d(0.25, 0.25, 0.25);
    fixed.movable = {{false, false, false}};
    structure.atoms.push_back(fixed);

    StructureAtom moving;
    moving.element = "X";
    moving.frac_position = Eigen::Vector3d(0.50, 0.50, 0.50);
    structure.atoms.push_back(moving);
    return structure;
}

FixedCellEvaluator harmonic_evaluator(
    const std::vector<Eigen::Vector3d>& reference_positions,
    double curvature,
    bool& warm_start_seen) {

    return [
        reference_positions,
        curvature,
        &warm_start_seen
    ](
        const AtomicStructure& structure,
        const KPointSCFInitialGuess& initial_guess) {

        if (!initial_guess.density.empty() &&
            !initial_guess.orbitals.empty() &&
            initial_guess.orbitals[0].size() != 0) {
            warm_start_seen = true;
        }

        SinglePointResult result;
        result.converged = true;
        result.scf.converged = true;
        result.scf.iterations =
            initial_guess.density.empty() ? 7 : 2;
        result.forces.total.assign(
            structure.atoms.size(), Eigen::Vector3d::Zero()
        );

        double energy = 0.0;
        for (int iatom = 0;
             iatom < static_cast<int>(structure.atoms.size());
             ++iatom) {
            const Eigen::Vector3d position =
                structure.lattice_bohr
                * structure.atoms[iatom].frac_position;
            const Eigen::Vector3d displacement =
                position - reference_positions[iatom];
            energy += 0.5 * curvature * displacement.squaredNorm();
            result.forces.total[iatom] =
                -curvature * displacement;
        }
        result.scf.variational_energy = energy;
        result.scf.density = {1.0};
        result.scf.kpoints.resize(1);
        result.scf.kpoints[0].orbitals =
            Eigen::MatrixXcd::Identity(1, 1);
        return result;
    };
}

void test_selective_dynamics_and_warm_start() {
    AtomicStructure structure = make_two_atom_structure();
    structure.atoms[1].movable = {{true, false, true}};

    std::vector<Eigen::Vector3d> reference_positions{
        structure.lattice_bohr * structure.atoms[0].frac_position,
        structure.lattice_bohr * structure.atoms[1].frac_position
    };
    const Eigen::Vector3d initial_displacement(0.20, 0.30, -0.10);
    structure.atoms[1].frac_position +=
        structure.lattice_bohr.inverse() * initial_displacement;

    CalculationConfig config;
    config.calculation = CalculationType::Relax;
    config.relaxation.force_tolerance_ha_bohr = 1.0e-11;
    config.relaxation.max_step_angstrom = 1.0;
    config.relaxation.initial_curvature_ha_bohr2 = 0.5;
    config.relaxation.energy_increase_tolerance_ha = 0.0;
    config.relaxation.contcar_path.clear();
    config.relaxation.trajectory_path.clear();

    bool warm_start_seen = false;
    const RelaxationResult result = run_fixed_cell_relaxation(
        structure,
        config,
        harmonic_evaluator(
            reference_positions, 0.5, warm_start_seen
        )
    );

    require_true(result.converged, "Selective-dynamics BFGS did not converge");
    require_true(result.ionic_steps == 1,
                 "Exact harmonic inverse Hessian should converge in one step");
    require_true(warm_start_seen,
                 "The accepted electronic state was not used as a warm start");

    const Eigen::Vector3d final_position =
        result.structure.lattice_bohr
        * result.structure.atoms[1].frac_position;
    require_close(final_position[0], reference_positions[1][0], 1.0e-12,
                  "Movable x coordinate");
    require_close(
        final_position[1],
        reference_positions[1][1] + initial_displacement[1],
        1.0e-12,
        "Fixed y coordinate"
    );
    require_close(final_position[2], reference_positions[1][2], 1.0e-12,
                  "Movable z coordinate");
}

void test_energy_backtracking() {
    AtomicStructure structure = make_two_atom_structure();
    structure.atoms[1].movable = {{true, false, false}};

    std::vector<Eigen::Vector3d> reference_positions{
        structure.lattice_bohr * structure.atoms[0].frac_position,
        structure.lattice_bohr * structure.atoms[1].frac_position
    };
    structure.atoms[1].frac_position[0] += 1.0 / 20.0;

    CalculationConfig config;
    config.calculation = CalculationType::Relax;
    config.relaxation.force_tolerance_ha_bohr = 1.0e-11;
    config.relaxation.max_step_angstrom = 20.0 / ANGSTROM_TO_BOHR;
    config.relaxation.initial_curvature_ha_bohr2 = 0.10;
    config.relaxation.max_backtracks = 6;
    config.relaxation.energy_increase_tolerance_ha = 0.0;
    config.relaxation.contcar_path.clear();
    config.relaxation.trajectory_path.clear();

    bool warm_start_seen = false;
    const RelaxationResult result = run_fixed_cell_relaxation(
        structure,
        config,
        harmonic_evaluator(
            reference_positions, 1.0, warm_start_seen
        )
    );

    require_true(result.converged, "Backtracked BFGS did not converge");
    require_true(result.history.size() >= 3,
                 "Backtracking test has too few accepted points");
    require_true(result.history[1].backtracks == 3,
                 "The uphill BFGS trial was not backtracked three times");
    for (int step = 1; step < static_cast<int>(result.history.size()); ++step) {
        require_true(
            result.history[step].free_energy_ha <=
                result.history[step - 1].free_energy_ha,
            "An uphill ionic step was accepted"
        );
    }
}

} // namespace

int main() {
    try {
        test_selective_dynamics_and_warm_start();
        test_energy_backtracking();
        std::cout << "Fixed-cell BFGS relaxation tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Relaxation test failed: " << error.what() << "\n";
        return 1;
    }
}
