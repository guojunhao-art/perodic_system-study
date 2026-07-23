#include "relaxation.hpp"

#include "parallel.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

struct DegreeOfFreedom {
    int atom = 0;
    int basis_index = 0;
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
};

std::vector<DegreeOfFreedom> active_degrees_of_freedom(
    const AtomicStructure& structure) {

    std::vector<DegreeOfFreedom> degrees;
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        std::vector<Eigen::Vector3d> atom_directions;
        for (int input_direction = 0; input_direction < 3;
             ++input_direction) {
            if (!structure.atoms[iatom].movable[input_direction]) {
                continue;
            }

            Eigen::Vector3d direction =
                Eigen::Vector3d::Unit(input_direction);
            if (structure.selective_dynamics &&
                !structure.selective_dynamics_cartesian) {
                direction =
                    structure.lattice_bohr.col(input_direction);
            }
            for (const Eigen::Vector3d& previous : atom_directions) {
                direction -= direction.dot(previous) * previous;
            }
            const double norm = direction.norm();
            if (!std::isfinite(norm) || norm < 1.0e-12) {
                throw std::runtime_error(
                    "Selective-dynamics directions are linearly dependent."
                );
            }
            direction /= norm;
            const int basis_index =
                static_cast<int>(atom_directions.size());
            atom_directions.push_back(direction);
            degrees.push_back({iatom, basis_index, direction});
        }
    }
    return degrees;
}

Eigen::VectorXd projected_gradient(
    const AtomicStructure& structure,
    const std::vector<DegreeOfFreedom>& degrees,
    const std::vector<Eigen::Vector3d>& forces) {

    if (forces.size() != structure.atoms.size()) {
        throw std::runtime_error(
            "The force count does not match the structure in BFGS."
        );
    }

    Eigen::VectorXd gradient(degrees.size());
    for (int index = 0; index < static_cast<int>(degrees.size()); ++index) {
        const DegreeOfFreedom& degree = degrees[index];
        gradient[index] =
            -forces[degree.atom].dot(degree.direction);
    }

    /*
     * A fully free periodic structure has three exact translational zero
     * modes. Remove the corresponding mean gradient so small SCF force
     * residuals cannot translate the whole cell.
     */
    bool all_atoms_fully_movable = true;
    for (const StructureAtom& atom : structure.atoms) {
        for (bool movable : atom.movable) {
            all_atoms_fully_movable =
                all_atoms_fully_movable && movable;
        }
    }
    if (!all_atoms_fully_movable) {
        return gradient;
    }

    for (int basis_index = 0; basis_index < 3; ++basis_index) {
        double mean = 0.0;
        int count = 0;
        for (int index = 0;
             index < static_cast<int>(degrees.size());
             ++index) {
            if (degrees[index].basis_index == basis_index) {
                mean += gradient[index];
                ++count;
            }
        }
        if (count > 0) {
            mean /= static_cast<double>(count);
            for (int index = 0;
                 index < static_cast<int>(degrees.size());
                 ++index) {
                if (degrees[index].basis_index == basis_index) {
                    gradient[index] -= mean;
                }
            }
        }
    }
    return gradient;
}

double maximum_atomic_norm(
    int atom_count,
    const std::vector<DegreeOfFreedom>& degrees,
    const Eigen::VectorXd& values) {

    std::vector<Eigen::Vector3d> atomic_vectors(
        atom_count, Eigen::Vector3d::Zero()
    );
    for (int index = 0; index < static_cast<int>(degrees.size()); ++index) {
        atomic_vectors[degrees[index].atom] +=
            values[index] * degrees[index].direction;
    }

    double maximum = 0.0;
    for (const Eigen::Vector3d& vector : atomic_vectors) {
        maximum = std::max(maximum, vector.norm());
    }
    return maximum;
}

void limit_maximum_atomic_step(
    int atom_count,
    const std::vector<DegreeOfFreedom>& degrees,
    double maximum_step_bohr,
    Eigen::VectorXd& step) {

    const double largest =
        maximum_atomic_norm(atom_count, degrees, step);
    if (largest > maximum_step_bohr) {
        step *= maximum_step_bohr / largest;
    }
}

double wrap_fractional(double value) {
    value -= std::floor(value);
    if (value >= 1.0 - 1.0e-14) {
        value = 0.0;
    }
    return value;
}

AtomicStructure displaced_structure(
    const AtomicStructure& structure,
    const std::vector<DegreeOfFreedom>& degrees,
    const Eigen::VectorXd& step) {

    AtomicStructure displaced = structure;
    std::vector<Eigen::Vector3d> cartesian_steps(
        structure.atoms.size(), Eigen::Vector3d::Zero()
    );
    for (int index = 0; index < static_cast<int>(degrees.size()); ++index) {
        cartesian_steps[degrees[index].atom] +=
            step[index] * degrees[index].direction;
    }

    const Eigen::Matrix3d inverse_lattice =
        structure.lattice_bohr.inverse();
    for (int iatom = 0;
         iatom < static_cast<int>(structure.atoms.size());
         ++iatom) {
        displaced.atoms[iatom].frac_position +=
            inverse_lattice * cartesian_steps[iatom];
        for (int direction = 0; direction < 3; ++direction) {
            displaced.atoms[iatom].frac_position[direction] =
                wrap_fractional(
                    displaced.atoms[iatom].frac_position[direction]
                );
        }
    }
    return displaced;
}

KPointSCFInitialGuess make_initial_guess(
    const KPointSCFResult& result) {

    KPointSCFInitialGuess guess;
    guess.density = result.density;
    guess.spin_densities = result.spin_densities;
    guess.orbitals.resize(result.kpoints.size());
    for (int ik = 0; ik < static_cast<int>(result.kpoints.size()); ++ik) {
        guess.orbitals[ik] = result.kpoints[ik].orbitals;
    }
    return guess;
}

const char* search_direction_name(IonicSearchDirection direction) {
    switch (direction) {
    case IonicSearchDirection::None:
        return "none";
    case IonicSearchDirection::InitialHessian:
        return "initial-Hessian";
    case IonicSearchDirection::BFGS:
        return "BFGS";
    case IonicSearchDirection::SteepestDescentReset:
        return "steepest-descent(reset)";
    }
    return "unknown";
}

void print_ionic_step(
    std::ostream& out,
    const IonicStepSummary& summary) {

    const auto flags = out.flags();
    const auto precision = out.precision();
    out << " ION: " << std::setw(4) << summary.step
        << std::scientific << std::setprecision(12)
        << "  F= " << std::setw(20) << summary.free_energy_ha
        << "  dF= " << std::setw(14) << summary.energy_change_ha
        << "  max|force|= " << std::setw(13)
        << summary.maximum_force_ha_bohr
        << "  SCF= " << std::setw(3) << summary.scf_iterations
        << "\n";
    if (summary.search_direction != IonicSearchDirection::None) {
        out << "       search= "
            << search_direction_name(summary.search_direction)
            << "  dE(linear)= " << std::setw(13)
            << summary.linear_energy_change_ha
            << "  max|dR|= " << std::setw(13)
            << summary.maximum_displacement_angstrom << " Angstrom";
        if (summary.backtracks > 0) {
            out << "  backtracks= " << summary.backtracks;
        }
        out << "\n";
    }
    out.flags(flags);
    out.precision(precision);
}

void reset_trajectory_collective(const std::string& path) {
    std::string local_error;
    if (parallel::is_root() && !path.empty()) {
        try {
            std::ofstream output(path, std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "Cannot create relaxation trajectory: " + path
                );
            }
        } catch (const std::exception& error) {
            local_error = error.what();
        }
    }
    const std::string error = parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

void append_xyz(
    const std::string& path,
    const AtomicStructure& structure,
    const IonicStepSummary& summary) {

    if (path.empty()) {
        return;
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error(
            "Cannot append relaxation trajectory: " + path
        );
    }

    output << structure.atoms.size() << "\n"
        << "step=" << summary.step
        << " free_energy_ha=" << std::setprecision(16)
        << summary.free_energy_ha
        << " energy_change_ha=" << summary.energy_change_ha
        << " max_force_ha_bohr="
        << summary.maximum_force_ha_bohr
        << " max_displacement_angstrom="
        << summary.maximum_displacement_angstrom
        << " scf_iterations=" << summary.scf_iterations
        << " search=" << search_direction_name(summary.search_direction)
        << "\n";
    for (const StructureAtom& atom : structure.atoms) {
        const Eigen::Vector3d position_angstrom =
            structure.lattice_bohr * atom.frac_position
            / ANGSTROM_TO_BOHR;
        output << atom.element
            << std::fixed << std::setprecision(12)
            << " " << position_angstrom[0]
            << " " << position_angstrom[1]
            << " " << position_angstrom[2] << "\n";
    }
}

void write_relaxation_outputs(
    const CalculationConfig& config,
    const AtomicStructure& structure,
    const IonicStepSummary& summary) {

    std::string local_error;
    if (parallel::is_root()) {
        try {
            if (!config.relaxation.contcar_path.empty()) {
                write_poscar(
                    config.relaxation.contcar_path,
                    structure,
                    structure.comment + " (fixed-cell BFGS step "
                        + std::to_string(summary.step) + ")"
                );
            }
            append_xyz(
                config.relaxation.trajectory_path,
                structure,
                summary
            );
        } catch (const std::exception& error) {
            local_error = error.what();
        }
    }
    const std::string error = parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

} // namespace

RelaxationResult run_fixed_cell_relaxation(
    const AtomicStructure& initial_structure,
    const CalculationConfig& config,
    const FixedCellEvaluator& evaluator,
    std::ostream* log_stream) {

    if (!evaluator) {
        throw std::runtime_error("The fixed-cell evaluator is empty.");
    }
    if (config.relaxation.algorithm != IonAlgorithm::BFGS) {
        throw std::runtime_error("Only the BFGS ion algorithm is implemented.");
    }

    const std::vector<DegreeOfFreedom> degrees =
        active_degrees_of_freedom(initial_structure);
    if (degrees.empty()) {
        throw std::runtime_error(
            "A relaxation requires at least one movable coordinate."
        );
    }

    const RelaxationOptions& options = config.relaxation;
    const int variable_count = static_cast<int>(degrees.size());
    Eigen::MatrixXd inverse_hessian =
        Eigen::MatrixXd::Identity(variable_count, variable_count)
        / options.initial_curvature_ha_bohr2;
    bool has_bfgs_curvature = false;

    RelaxationResult result;
    result.structure = initial_structure;

    reset_trajectory_collective(options.trajectory_path);

    KPointSCFInitialGuess empty_guess;
    SinglePointResult current = evaluator(result.structure, empty_guess);
    if (!current.converged) {
        result.stop_reason = "initial electronic SCF did not converge";
        result.electronic = std::move(current);
        return result;
    }

    Eigen::VectorXd gradient = projected_gradient(
        result.structure, degrees, current.forces.total
    );
    double maximum_force = maximum_atomic_norm(
        static_cast<int>(result.structure.atoms.size()),
        degrees,
        gradient
    );
    IonicStepSummary initial_summary;
    initial_summary.free_energy_ha = current.scf.variational_energy;
    initial_summary.maximum_force_ha_bohr = maximum_force;
    initial_summary.scf_iterations = current.scf.iterations;
    result.history.push_back(initial_summary);
    if (log_stream) {
        *log_stream << "\n"
            << " FIXED-CELL IONIC RELAXATION (BFGS)\n"
            << " -------------------------------------------------------------------------------\n";
        print_ionic_step(*log_stream, initial_summary);
    }
    write_relaxation_outputs(
        config, result.structure, initial_summary
    );

    if (maximum_force <= options.force_tolerance_ha_bohr) {
        result.converged = true;
        result.stop_reason = "force tolerance reached at the initial structure";
        result.electronic = std::move(current);
        return result;
    }

    for (int ionic_step = 1;
         ionic_step <= options.max_ionic_steps;
         ++ionic_step) {
        IonicSearchDirection search_direction =
            has_bfgs_curvature
            ? IonicSearchDirection::BFGS
            : IonicSearchDirection::InitialHessian;
        Eigen::VectorXd direction = -inverse_hessian * gradient;
        const double norm_product = direction.norm() * gradient.norm();
        if (!direction.allFinite() ||
            direction.dot(gradient) >=
                -std::numeric_limits<double>::epsilon() * norm_product) {
            inverse_hessian.setIdentity();
            inverse_hessian /= options.initial_curvature_ha_bohr2;
            has_bfgs_curvature = false;
            direction = -inverse_hessian * gradient;
            search_direction =
                IonicSearchDirection::SteepestDescentReset;
        }

        limit_maximum_atomic_step(
            static_cast<int>(result.structure.atoms.size()),
            degrees,
            options.max_step_angstrom * ANGSTROM_TO_BOHR,
            direction
        );

        const KPointSCFInitialGuess accepted_guess =
            make_initial_guess(current.scf);
        const double previous_energy = current.scf.variational_energy;
        bool accepted = false;
        AtomicStructure trial_structure;
        SinglePointResult trial;
        Eigen::VectorXd actual_step;
        int accepted_backtracks = 0;

        for (int backtrack = 0;
             backtrack <= options.max_backtracks;
             ++backtrack) {
            const double scale = std::ldexp(1.0, -backtrack);
            actual_step = scale * direction;
            trial_structure = displaced_structure(
                result.structure, degrees, actual_step
            );
            trial = evaluator(trial_structure, accepted_guess);
            if (trial.converged &&
                trial.scf.variational_energy <= previous_energy
                    + options.energy_increase_tolerance_ha) {
                accepted = true;
                accepted_backtracks = backtrack;
                break;
            }
        }

        if (!accepted) {
            result.stop_reason =
                "BFGS step was rejected after maximum backtracking";
            break;
        }

        const double linear_energy_change =
            gradient.dot(actual_step);
        const double maximum_displacement_angstrom =
            maximum_atomic_norm(
                static_cast<int>(result.structure.atoms.size()),
                degrees,
                actual_step
            ) / ANGSTROM_TO_BOHR;
        Eigen::VectorXd trial_gradient = projected_gradient(
            trial_structure, degrees, trial.forces.total
        );
        const Eigen::VectorXd gradient_change =
            trial_gradient - gradient;
        const double curvature =
            actual_step.dot(gradient_change);
        const double curvature_scale =
            actual_step.norm() * gradient_change.norm();
        if (std::isfinite(curvature) &&
            curvature >
                options.curvature_tolerance * curvature_scale) {
            const double rho = 1.0 / curvature;
            const Eigen::MatrixXd identity =
                Eigen::MatrixXd::Identity(variable_count, variable_count);
            const Eigen::MatrixXd left =
                identity - rho * actual_step * gradient_change.transpose();
            inverse_hessian =
                left * inverse_hessian * left.transpose()
                + rho * actual_step * actual_step.transpose();
            inverse_hessian =
                0.5 * (inverse_hessian + inverse_hessian.transpose());
            if (!inverse_hessian.allFinite()) {
                inverse_hessian.setIdentity();
                inverse_hessian /= options.initial_curvature_ha_bohr2;
                has_bfgs_curvature = false;
            } else {
                has_bfgs_curvature = true;
            }
        }

        result.structure = std::move(trial_structure);
        current = std::move(trial);
        gradient = std::move(trial_gradient);
        maximum_force = maximum_atomic_norm(
            static_cast<int>(result.structure.atoms.size()),
            degrees,
            gradient
        );

        IonicStepSummary summary;
        summary.step = ionic_step;
        summary.free_energy_ha = current.scf.variational_energy;
        summary.energy_change_ha =
            current.scf.variational_energy - previous_energy;
        summary.maximum_force_ha_bohr = maximum_force;
        summary.maximum_displacement_angstrom =
            maximum_displacement_angstrom;
        summary.linear_energy_change_ha = linear_energy_change;
        summary.scf_iterations = current.scf.iterations;
        summary.backtracks = accepted_backtracks;
        summary.search_direction = search_direction;
        result.history.push_back(summary);
        result.ionic_steps = ionic_step;
        if (log_stream) {
            print_ionic_step(*log_stream, summary);
        }
        write_relaxation_outputs(config, result.structure, summary);

        if (maximum_force <= options.force_tolerance_ha_bohr) {
            result.converged = true;
            result.stop_reason = "force tolerance reached";
            break;
        }
    }

    if (!result.converged && result.stop_reason.empty()) {
        result.stop_reason = "maximum ionic steps reached";
    }
    result.electronic = std::move(current);
    if (log_stream) {
        *log_stream
            << " -------------------------------------------------------------------------------\n"
            << "  relaxation status: " << result.stop_reason << "\n";
    }
    return result;
}

RelaxationResult run_fixed_cell_relaxation(
    const AtomicStructure& initial_structure,
    const CalculationConfig& config,
    std::ostream* log_stream) {

    bool print_setup = true;
    const FixedCellEvaluator evaluator =
        [&config, log_stream, &print_setup](
            const AtomicStructure& structure,
            const KPointSCFInitialGuess& initial_guess) {
            const bool print_this_setup = print_setup;
            print_setup = false;
            return run_single_point(
                structure,
                config,
                log_stream,
                initial_guess,
                print_this_setup
            );
        };
    return run_fixed_cell_relaxation(
        initial_structure, config, evaluator, log_stream
    );
}

void write_poscar(
    const std::string& path,
    const AtomicStructure& structure,
    const std::string& comment) {

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot write POSCAR file: " + path);
    }

    bool selective_dynamics = structure.selective_dynamics;
    for (const StructureAtom& atom : structure.atoms) {
        for (bool movable : atom.movable) {
            selective_dynamics = selective_dynamics || !movable;
        }
    }

    output << comment << "\n1.0\n"
        << std::fixed << std::setprecision(12);
    for (int vector = 0; vector < 3; ++vector) {
        const Eigen::Vector3d lattice_angstrom =
            structure.lattice_bohr.col(vector) / ANGSTROM_TO_BOHR;
        output << std::setw(18) << lattice_angstrom[0]
            << " " << std::setw(18) << lattice_angstrom[1]
            << " " << std::setw(18) << lattice_angstrom[2] << "\n";
    }

    for (int ispecies = 0;
         ispecies < static_cast<int>(structure.species_order.size());
         ++ispecies) {
        if (ispecies > 0) {
            output << " ";
        }
        output << structure.species_order[ispecies];
    }
    output << "\n";
    for (int ispecies = 0;
         ispecies < static_cast<int>(structure.species_order.size());
         ++ispecies) {
        if (ispecies > 0) {
            output << " ";
        }
        const std::string& species = structure.species_order[ispecies];
        output << std::count_if(
            structure.atoms.begin(),
            structure.atoms.end(),
            [&species](const StructureAtom& atom) {
                return atom.element == species;
            }
        );
    }
    output << "\n";
    if (selective_dynamics) {
        output << "Selective dynamics\n";
    }
    const bool cartesian = selective_dynamics &&
        (!structure.selective_dynamics ||
         structure.selective_dynamics_cartesian);
    output << (cartesian ? "Cartesian\n" : "Direct\n");

    for (const std::string& species : structure.species_order) {
        for (const StructureAtom& atom : structure.atoms) {
            if (atom.element != species) {
                continue;
            }
            Eigen::Vector3d coordinates(
                    wrap_fractional(atom.frac_position[0]),
                    wrap_fractional(atom.frac_position[1]),
                    wrap_fractional(atom.frac_position[2])
            );
            if (cartesian) {
                coordinates =
                    structure.lattice_bohr * atom.frac_position
                    / ANGSTROM_TO_BOHR;
            }
            output << std::setw(18) << coordinates[0]
                << " " << std::setw(18) << coordinates[1]
                << " " << std::setw(18) << coordinates[2];
            if (selective_dynamics) {
                for (int direction = 0; direction < 3; ++direction) {
                    output << "  "
                        << (atom.movable[direction] ? "T" : "F");
                }
            }
            output << "\n";
        }
    }
}
