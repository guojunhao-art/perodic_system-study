#include "dos.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace {

constexpr double inverse_sqrt_two_pi =
    0.398942280401432677939946059934381868;
constexpr double inverse_sqrt_two =
    0.707106781186547524400844362104849039;
constexpr double automatic_window_sigma_margin = 5.0;
constexpr double weight_tolerance = 1.0e-10;

double gaussian_density(double difference, double sigma) {
    const double scaled = difference / sigma;
    return inverse_sqrt_two_pi / sigma
        * std::exp(-0.5 * scaled * scaled);
}

double gaussian_cumulative(double difference, double sigma) {
    return 0.5 * (
        1.0 + std::erf(difference / sigma * inverse_sqrt_two)
    );
}

void validate_dos_inputs(
    const KPointSCFResult& scf,
    const SCFOptions& scf_options,
    const DensityOfStatesOptions& dos_options) {

    if (!scf.converged) {
        throw std::runtime_error(
            "Density of states requires a converged SCF result."
        );
    }
    if (scf_options.nspin != 1 && scf_options.nspin != 2) {
        throw std::runtime_error("DOS nspin must be 1 or 2.");
    }
    if (scf_options.nbands <= 0) {
        throw std::runtime_error("DOS nbands must be positive.");
    }
    if (dos_options.points < 2) {
        throw std::runtime_error("dos_points must be at least 2.");
    }
    if (!std::isfinite(dos_options.smearing_ev) ||
        dos_options.smearing_ev <= 0.0) {
        throw std::runtime_error(
            "dos_smearing_ev must be finite and positive."
        );
    }
    if (!std::isfinite(scf.occupations.mu)) {
        throw std::runtime_error("The SCF Fermi energy is not finite.");
    }
    if (scf.kpoints.empty() ||
        scf.kpoints.size() % scf_options.nspin != 0) {
        throw std::runtime_error(
            "The SCF electronic-state count is incompatible with nspin."
        );
    }
    if (!dos_options.energy_min_auto &&
        !std::isfinite(dos_options.energy_min_ev)) {
        throw std::runtime_error("dos_energy_min_ev must be finite or auto.");
    }
    if (!dos_options.energy_max_auto &&
        !std::isfinite(dos_options.energy_max_ev)) {
        throw std::runtime_error("dos_energy_max_ev must be finite or auto.");
    }
    if (!dos_options.energy_min_auto &&
        !dos_options.energy_max_auto &&
        dos_options.energy_min_ev >= dos_options.energy_max_ev) {
        throw std::runtime_error(
            "dos_energy_min_ev must be smaller than dos_energy_max_ev."
        );
    }
}

} // namespace

DensityOfStatesResult compute_density_of_states(
    const KPointSCFResult& scf,
    const SCFOptions& scf_options,
    const DensityOfStatesOptions& dos_options) {

    validate_dos_inputs(scf, scf_options, dos_options);

    DensityOfStatesResult result;
    result.nspin = scf_options.nspin;
    result.nbands = scf_options.nbands;
    result.kpoint_count =
        static_cast<int>(scf.kpoints.size()) / result.nspin;
    result.fermi_energy_ha = scf.occupations.mu;
    result.gaussian_sigma_ha =
        dos_options.smearing_ev / HARTREE_TO_EV;

    double minimum_eigenvalue = std::numeric_limits<double>::infinity();
    double maximum_eigenvalue = -std::numeric_limits<double>::infinity();
    std::vector<double> spin_weight_sums(result.nspin, 0.0);
    std::vector<int> spin_state_counts(result.nspin, 0);
    std::vector<std::vector<bool>> state_seen(
        result.nspin,
        std::vector<bool>(result.kpoint_count, false)
    );
    for (const KPointElectronicState& state : scf.kpoints) {
        if (state.spin_channel < 0 ||
            state.spin_channel >= result.nspin) {
            throw std::runtime_error(
                "An SCF electronic state has an invalid spin channel."
            );
        }
        if (state.kpoint_index < 0 ||
            state.kpoint_index >= result.kpoint_count) {
            throw std::runtime_error(
                "An SCF electronic state has an invalid k-point index."
            );
        }
        if (state_seen[state.spin_channel][state.kpoint_index]) {
            throw std::runtime_error(
                "A DOS spin/k-point electronic state is duplicated."
            );
        }
        state_seen[state.spin_channel][state.kpoint_index] = true;
        if (!std::isfinite(state.weight) || state.weight <= 0.0) {
            throw std::runtime_error(
                "DOS k-point weights must be finite and positive."
            );
        }
        if (state.eigenvalues.size() != result.nbands) {
            throw std::runtime_error(
                "An SCF electronic state has the wrong number of bands."
            );
        }
        spin_weight_sums[state.spin_channel] += state.weight;
        ++spin_state_counts[state.spin_channel];
        for (int band = 0; band < result.nbands; ++band) {
            const double eigenvalue = state.eigenvalues[band];
            if (!std::isfinite(eigenvalue)) {
                throw std::runtime_error(
                    "DOS eigenvalues must be finite."
                );
            }
            minimum_eigenvalue = std::min(
                minimum_eigenvalue, eigenvalue
            );
            maximum_eigenvalue = std::max(
                maximum_eigenvalue, eigenvalue
            );
        }
    }
    for (int spin = 0; spin < result.nspin; ++spin) {
        if (spin_state_counts[spin] != result.kpoint_count) {
            throw std::runtime_error(
                "Each DOS spin channel must contain every k point once."
            );
        }
        if (std::abs(spin_weight_sums[spin] - 1.0) >
            weight_tolerance) {
            throw std::runtime_error(
                "DOS k-point weights must sum to one in each spin channel."
            );
        }
    }

    const double automatic_minimum =
        minimum_eigenvalue
        - automatic_window_sigma_margin * result.gaussian_sigma_ha;
    const double automatic_maximum =
        maximum_eigenvalue
        + automatic_window_sigma_margin * result.gaussian_sigma_ha;
    result.energy_min_ha = dos_options.energy_min_auto
        ? automatic_minimum
        : result.fermi_energy_ha
            + dos_options.energy_min_ev / HARTREE_TO_EV;
    result.energy_max_ha = dos_options.energy_max_auto
        ? automatic_maximum
        : result.fermi_energy_ha
            + dos_options.energy_max_ev / HARTREE_TO_EV;
    if (!std::isfinite(result.energy_min_ha) ||
        !std::isfinite(result.energy_max_ha) ||
        result.energy_min_ha >= result.energy_max_ha) {
        throw std::runtime_error(
            "The resolved DOS energy window is empty or invalid."
        );
    }

    result.samples.resize(dos_options.points);
    const double spacing =
        (result.energy_max_ha - result.energy_min_ha)
        / static_cast<double>(dos_options.points - 1);
    const double spin_degeneracy = result.nspin == 1 ? 2.0 : 1.0;
    for (const KPointElectronicState& state : scf.kpoints) {
        result.expected_total_states +=
            spin_degeneracy * state.weight
            * static_cast<double>(result.nbands);
    }
    for (int point = 0; point < dos_options.points; ++point) {
        DensityOfStatesSample& sample = result.samples[point];
        sample.energy_ha =
            result.energy_min_ha + static_cast<double>(point) * spacing;
        sample.spin_per_ha.assign(result.nspin, 0.0);
        sample.integrated_spin.assign(result.nspin, 0.0);
        for (const KPointElectronicState& state : scf.kpoints) {
            const double state_weight =
                spin_degeneracy * state.weight;
            for (int band = 0; band < result.nbands; ++band) {
                const double difference =
                    sample.energy_ha - state.eigenvalues[band];
                sample.spin_per_ha[state.spin_channel] +=
                    state_weight * gaussian_density(
                        difference, result.gaussian_sigma_ha
                    );
                sample.integrated_spin[state.spin_channel] +=
                    state_weight * gaussian_cumulative(
                        difference, result.gaussian_sigma_ha
                    );
            }
        }
        for (int spin = 0; spin < result.nspin; ++spin) {
            sample.total_per_ha += sample.spin_per_ha[spin];
            sample.integrated_total +=
                sample.integrated_spin[spin];
        }
    }
    result.analytic_states_in_window =
        result.samples.back().integrated_total
        - result.samples.front().integrated_total;
    for (int point = 1; point < dos_options.points; ++point) {
        result.numerical_states_in_window +=
            0.5 * spacing
            * (result.samples[point - 1].total_per_ha
               + result.samples[point].total_per_ha);
    }
    result.numerical_minus_analytic_states =
        result.numerical_states_in_window
        - result.analytic_states_in_window;
    return result;
}

void write_density_of_states(
    const std::string& path,
    const DensityOfStatesResult& result) {

    if (path.empty()) {
        throw std::runtime_error("The DOS output path is empty.");
    }
    if ((result.nspin != 1 && result.nspin != 2) ||
        result.nbands <= 0 || result.kpoint_count <= 0 ||
        result.samples.size() < 2 ||
        !std::isfinite(result.fermi_energy_ha) ||
        !std::isfinite(result.gaussian_sigma_ha) ||
        result.gaussian_sigma_ha <= 0.0) {
        throw std::runtime_error(
            "Cannot write an incomplete density-of-states result."
        );
    }

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create density-of-states output: " + path
        );
    }
    output << "# PWDFT Gaussian-broadened density of states\n"
        << "# fermi_energy_ha = " << std::scientific
        << std::setprecision(16) << result.fermi_energy_ha << "\n"
        << "# fermi_energy_ev = "
        << result.fermi_energy_ha * HARTREE_TO_EV << "\n"
        << "# gaussian_sigma_ha = "
        << result.gaussian_sigma_ha << "\n"
        << "# gaussian_sigma_ev = "
        << result.gaussian_sigma_ha * HARTREE_TO_EV << "\n"
        << "# nspin = " << result.nspin << "\n"
        << "# nbands = " << result.nbands << "\n"
        << "# irreducible_kpoints = " << result.kpoint_count << "\n"
        << "# expected_total_states = "
        << result.expected_total_states << "\n"
        << "# analytic_states_in_energy_window = "
        << result.analytic_states_in_window << "\n"
        << "# numerical_states_in_energy_window = "
        << result.numerical_states_in_window << "\n"
        << "# numerical_minus_analytic_states = "
        << result.numerical_minus_analytic_states << "\n"
        << "# states_outside_energy_window = "
        << result.expected_total_states
            - result.analytic_states_in_window
        << "\n";
    if (result.nspin == 1) {
        output
            << "# columns: energy_ha energy_ev "
               "energy_minus_fermi_ev dos_total_states_per_ev "
               "integrated_total_states\n";
    } else {
        output
            << "# columns: energy_ha energy_ev "
               "energy_minus_fermi_ev dos_total_states_per_ev "
               "dos_up_states_per_ev dos_down_states_per_ev "
               "integrated_total_states integrated_up_states "
               "integrated_down_states\n";
    }

    for (const DensityOfStatesSample& sample : result.samples) {
        if (static_cast<int>(sample.spin_per_ha.size()) != result.nspin ||
            static_cast<int>(sample.integrated_spin.size()) !=
                result.nspin) {
            throw std::runtime_error(
                "A DOS sample has the wrong spin-channel count."
            );
        }
        output << std::scientific << std::setprecision(12)
            << std::setw(20) << sample.energy_ha << " "
            << std::setw(20) << sample.energy_ha * HARTREE_TO_EV
            << " " << std::setw(20)
            << (sample.energy_ha - result.fermi_energy_ha)
                * HARTREE_TO_EV
            << " " << std::setw(20)
            << sample.total_per_ha / HARTREE_TO_EV;
        if (result.nspin == 2) {
            output << " " << std::setw(20)
                << sample.spin_per_ha[0] / HARTREE_TO_EV
                << " " << std::setw(20)
                << sample.spin_per_ha[1] / HARTREE_TO_EV;
        }
        output << " " << std::setw(20) << sample.integrated_total;
        if (result.nspin == 2) {
            output << " " << std::setw(20)
                << sample.integrated_spin[0]
                << " " << std::setw(20)
                << sample.integrated_spin[1];
        }
        output << "\n";
    }
}
