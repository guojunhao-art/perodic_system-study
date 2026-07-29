#include "dos.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double inverse_sqrt_two_pi =
    0.398942280401432677939946059934381868;

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

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

KPointElectronicState make_state(
    int spin,
    int kpoint,
    double weight,
    double eigenvalue_ha) {

    KPointElectronicState state;
    state.spin_channel = spin;
    state.kpoint_index = kpoint;
    state.weight = weight;
    state.eigenvalues = Eigen::VectorXd::Constant(1, eigenvalue_ha);
    return state;
}

DensityOfStatesOptions symmetric_options() {
    DensityOfStatesOptions options;
    options.smearing_ev = 1.0;
    options.points = 21;
    options.energy_min_auto = false;
    options.energy_max_auto = false;
    options.energy_min_ev = -10.0;
    options.energy_max_ev = 10.0;
    return options;
}

} // namespace

int main() {
    try {
        SCFOptions nonmagnetic_options;
        nonmagnetic_options.nspin = 1;
        nonmagnetic_options.nbands = 1;

        KPointSCFResult nonmagnetic_scf;
        nonmagnetic_scf.converged = true;
        nonmagnetic_scf.occupations.mu = 0.0;
        nonmagnetic_scf.kpoints.push_back(
            make_state(0, 0, 0.25, 0.0)
        );
        nonmagnetic_scf.kpoints.push_back(
            make_state(0, 1, 0.75, 0.0)
        );

        const DensityOfStatesResult nonmagnetic =
            compute_density_of_states(
                nonmagnetic_scf,
                nonmagnetic_options,
                symmetric_options()
            );
        require_true(
            nonmagnetic.samples.size() == 21 &&
            nonmagnetic.kpoint_count == 2,
            "Nonmagnetic DOS grid or k-point count mismatch"
        );
        const double sigma_ha = 1.0 / HARTREE_TO_EV;
        require_close(
            nonmagnetic.samples[10].energy_ha,
            0.0,
            1.0e-15,
            "The explicit DOS grid should contain the Fermi energy"
        );
        require_close(
            nonmagnetic.samples[10].total_per_ha,
            2.0 * inverse_sqrt_two_pi / sigma_ha,
            1.0e-11,
            "Nonmagnetic spin degeneracy in DOS"
        );
        require_close(
            nonmagnetic.samples[10].integrated_total,
            1.0,
            1.0e-14,
            "Integrated nonmagnetic DOS at a centered eigenvalue"
        );
        require_close(
            nonmagnetic.samples.back().integrated_total,
            2.0,
            1.0e-12,
            "Integrated nonmagnetic DOS state count"
        );

        SCFOptions spin_options = nonmagnetic_options;
        spin_options.nspin = 2;
        KPointSCFResult spin_scf;
        spin_scf.converged = true;
        spin_scf.occupations.mu = 0.0;
        spin_scf.kpoints.push_back(
            make_state(0, 0, 1.0, -sigma_ha)
        );
        spin_scf.kpoints.push_back(
            make_state(1, 0, 1.0, sigma_ha)
        );
        const DensityOfStatesResult spin_dos =
            compute_density_of_states(
                spin_scf,
                spin_options,
                symmetric_options()
            );
        require_true(
            spin_dos.samples[10].spin_per_ha.size() == 2,
            "Spin-resolved DOS channel count mismatch"
        );
        require_close(
            spin_dos.samples[10].spin_per_ha[0],
            spin_dos.samples[10].spin_per_ha[1],
            1.0e-13,
            "Symmetric spin channels should have equal DOS at the midpoint"
        );
        require_close(
            spin_dos.samples[10].integrated_total,
            1.0,
            1.0e-14,
            "Spin-resolved integrated DOS at the midpoint"
        );

        DensityOfStatesOptions automatic;
        automatic.smearing_ev = 0.5;
        automatic.points = 11;
        const DensityOfStatesResult automatic_dos =
            compute_density_of_states(
                spin_scf,
                spin_options,
                automatic
            );
        const double automatic_sigma_ha =
            automatic.smearing_ev / HARTREE_TO_EV;
        require_close(
            automatic_dos.energy_min_ha,
            -sigma_ha - 5.0 * automatic_sigma_ha,
            1.0e-14,
            "Automatic DOS lower limit"
        );
        require_close(
            automatic_dos.energy_max_ha,
            sigma_ha + 5.0 * automatic_sigma_ha,
            1.0e-14,
            "Automatic DOS upper limit"
        );

        const std::filesystem::path output_path =
            std::filesystem::temp_directory_path()
            / "pwdft-test-dos.dat";
        write_density_of_states(output_path.string(), spin_dos);
        std::ifstream output(output_path);
        std::ostringstream text;
        text << output.rdbuf();
        require_true(
            text.str().find("dos_up_states_per_ev") !=
                std::string::npos &&
            text.str().find("gaussian_sigma_ev") !=
                std::string::npos,
            "Spin-resolved DOS output header mismatch"
        );
        std::filesystem::remove(output_path);

        std::cout << "Density-of-states tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Density-of-states test failed: "
            << error.what() << "\n";
        return 1;
    }
}
