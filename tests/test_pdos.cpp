#include "pdos.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

} // namespace

int main() {
    try {
        const RealHarmonicDescriptor px =
            describe_real_harmonic(1, 1);
        const RealHarmonicDescriptor dxy =
            describe_real_harmonic(2, 4);
        require(
            px.absolute_m == 1 &&
            px.branch == "cos" &&
            px.orbital_name == "px" &&
            dxy.absolute_m == 2 &&
            dxy.branch == "sin" &&
            dxy.orbital_name == "dxy",
            "QE real-harmonic descriptor mapping"
        );

        const UPFData upf = read_nc_upf(
            (std::filesystem::path(TEST_DATA_DIR)
             / "minimal_local_nc.upf").string()
        );
        const AtomicProjectionSpecies species =
            prepare_atomic_projection_species(upf);
        const Lattice lattice(
            Eigen::Vector3d(10.0, 0.0, 0.0),
            Eigen::Vector3d(0.0, 10.0, 0.0),
            Eigen::Vector3d(0.0, 0.0, 10.0)
        );
        PlaneWaveBasis3D plane_waves;
        plane_waves.generate(
            lattice, Eigen::Vector3d::Zero(), 2.0
        );
        const std::vector<UPFLocalIon> ions{
            {0, Eigen::Vector3d::Zero()}
        };
        const AtomicProjectionBasis basis =
            build_lowdin_atomic_projection_basis(
                lattice,
                plane_waves,
                {species},
                ions,
                1.0e-12
            );
        require(
            basis.orbitals.size() == 1 &&
            basis.orbitals[0].label == "1S" &&
            basis.orbitals[0].angular_momentum == 0,
            "Hydrogen atomic-projection metadata mismatch"
        );
        require_close(
            (basis.coefficients.adjoint()
             * basis.coefficients)(0, 0).real(),
            1.0,
            1.0e-12,
            "Löwdin orbital normalization"
        );
        const Eigen::MatrixXd weights =
            compute_lowdin_projection_weights(
                basis, basis.coefficients
            );
        require_close(
            weights(0, 0),
            1.0,
            1.0e-12,
            "Projection of an atomic orbital onto itself"
        );

        SCFOptions options;
        options.nspin = 1;
        options.nbands = 1;
        KPointSCFResult electronic;
        electronic.converged = true;
        electronic.occupations.mu = 0.0;
        KPointElectronicState state;
        state.spin_channel = 0;
        state.kpoint_index = 0;
        state.weight = 1.0;
        state.eigenvalues =
            Eigen::VectorXd::Zero(1);
        state.occupations = {2.0};
        electronic.kpoints.push_back(state);

        DensityOfStatesOptions dos_options;
        dos_options.smearing_ev = 0.5;
        dos_options.points = 401;
        dos_options.energy_min_auto = false;
        dos_options.energy_max_auto = false;
        dos_options.energy_min_ev = -5.0;
        dos_options.energy_max_ev = 5.0;
        const DensityOfStatesResult dos =
            compute_density_of_states(
                electronic, options, dos_options
            );

        AtomicProjectionResult projection;
        projection.orbitals = basis.orbitals;
        projection.minimum_overlap_eigenvalue =
            basis.minimum_overlap_eigenvalue;
        projection.maximum_overlap_eigenvalue =
            basis.maximum_overlap_eigenvalue;
        projection.maximum_orthonormality_error =
            basis.orthonormality_error;
        AtomicProjectionState projected_state;
        projected_state.spin_channel = 0;
        projected_state.kpoint_index = 0;
        projected_state.weights = weights;
        projection.states.push_back(projected_state);

        const ProjectedDensityOfStatesResult pdos =
            compute_projected_density_of_states(
                electronic, options, dos, projection
            );
        require_close(
            pdos.occupied_spilling,
            0.0,
            1.0e-12,
            "Complete one-orbital occupied spilling"
        );
        require_close(
            pdos.channels[0].integrated_spin[0].back(),
            2.0,
            1.0e-12,
            "Integrated one-orbital PDOS"
        );
        require_close(
            pdos.full_projected_state_weight,
            2.0,
            1.0e-12,
            "Full projected state weight"
        );
        require_close(
            pdos.analytic_projected_states_in_window,
            2.0,
            1.0e-12,
            "Analytic PDOS energy-window integral"
        );
        require_close(
            pdos.numerical_projected_states_in_window,
            pdos.analytic_projected_states_in_window,
            1.0e-11,
            "Numerical PDOS quadrature self-check"
        );

        const std::filesystem::path output_path =
            std::filesystem::temp_directory_path()
            / "pwdft-test-pdos.dat";
        write_projected_density_of_states(
            output_path.string(), pdos
        );
        std::ifstream output(output_path);
        std::ostringstream text;
        text << output.rdbuf();
        require(
            text.str().find("Löwdin projected density") !=
                std::string::npos &&
            text.str().find("occupied_spilling") !=
                std::string::npos &&
            text.str().find(
                "numerical_minus_analytic_projected_states"
            ) !=
                std::string::npos,
            "PDOS output header or integral diagnostics mismatch"
        );
        std::filesystem::remove(output_path);

        std::cout
            << "Löwdin projection and PDOS tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PDOS test failed: "
            << error.what() << "\n";
        return 1;
    }
}
