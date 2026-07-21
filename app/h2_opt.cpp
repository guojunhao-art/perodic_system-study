#include "core.hpp"
#include "ewald.hpp"
#include "forces.hpp"
#include "scf.hpp"
#include "upf_local_potential.hpp"
#include "upf_reader.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double bohr_to_angstrom = 0.529177210903;

struct H2Point {
    double bond_length_bohr = 0.0;
    double energy_hartree = 0.0;
    double bond_force_hartree_per_bohr = 0.0;
    double local_force_hartree_per_bohr = 0.0;
    double ion_force_hartree_per_bohr = 0.0;
    int scf_iterations = 0;
    long long hamiltonian_applications = 0;
    double scf_wall_time_seconds = 0.0;
    double density_residual = 0.0;
    SCFInitialGuess next_guess;
};

std::vector<UPFLocalIon> make_upf_ions(
    const Lattice& lattice,
    double bond_length_bohr) {

    const Eigen::Vector3d center =
        0.5 * (lattice.A.col(0) + lattice.A.col(1) + lattice.A.col(2));
    const Eigen::Vector3d half_bond(
        0.5 * bond_length_bohr,
        0.0,
        0.0
    );

    return {
        {0, lattice.A_inv * (center - half_bond)},
        {0, lattice.A_inv * (center + half_bond)}
    };
}

std::vector<Ion> make_ewald_ions(
    const std::vector<UPFLocalIon>& upf_ions,
    double charge,
    double gaussian_width_bohr) {

    std::vector<Ion> ions;
    ions.reserve(upf_ions.size());
    for (const UPFLocalIon& upf_ion : upf_ions) {
        Ion ion;
        ion.frac_position = upf_ion.frac_position;
        ion.Z = charge;
        ion.sigma = gaussian_width_bohr;
        ions.push_back(ion);
    }
    return ions;
}

H2Point evaluate_h2_point(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const UPFLocalSpecies& hydrogen,
    double bond_length_bohr,
    double ewald_width_bohr,
    const SCFOptions& options,
    const SCFInitialGuess& initial_guess) {

    FFTWorkspace fft(grid);
    const std::vector<UPFLocalSpecies> species{hydrogen};
    const std::vector<UPFLocalIon> upf_ions =
        make_upf_ions(lattice, bond_length_bohr);
    const std::vector<Ion> ewald_ions = make_ewald_ions(
        upf_ions,
        hydrogen.valence_charge,
        ewald_width_bohr
    );

    const std::vector<double> local_potential =
        build_upf_local_potential_real(
            lattice,
            fft,
            species,
            upf_ions
        );
    const EwaldEnergyComponents ewald =
        compute_point_ion_ion_ewald_energy(
            lattice,
            grid,
            ewald_ions,
            ewald_width_bohr
        );

    const SCFResult scf = run_scf(
        lattice,
        basis,
        fft,
        local_potential,
        ewald.total,
        {},
        options,
        initial_guess
    );
    if (!scf.converged) {
        throw std::runtime_error(
            "H2 SCF did not converge at R = "
            + std::to_string(bond_length_bohr) + " Bohr."
        );
    }

    const auto density_G = build_density_G(fft, scf.density);
    const auto local_forces = compute_upf_local_ionic_forces(
        lattice,
        grid,
        species,
        upf_ions,
        density_G
    );
    const auto ion_forces = compute_point_ion_ion_ewald_forces(
        lattice,
        grid,
        ewald_ions,
        ewald_width_bohr
    );

    const Eigen::Vector3d bond_direction = Eigen::Vector3d::UnitX();
    const double local_bond_force =
        0.5 * (local_forces[1] - local_forces[0])
            .dot(bond_direction);
    const double ion_bond_force =
        0.5 * (ion_forces[1] - ion_forces[0])
            .dot(bond_direction);

    H2Point result;
    result.bond_length_bohr = bond_length_bohr;
    result.energy_hartree = scf.variational_energy;
    result.local_force_hartree_per_bohr = local_bond_force;
    result.ion_force_hartree_per_bohr = ion_bond_force;
    result.bond_force_hartree_per_bohr =
        local_bond_force + ion_bond_force;
    result.scf_iterations = scf.iterations;
    result.hamiltonian_applications =
        scf.eigensolver_hamiltonian_applications;
    result.scf_wall_time_seconds = scf.wall_time_seconds;
    result.density_residual = scf.final_density_residual;
    result.next_guess.density = scf.density;
    result.next_guess.orbitals = scf.orbitals;
    return result;
}

void print_point(int step, const H2Point& point) {
    std::cout << std::setw(4) << step
              << "  " << std::fixed << std::setprecision(7)
              << std::setw(11) << point.bond_length_bohr
              << "  " << std::setw(11)
              << point.bond_length_bohr * bohr_to_angstrom
              << "  " << std::scientific << std::setprecision(12)
              << std::setw(20) << point.energy_hartree
              << "  " << std::setw(15)
              << point.bond_force_hartree_per_bohr
              << "  " << std::setw(4) << point.scf_iterations
              << "  " << std::setw(7) << point.hamiltonian_applications
              << "  " << std::fixed << std::setprecision(2)
              << std::setw(8) << point.scf_wall_time_seconds
              << "  " << std::scientific << std::setprecision(3)
              << std::setw(11) << point.density_residual
              << "\n";
}

double safeguarded_secant(
    double lower,
    double upper,
    double lower_force,
    double upper_force) {

    const double denominator = upper_force - lower_force;
    if (std::abs(denominator) < 1.0e-14) {
        return 0.5 * (lower + upper);
    }

    const double secant =
        (lower * upper_force - upper * lower_force) / denominator;
    const double margin = 0.15 * (upper - lower);
    if (secant <= lower + margin || secant >= upper - margin) {
        return 0.5 * (lower + upper);
    }
    return secant;
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("H2 plane-wave basis is empty.");
    }

    Eigen::Vector3i minimum_frequency = basis.gvectors.front().n;
    Eigen::Vector3i maximum_frequency = basis.gvectors.front().n;
    for (const GVector& gvector : basis.gvectors) {
        minimum_frequency = minimum_frequency.cwiseMin(gvector.n);
        maximum_frequency = maximum_frequency.cwiseMax(gvector.n);
    }

    const Eigen::Vector3i grid_sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_product_frequency =
            maximum_frequency[direction] - minimum_frequency[direction];
        const int largest_unaliased_frequency =
            (grid_sizes[direction] - 1) / 2;
        if (largest_product_frequency > largest_unaliased_frequency) {
            throw std::runtime_error(
                "FFT grid is too small for products of cutoff "
                "wavefunctions in direction "
                + std::to_string(direction) + "."
            );
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr
            << "Usage: h2_opt H.UPF [ECUT_HA] [CELL_BOHR] [FFT_N]\n";
        return 2;
    }

    try {
        const double ecut_hartree =
            argc >= 3 ? std::stod(argv[2]) : 6.0;
        const double cell_length_bohr =
            argc >= 4 ? std::stod(argv[3]) : 12.0;
        const int fft_size =
            argc >= 5 ? std::stoi(argv[4]) : 32;
        const double screening_width_bohr = 0.70;

        if (!std::isfinite(ecut_hartree) || ecut_hartree <= 0.0 ||
            !std::isfinite(cell_length_bohr) || cell_length_bohr <= 2.2 ||
            fft_size < 4) {
            throw std::runtime_error(
                "H2 cutoff, cell length, or FFT size is invalid."
            );
        }
        const UPFData upf = read_nc_upf(argv[1]);
        if (upf.header.element != "H") {
            throw std::runtime_error(
                "h2_opt requires a hydrogen UPF (element = H)."
            );
        }
        if (upf.header.number_of_projectors != 0 ||
            !upf.projectors.empty()) {
            throw std::runtime_error(
                "h2_opt currently accepts only a local, projector-free H UPF."
            );
        }
        if (std::abs(upf.header.z_valence - 1.0) > 1.0e-12) {
            throw std::runtime_error(
                "h2_opt requires a hydrogen UPF with z_valence = 1."
            );
        }

        const Lattice lattice(
            Eigen::Vector3d(cell_length_bohr, 0.0, 0.0),
            Eigen::Vector3d(0.0, cell_length_bohr, 0.0),
            Eigen::Vector3d(0.0, 0.0, cell_length_bohr)
        );
        const FFTGrid grid(fft_size, fft_size, fft_size);
        PlaneWaveBasis3D basis;
        basis.generate(
            lattice,
            Eigen::Vector3d::Zero(),
            ecut_hartree
        );
        require_fft_grid_for_basis_products(basis, grid);
        const UPFLocalSpecies hydrogen =
            prepare_upf_local_species(upf, screening_width_bohr);

        SCFOptions options;
        options.nelec = 2.0;
        options.nbands = 4;
        options.occupation_mode = OccupationMode::Fixed;
        options.fixed_occupations.assign(options.nbands, 0.0);
        options.fixed_occupations[0] = 2.0;
        options.max_iterations = 240;
        options.density_tolerance = 2.0e-8;
        options.energy_tolerance = 2.0e-10;
        options.eigensolver_max_iterations = 100;
        options.eigensolver_max_subspace =
            std::min(basis.size(), 4 * options.nbands + 16);
        options.eigensolver_tolerance = 2.0e-10;
        options.mixing_alpha = 0.12;
        options.pulay_max_history = 7;
        options.pulay_min_history = 2;
        options.pulay_regularization = 1.0e-12;
        options.verbosity = SCFVerbosity::Silent;

        std::cout
            << "H2 local-NC-UPF bond optimization\n"
            << "---------------------------------\n"
            << "UPF       : " << upf.source_path << "\n"
            << "XC        : " << lda_functional_name(options.lda_functional)
            << "\n"
            << "cell      : " << cell_length_bohr << " Bohr\n"
            << "ecut      : " << ecut_hartree << " Ha\n"
            << "FFT       : " << fft_size << "^3\n"
            << "plane waves: " << basis.size() << "\n\n"
            << "step      R/Bohr        R/Angstrom"
            << "             E/Ha        F_R/HaBohr  NELM  N_Hpsi"
            << "    time/s         drho\n";

        int step = 0;
        H2Point lower = evaluate_h2_point(
            lattice, grid, basis, hydrogen, 1.05,
            screening_width_bohr, options, {}
        );
        print_point(step++, lower);
        H2Point upper = evaluate_h2_point(
            lattice, grid, basis, hydrogen, 2.10,
            screening_width_bohr, options, lower.next_guess
        );
        print_point(step++, upper);

        if (!(lower.bond_force_hartree_per_bohr > 0.0 &&
              upper.bond_force_hartree_per_bohr < 0.0)) {
            throw std::runtime_error(
                "Initial H2 interval does not bracket a force zero."
            );
        }

        H2Point best = std::abs(lower.bond_force_hartree_per_bohr)
                < std::abs(upper.bond_force_hartree_per_bohr)
            ? lower
            : upper;
        for (int iteration = 0; iteration < 18; ++iteration) {
            const double trial_bond = safeguarded_secant(
                lower.bond_length_bohr,
                upper.bond_length_bohr,
                lower.bond_force_hartree_per_bohr,
                upper.bond_force_hartree_per_bohr
            );
            const SCFInitialGuess& guess =
                std::abs(trial_bond - lower.bond_length_bohr)
                    < std::abs(trial_bond - upper.bond_length_bohr)
                ? lower.next_guess
                : upper.next_guess;
            H2Point trial = evaluate_h2_point(
                lattice, grid, basis, hydrogen, trial_bond,
                screening_width_bohr, options, guess
            );
            print_point(step++, trial);

            if (std::abs(trial.bond_force_hartree_per_bohr)
                < std::abs(best.bond_force_hartree_per_bohr)) {
                best = trial;
            }
            if (trial.bond_force_hartree_per_bohr > 0.0) {
                lower = trial;
            } else {
                upper = trial;
            }

            if (std::abs(best.bond_force_hartree_per_bohr) < 2.0e-5 ||
                upper.bond_length_bohr - lower.bond_length_bohr < 2.0e-4) {
                break;
            }
        }

        if (std::abs(best.bond_force_hartree_per_bohr) >= 2.0e-5) {
            throw std::runtime_error(
                "H2 bond optimization did not converge the bond force."
            );
        }

        const double finite_difference_step = 1.0e-3;
        const H2Point plus = evaluate_h2_point(
            lattice, grid, basis, hydrogen,
            best.bond_length_bohr + finite_difference_step,
            screening_width_bohr, options, best.next_guess
        );
        const H2Point minus = evaluate_h2_point(
            lattice, grid, basis, hydrogen,
            best.bond_length_bohr - finite_difference_step,
            screening_width_bohr, options, best.next_guess
        );
        const double finite_difference_force =
            -(plus.energy_hartree - minus.energy_hartree)
            / (2.0 * finite_difference_step);

        std::cout << "\nOptimized H-H bond\n"
                  << "  R = " << std::fixed << std::setprecision(8)
                  << best.bond_length_bohr << " Bohr = "
                  << best.bond_length_bohr * bohr_to_angstrom
                  << " Angstrom\n"
                  << std::scientific << std::setprecision(10)
                  << "  analytic bond force = "
                  << best.bond_force_hartree_per_bohr << " Ha/Bohr\n"
                  << "  finite-difference force = "
                  << finite_difference_force << " Ha/Bohr\n"
                  << "  force difference = "
                  << std::abs(
                      best.bond_force_hartree_per_bohr
                      - finite_difference_force
                     ) << " Ha/Bohr\n"
                  << "  force split (local / ion-ion) = "
                  << best.local_force_hartree_per_bohr << " / "
                  << best.ion_force_hartree_per_bohr << " Ha/Bohr\n";

        if (std::abs(
                best.bond_force_hartree_per_bohr
                - finite_difference_force
            ) > 2.0e-4) {
            throw std::runtime_error(
                "H2 analytic bond force failed the SCF finite-difference check."
            );
        }
    } catch (const std::exception& error) {
        std::cerr << "H2 optimization failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
