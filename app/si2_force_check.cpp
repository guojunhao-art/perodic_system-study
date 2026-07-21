#include "core.hpp"
#include "ewald.hpp"
#include "forces.hpp"
#include "potentials.hpp"
#include "scf.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double hartree_to_eV = 27.211386245988;

struct ElectronicReference {
    SCFResult scf;
    std::vector<std::complex<double>> density_G;
    std::vector<NonlocalProjector> projectors;
    IonicForceComponents forces;
};

struct DisplacedEnergies {
    double local = 0.0;
    double ion_ion = 0.0;
    double nonlocal = 0.0;
    double total_free = 0.0;
    int scf_iterations = 0;
    double final_density_residual = 0.0;
};

struct MaximumForceError {
    double value = -1.0;
    int ion = -1;
    int direction = -1;
    double analytic = 0.0;
    double finite_difference = 0.0;

    void consider(
        int ion_index,
        int cartesian_direction,
        double analytic_value,
        double finite_difference_value) {

        if (!std::isfinite(analytic_value) ||
            !std::isfinite(finite_difference_value)) {
            throw std::runtime_error(
                "Force comparison contains a non-finite value."
            );
        }
        const double error =
            std::abs(analytic_value - finite_difference_value);
        if (error > value) {
            value = error;
            ion = ion_index;
            direction = cartesian_direction;
            analytic = analytic_value;
            finite_difference = finite_difference_value;
        }
    }
};

bool is_perdew_zunger_lda(const std::string& functional) {
    std::string uppercase = functional;
    std::transform(
        uppercase.begin(), uppercase.end(), uppercase.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        }
    );
    return uppercase.find("PZ") != std::string::npos;
}

void wrap_fractional(Eigen::Vector3d& position) {
    for (int direction = 0; direction < 3; ++direction) {
        position[direction] -= std::floor(position[direction]);
    }
}

std::vector<UPFLocalIon> displace_ion(
    const Lattice& lattice,
    const std::vector<UPFLocalIon>& ions,
    int ion_index,
    int direction,
    double displacement_bohr) {

    std::vector<UPFLocalIon> moved = ions;
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    displacement[direction] = displacement_bohr;
    moved[ion_index].frac_position += lattice.A_inv * displacement;
    wrap_fractional(moved[ion_index].frac_position);
    return moved;
}

std::vector<Ion> make_ewald_ions(
    const std::vector<UPFLocalIon>& upf_ions,
    double charge,
    double width_bohr) {

    std::vector<Ion> ions;
    ions.reserve(upf_ions.size());
    for (const UPFLocalIon& upf_ion : upf_ions) {
        Ion ion;
        ion.frac_position = upf_ion.frac_position;
        ion.Z = charge;
        ion.sigma = width_bohr;
        ions.push_back(ion);
    }
    return ions;
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("The Si2 plane-wave basis is empty.");
    }
    Eigen::Vector3i minimum = basis.gvectors.front().n;
    Eigen::Vector3i maximum = basis.gvectors.front().n;
    for (const GVector& gvector : basis.gvectors) {
        minimum = minimum.cwiseMin(gvector.n);
        maximum = maximum.cwiseMax(gvector.n);
    }
    const Eigen::Vector3i sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_product_frequency =
            maximum[direction] - minimum[direction];
        if (largest_product_frequency > (sizes[direction] - 1) / 2) {
            throw std::runtime_error(
                "FFT_N is too small for products of cutoff wavefunctions."
            );
        }
    }
}

IonicForceComponents assemble_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFLocalSpecies>& local_species,
    const std::vector<UPFLocalIon>& upf_ions,
    const std::vector<Ion>& ewald_ions,
    double ewald_width_bohr,
    const std::vector<std::complex<double>>& density_G,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& orbitals,
    const std::vector<double>& occupations) {

    IonicForceComponents result;
    result.local = compute_upf_local_ionic_forces(
        lattice, grid, local_species, upf_ions, density_G
    );
    result.ion_ion = compute_point_ion_ion_ewald_forces(
        lattice, grid, ewald_ions, ewald_width_bohr
    );
    result.nonlocal = compute_nonlocal_ionic_forces(
        basis, projectors, orbitals, occupations,
        static_cast<int>(upf_ions.size())
    );
    result.total.resize(upf_ions.size(), Eigen::Vector3d::Zero());
    for (int ion = 0; ion < static_cast<int>(upf_ions.size()); ++ion) {
        result.total[ion] =
            result.local[ion] + result.ion_ion[ion] + result.nonlocal[ion];
    }
    return result;
}

double fixed_density_local_energy(
    const std::vector<double>& density,
    const std::vector<double>& local_potential,
    double dV) {

    if (density.size() != local_potential.size()) {
        throw std::runtime_error("Density/local-potential size mismatch.");
    }
    double energy = 0.0;
    for (int index = 0; index < static_cast<int>(density.size()); ++index) {
        energy += density[index] * local_potential[index];
    }
    return energy * dV;
}

ElectronicReference converge_reference(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFLocalSpecies>& local_species,
    const std::vector<UPFNonlocalSpecies>& nonlocal_species,
    const std::vector<UPFLocalIon>& ions,
    double ewald_width_bohr,
    const SCFOptions& options) {

    FFTWorkspace fft(grid);
    const auto ewald_ions = make_ewald_ions(
        ions, local_species[0].valence_charge, ewald_width_bohr
    );
    const auto local_potential = build_upf_local_potential_real(
        lattice, fft, local_species, ions
    );
    const auto projectors = build_upf_nonlocal_projectors(
        lattice, basis, nonlocal_species, ions
    );
    const auto ewald = compute_point_ion_ion_ewald_energy(
        lattice, grid, ewald_ions, ewald_width_bohr
    );
    SCFResult scf = run_scf(
        lattice, basis, fft, local_potential, ewald.total,
        projectors, options
    );
    if (!scf.converged) {
        throw std::runtime_error("The central Si2 SCF did not converge.");
    }

    ElectronicReference result;
    result.density_G = build_density_G(fft, scf.density);
    result.forces = assemble_forces(
        lattice, grid, basis, local_species, ions, ewald_ions,
        ewald_width_bohr, result.density_G, projectors,
        scf.orbitals, scf.occupations.occ
    );
    result.projectors = projectors;
    result.scf = std::move(scf);
    return result;
}

DisplacedEnergies evaluate_displacement(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFLocalSpecies>& local_species,
    const std::vector<UPFNonlocalSpecies>& nonlocal_species,
    const std::vector<UPFLocalIon>& ions,
    double ewald_width_bohr,
    const SCFOptions& options,
    const ElectronicReference& reference,
    bool run_self_consistent_calculation) {

    FFTWorkspace fft(grid);
    const auto ewald_ions = make_ewald_ions(
        ions, local_species[0].valence_charge, ewald_width_bohr
    );
    const auto local_potential = build_upf_local_potential_real(
        lattice, fft, local_species, ions
    );
    const auto projectors = build_upf_nonlocal_projectors(
        lattice, basis, nonlocal_species, ions
    );
    const auto ewald = compute_point_ion_ion_ewald_energy(
        lattice, grid, ewald_ions, ewald_width_bohr
    );

    DisplacedEnergies result;
    result.local = fixed_density_local_energy(
        reference.scf.density, local_potential,
        lattice.volume() / static_cast<double>(grid.ngrid)
    );
    result.ion_ion = ewald.total;
    result.nonlocal = compute_nonlocal_energy(
        projectors, reference.scf.orbitals,
        reference.scf.occupations.occ
    );

    if (run_self_consistent_calculation) {
        SCFInitialGuess guess;
        guess.density = reference.scf.density;
        guess.orbitals = reference.scf.orbitals;
        const SCFResult displaced = run_scf(
            lattice, basis, fft, local_potential, ewald.total,
            projectors, options, guess
        );
        if (!displaced.converged) {
            throw std::runtime_error("A displaced Si2 SCF did not converge.");
        }
        result.total_free = displaced.variational_energy;
        result.scf_iterations = displaced.iterations;
        result.final_density_residual = displaced.final_density_residual;
    }
    return result;
}

double max_abs_force_sum(const std::vector<Eigen::Vector3d>& forces) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& force : forces) {
        sum += force;
    }
    return sum.cwiseAbs().maxCoeff();
}

MaximumForceError compare_forces(
    const std::vector<Eigen::Vector3d>& analytic,
    const std::vector<Eigen::Vector3d>& finite_difference) {

    if (analytic.size() != finite_difference.size()) {
        throw std::runtime_error("Force comparison size mismatch.");
    }
    if (analytic.empty()) {
        throw std::runtime_error("Cannot compare empty force arrays.");
    }

    MaximumForceError maximum;
    for (int ion = 0; ion < static_cast<int>(analytic.size()); ++ion) {
        for (int direction = 0; direction < 3; ++direction) {
            maximum.consider(
                ion,
                direction,
                analytic[ion][direction],
                finite_difference[ion][direction]
            );
        }
    }
    return maximum;
}

void print_force_comparison(
    const std::string& label,
    const MaximumForceError& error,
    double tolerance = 0.0) {

    static const std::array<const char*, 3> directions{"x", "y", "z"};
    std::cout << std::left << std::setw(18) << label << std::right
              << " max |analytic-FD| = " << std::scientific
              << std::setprecision(6) << error.value;
    if (tolerance > 0.0) {
        std::cout << "  tolerance = " << tolerance;
    }
    std::cout << "  worst ion/direction = " << error.ion << "/"
              << directions.at(error.direction)
              << "  analytic/FD = " << error.analytic << "/"
              << error.finite_difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 7) {
        std::cerr
            << "Usage: si2_force_check Si.UPF [ECUT_HA] [CELL_BOHR] "
               "[FFT_N] [SMEARING_EV] [MIN_SCF_FD_STEP_BOHR]\n";
        return 2;
    }

    try {
        const double ecut = argc >= 3 ? std::stod(argv[2]) : 10.0;
        const double cell = argc >= 4 ? std::stod(argv[3]) : 16.0;
        const int fft_size = argc >= 5 ? std::stoi(argv[4]) : 48;
        const double smearing_eV = argc >= 6 ? std::stod(argv[5]) : 0.05;
        const double finest_scf_step =
            argc >= 7 ? std::stod(argv[6]) : 2.0e-3;
        const double ewald_width = 0.70;
        if (!std::isfinite(ecut) || !(ecut > 0.0) ||
            !std::isfinite(cell) || !(cell > 8.0) || fft_size < 4 ||
            !std::isfinite(smearing_eV) || smearing_eV < 0.0 ||
            !std::isfinite(finest_scf_step) || !(finest_scf_step > 0.0)) {
            throw std::runtime_error("Invalid Si2 numerical parameter.");
        }

        const UPFData upf = read_nc_upf(argv[1]);
        if (upf.header.element != "Si") {
            throw std::runtime_error("si2_force_check requires a Si UPF.");
        }
        if (!is_perdew_zunger_lda(upf.header.functional)) {
            throw std::runtime_error("The Si UPF must use PZ-LDA.");
        }
        if (upf.projectors.empty()) {
            throw std::runtime_error(
                "The Si force check requires nonlocal UPF projectors."
            );
        }

        const Lattice lattice(
            Eigen::Vector3d(cell, 0.0, 0.0),
            Eigen::Vector3d(0.0, cell, 0.0),
            Eigen::Vector3d(0.0, 0.0, cell)
        );
        const FFTGrid grid(fft_size, fft_size, fft_size);
        PlaneWaveBasis3D basis;
        basis.generate(lattice, Eigen::Vector3d::Zero(), ecut);
        require_fft_grid_for_basis_products(basis, grid);

        const std::vector<UPFLocalSpecies> local_species{
            prepare_upf_local_species(upf, ewald_width)
        };
        const std::vector<UPFNonlocalSpecies> nonlocal_species{
            prepare_upf_nonlocal_species(upf)
        };
        const std::vector<UPFLocalIon> ions{
            {0, Eigen::Vector3d(0.310, 0.370, 0.410)},
            {0, Eigen::Vector3d(0.530, 0.480, 0.550)}
        };

        SCFOptions options;
        options.nelec = 2.0 * upf.header.z_valence;
        options.nbands = std::max(
            12,
            static_cast<int>(std::ceil(0.5 * options.nelec)) + 4
        );
        if (basis.size() < options.nbands) {
            throw std::runtime_error(
                "The plane-wave basis has fewer vectors than Si2 bands."
            );
        }
        options.occupation_mode = smearing_eV > 0.0
            ? OccupationMode::FermiDirac
            : OccupationMode::DegeneracyAwareZeroT;
        options.fixed_occupations.assign(options.nbands, 0.0);
        options.smearing_sigma = smearing_eV / hartree_to_eV;
        options.max_iterations = 300;
        options.density_tolerance = 1.0e-8;
        options.energy_tolerance = 1.0e-10;
        options.eigensolver_max_iterations = 120;
        options.eigensolver_max_subspace =
            std::min(basis.size(), 4 * options.nbands + 12);
        options.eigensolver_tolerance = 2.0e-10;
        options.mixing_alpha = 0.08;
        options.pulay_max_history = 8;
        options.pulay_min_history = 2;
        options.pulay_regularization = 1.0e-12;
        options.verbosity = SCFVerbosity::Silent;

        const Eigen::Vector3d bond = lattice.cart_from_frac(
            ions[1].frac_position - ions[0].frac_position
        );
        std::cout << std::setprecision(12)
                  << "Si2 real-UPF force validation\n"
                  << "  N_PW / NELECT / NBANDS = " << basis.size()
                  << " / " << options.nelec << " / " << options.nbands
                  << "\n  cell / cutoff / FFT = " << cell << " Bohr / "
                  << ecut << " Ha / " << fft_size << "^3\n"
                  << "  bond vector / length = " << bond.transpose()
                  << " / " << bond.norm() << " Bohr\n"
                  << "  smearing = " << smearing_eV << " eV"
                  << "  SCF drho/dE = " << options.density_tolerance
                  << "/" << options.energy_tolerance << " Ha\n";
        const ElectronicReference reference = converge_reference(
            lattice, grid, basis, local_species, nonlocal_species,
            ions, ewald_width, options
        );

        std::cout << "central free energy = "
                  << reference.scf.variational_energy
                  << " Ha  SCF iterations = " << reference.scf.iterations
                  << "  drho = " << reference.scf.final_density_residual
                  << "\n\n";
        std::cout << "Analytic forces (Ha/Bohr)\n";
        for (int ion = 0; ion < 2; ++ion) {
            std::cout << "ion " << ion
                      << "  F_loc " << reference.forces.local[ion].transpose()
                      << "  F_II " << reference.forces.ion_ion[ion].transpose()
                      << "  F_NL " << reference.forces.nonlocal[ion].transpose()
                      << "  F_total " << reference.forces.total[ion].transpose()
                      << "\n";
        }
        std::cout << "max |sum_I F_I| = "
                  << max_abs_force_sum(reference.forces.total) << "\n\n";

        constexpr double frozen_step = 1.0e-5;
        std::vector<Eigen::Vector3d> fd_local(
            ions.size(), Eigen::Vector3d::Zero()
        );
        std::vector<Eigen::Vector3d> fd_ion_ion(
            ions.size(), Eigen::Vector3d::Zero()
        );
        std::vector<Eigen::Vector3d> fd_nonlocal(
            ions.size(), Eigen::Vector3d::Zero()
        );

        std::cout << "Frozen-state component checks, h = "
                  << frozen_step << " Bohr\n";
        for (int ion = 0; ion < static_cast<int>(ions.size()); ++ion) {
            for (int direction = 0; direction < 3; ++direction) {
                const auto plus_ions = displace_ion(
                    lattice, ions, ion, direction, frozen_step
                );
                const auto minus_ions = displace_ion(
                    lattice, ions, ion, direction, -frozen_step
                );
                const DisplacedEnergies plus = evaluate_displacement(
                    lattice, grid, basis, local_species, nonlocal_species,
                    plus_ions, ewald_width, options, reference, false
                );
                const DisplacedEnergies minus = evaluate_displacement(
                    lattice, grid, basis, local_species, nonlocal_species,
                    minus_ions, ewald_width, options, reference, false
                );
                const double denominator = 2.0 * frozen_step;
                fd_local[ion][direction] =
                    -(plus.local - minus.local) / denominator;
                fd_ion_ion[ion][direction] =
                    -(plus.ion_ion - minus.ion_ion) / denominator;
                fd_nonlocal[ion][direction] =
                    -(plus.nonlocal - minus.nonlocal) / denominator;
            }
        }

        const MaximumForceError local_error = compare_forces(
            reference.forces.local, fd_local
        );
        const MaximumForceError ion_ion_error = compare_forces(
            reference.forces.ion_ion, fd_ion_ion
        );
        const MaximumForceError nonlocal_error = compare_forces(
            reference.forces.nonlocal, fd_nonlocal
        );
        print_force_comparison("local", local_error, 1.0e-7);
        print_force_comparison("ion-ion", ion_ion_error, 1.0e-8);
        print_force_comparison("nonlocal", nonlocal_error, 1.0e-7);

        const std::array<double, 3> scf_steps{
            5.0 * finest_scf_step,
            2.5 * finest_scf_step,
            finest_scf_step
        };
        MaximumForceError finest_total_error;
        std::cout
            << "\nFully self-consistent finite differences of Mermin TOTEN\n"
            << "  h/Bohr  ion dir  NELM(+/-)       drho(+/-)"
            << "         analytic              FD         |error|\n";
        static const std::array<const char*, 3> directions{"x", "y", "z"};

        for (std::size_t step_index = 0;
             step_index < scf_steps.size();
             ++step_index) {
            const double step = scf_steps[step_index];
            std::vector<Eigen::Vector3d> fd_total(
                ions.size(), Eigen::Vector3d::Zero()
            );
            for (int ion = 0; ion < static_cast<int>(ions.size()); ++ion) {
                for (int direction = 0; direction < 3; ++direction) {
                    const auto plus_ions = displace_ion(
                        lattice, ions, ion, direction, step
                    );
                    const auto minus_ions = displace_ion(
                        lattice, ions, ion, direction, -step
                    );
                    const DisplacedEnergies plus = evaluate_displacement(
                        lattice, grid, basis, local_species, nonlocal_species,
                        plus_ions, ewald_width, options, reference, true
                    );
                    const DisplacedEnergies minus = evaluate_displacement(
                        lattice, grid, basis, local_species, nonlocal_species,
                        minus_ions, ewald_width, options, reference, true
                    );
                    fd_total[ion][direction] =
                        -(plus.total_free - minus.total_free) / (2.0 * step);
                    const double analytic =
                        reference.forces.total[ion][direction];
                    std::cout << std::scientific << std::setprecision(3)
                              << "  " << step << "  " << ion << "  "
                              << directions[direction] << "     "
                              << plus.scf_iterations << "/"
                              << minus.scf_iterations << "     "
                              << plus.final_density_residual << "/"
                              << minus.final_density_residual << "  "
                              << std::setprecision(7) << analytic << "  "
                              << fd_total[ion][direction] << "  "
                              << std::abs(
                                  analytic - fd_total[ion][direction]
                              ) << "\n";
                }
            }

            const MaximumForceError total_error = compare_forces(
                reference.forces.total, fd_total
            );
            print_force_comparison(
                "SCF h=" + std::to_string(step),
                total_error,
                step_index + 1 == scf_steps.size() ? 2.0e-4 : 0.0
            );
            if (step_index + 1 == scf_steps.size()) {
                finest_total_error = total_error;
            }
        }

        const double acoustic_sum_error =
            max_abs_force_sum(reference.forces.total);
        const bool passed =
            local_error.value < 1.0e-7 &&
            ion_ion_error.value < 1.0e-8 &&
            nonlocal_error.value < 1.0e-7 &&
            finest_total_error.value < 2.0e-4 &&
            acoustic_sum_error < 1.0e-6;

        std::cout << "\nValidation summary\n"
                  << "  frozen local / ion-ion / nonlocal = "
                  << local_error.value << " / " << ion_ion_error.value
                  << " / " << nonlocal_error.value << " Ha/Bohr\n"
                  << "  finest-step total-SCF error       = "
                  << finest_total_error.value << " Ha/Bohr\n"
                  << "  max |sum_I F_I|                   = "
                  << acoustic_sum_error << " Ha/Bohr\n"
                  << "  result                            = "
                  << (passed ? "PASS" : "FAIL") << "\n";

        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Si2 force check failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
