#include "eigensolver.hpp"
#include "forces.hpp"
#include "mixing.hpp"
#include "potentials.hpp"
#include "scf_modules.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct SelfConsistentPoint {
    double energy = 0.0;
    Eigen::Vector3d force_on_ion0 = Eigen::Vector3d::Zero();
    int scf_iterations = 0;
    double final_drho = 0.0;
    double final_dE = 0.0;
};

void wrap_fractional(Eigen::Vector3d& s) {
    for (int a = 0; a < 3; ++a) {
        s[a] -= std::floor(s[a]);
    }
}

std::vector<Ion> displace_ion_cartesian(
    const Lattice& lattice,
    const std::vector<Ion>& ions,
    int iion,
    int idir,
    double displacement) {

    std::vector<Ion> moved = ions;
    Eigen::Vector3d dr = Eigen::Vector3d::Zero();
    dr[idir] = displacement;
    moved[iion].frac_position += lattice.A_inv * dr;
    wrap_fractional(moved[iion].frac_position);
    return moved;
}

SelfConsistentPoint run_scf_from_scratch(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<Ion>& ions) {

    FFTWorkspace fft(grid);

    const double nelec = 2.0;
    const int nbands = 4;
    const double volume = lattice.volume();
    const double dV = volume / static_cast<double>(grid.ngrid);

    const LocalPotentialComponents local_potential =
        build_local_pseudopotential_components(lattice, fft, ions);
    const std::vector<double>& Vion = local_potential.Vtotal;
    const auto projectors =
        build_gaussian_nonlocal_projectors(lattice, basis, ions);

    std::vector<double> rho(grid.ngrid, nelec / volume);
    Eigen::MatrixXcd C_guess = initial_low_kinetic_trials(
        basis.size(),
        std::min(basis.size(), nbands + 4)
    );

    std::vector<double> fixed_occupations(nbands, 0.0);
    fixed_occupations[0] = 2.0;

    PulayMixer pulay;
    pulay.alpha = 0.10;
    pulay.max_history = 6;
    pulay.min_history = 2;
    pulay.regularization = 1.0e-12;

    const int max_scf_iterations = 220;
    const double rho_tolerance = 2.0e-8;
    const double energy_tolerance = 2.0e-11;
    const double eigensolver_tolerance = 2.0e-11;

    SelfConsistentPoint result;
    Eigen::MatrixXcd final_C;
    std::vector<double> final_occupations;
    std::vector<double> final_density;
    double previous_energy = 0.0;
    bool converged = false;

    for (int iter = 0; iter < max_scf_iterations; ++iter) {
        const auto VH = build_hartree_potential(lattice, fft, rho);
        const auto exchange = build_lda_exchange(rho, dV);
        const auto Veff = combine_effective_potential(
            Vion,
            VH,
            exchange.Vx
        );

        const DavidsonResult ks = davidson_lowest_eigenstates(
            basis,
            fft,
            Veff,
            nbands,
            C_guess,
            90,
            std::min(basis.size(), 4 * nbands + 12),
            eigensolver_tolerance,
            1.0e-6,
            &projectors,
            false
        );

        if (!ks.converged) {
            throw std::runtime_error("Davidson did not converge in SCF FD test.");
        }

        const OccupationResult occupations = compute_occupations(
            ks.eigenvalues,
            nelec,
            OccupationMode::Fixed,
            fixed_occupations,
            0.0,
            1.0e-8
        );

        const auto rho_out = build_density_from_orbitals(
            basis,
            fft,
            ks.eigenvectors.leftCols(nbands),
            occupations.occ,
            volume
        );
        const auto VH_out = build_hartree_potential(lattice, fft, rho_out);
        const auto exchange_out = build_lda_exchange(rho_out, dV);
        const double Enl = compute_nonlocal_energy(
            projectors,
            ks.eigenvectors.leftCols(nbands),
            occupations.occ
        );

        EnergyTerms energy = compute_total_energy(
            basis,
            ks.eigenvectors.leftCols(nbands),
            occupations.occ,
            rho_out,
            Vion,
            VH_out,
            exchange_out.Ex,
            dV,
            0.0,
            0.0,
            Enl
        );
        energy.ion_smooth = local_potential.Eion_smooth;
        energy.total_with_ion_smooth =
            energy.total + energy.ion_smooth;

        const double drho = pulay_mix_density(
            pulay,
            rho,
            rho_out,
            dV,
            nelec
        );
        const double dE = (iter == 0)
            ? 0.0
            : std::abs(energy.total_with_ion_smooth - previous_energy);

        final_C = ks.eigenvectors.leftCols(nbands);
        final_occupations = occupations.occ;
        final_density = rho_out;
        result.energy = energy.total_with_ion_smooth;
        result.scf_iterations = iter + 1;
        result.final_drho = drho;
        result.final_dE = dE;

        if (iter > 0 && drho < rho_tolerance && dE < energy_tolerance) {
            converged = true;
            break;
        }

        previous_energy = energy.total_with_ion_smooth;
        C_guess = ks.eigenvectors.leftCols(nbands);
    }

    if (!converged) {
        throw std::runtime_error("SCF did not converge in force FD test.");
    }

    const auto n_G = build_density_G(fft, final_density);
    const IonicForceComponents forces = compute_ionic_forces(
        lattice,
        grid,
        basis,
        ions,
        n_G,
        projectors,
        final_C,
        final_occupations
    );
    result.force_on_ion0 = forces.total[0];

    return result;
}

Ion make_ion(const Eigen::Vector3d& position) {
    Ion ion;
    ion.frac_position = position;
    ion.Z = 1.0;
    ion.sigma = 0.70;
    ion.short_amp = 0.65;
    ion.short_rc = 0.55;
    ion.beta_s_rc = 0.60;
    ion.beta_s_D = 0.18;
    ion.beta_p_rc = 0.52;
    ion.beta_p_D = -0.06;
    return ion;
}

} // namespace

int main() {
    try {
        const double a = 10.0;
        const Lattice lattice(
            Eigen::Vector3d(a, 0.0, 0.0),
            Eigen::Vector3d(0.0, a, 0.0),
            Eigen::Vector3d(0.0, 0.0, a)
        );
        const FFTGrid grid(20, 20, 20);
        PlaneWaveBasis3D basis;
        basis.generate(lattice, Eigen::Vector3d::Zero(), 2.2);

        const std::vector<Ion> ions{
            make_ion(Eigen::Vector3d(0.27, 0.39, 0.46)),
            make_ion(Eigen::Vector3d(0.68, 0.57, 0.73))
        };

        std::cout << std::setprecision(12);
        std::cout << "Running central asymmetric two-ion SCF...\n";
        const SelfConsistentPoint center = run_scf_from_scratch(
            lattice,
            grid,
            basis,
            ions
        );

        const double h = 2.0e-3;
        Eigen::Vector3d finite_difference_force = Eigen::Vector3d::Zero();

        for (int a_cart = 0; a_cart < 3; ++a_cart) {
            std::cout << "Re-converging displaced SCFs for direction "
                      << a_cart << "...\n";
            const auto plus_ions = displace_ion_cartesian(
                lattice,
                ions,
                0,
                a_cart,
                h
            );
            const auto minus_ions = displace_ion_cartesian(
                lattice,
                ions,
                0,
                a_cart,
                -h
            );
            const SelfConsistentPoint plus = run_scf_from_scratch(
                lattice,
                grid,
                basis,
                plus_ions
            );
            const SelfConsistentPoint minus = run_scf_from_scratch(
                lattice,
                grid,
                basis,
                minus_ions
            );

            finite_difference_force[a_cart] =
                -(plus.energy - minus.energy) / (2.0 * h);

            std::cout << "  E(+h) = " << plus.energy
                      << "  E(-h) = " << minus.energy
                      << "  iterations = " << plus.scf_iterations
                      << "/" << minus.scf_iterations << "\n";
        }

        const Eigen::Vector3d error =
            center.force_on_ion0 - finite_difference_force;
        const double max_error = error.cwiseAbs().maxCoeff();

        std::cout << "Analytic force      = "
                  << center.force_on_ion0.transpose() << "\n";
        std::cout << "SCF finite diff     = "
                  << finite_difference_force.transpose() << "\n";
        std::cout << "Difference          = "
                  << error.transpose() << "\n";
        std::cout << "max |difference|    = " << max_error << "\n";
        std::cout << "central SCF iters   = "
                  << center.scf_iterations
                  << "  drho = " << center.final_drho
                  << "  dE = " << center.final_dE << "\n";

        if (max_error >= 2.0e-5) {
            throw std::runtime_error(
                "Self-consistent finite-difference force error is too large."
            );
        }
    } catch (const std::exception& error) {
        std::cerr << "SCF force finite-difference test failed: "
                  << error.what() << "\n";
        return 1;
    }

    std::cout << "Self-consistent force finite-difference test passed.\n";
    return 0;
}
