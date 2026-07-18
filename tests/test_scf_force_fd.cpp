#include "forces.hpp"
#include "potentials.hpp"
#include "scf.hpp"

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
    const LocalPotentialComponents local_potential =
        build_local_pseudopotential_components(lattice, fft, ions);
    const auto projectors =
        build_gaussian_nonlocal_projectors(lattice, basis, ions);

    SCFOptions options;
    options.nelec = nelec;
    options.nbands = nbands;
    options.occupation_mode = OccupationMode::Fixed;
    options.fixed_occupations.assign(nbands, 0.0);
    options.fixed_occupations[0] = 2.0;
    options.max_iterations = 220;
    options.density_tolerance = 2.0e-8;
    options.energy_tolerance = 2.0e-11;
    options.eigensolver_max_iterations = 90;
    options.eigensolver_max_subspace =
        std::min(basis.size(), 4 * nbands + 12);
    options.eigensolver_tolerance = 2.0e-11;
    options.mixing_alpha = 0.10;
    options.pulay_max_history = 6;
    options.pulay_min_history = 2;
    options.pulay_regularization = 1.0e-12;

    const SCFResult scf = run_scf(
        lattice,
        basis,
        fft,
        local_potential.Vtotal,
        local_potential.Eion_smooth,
        projectors,
        options
    );

    if (!scf.converged) {
        throw std::runtime_error("SCF did not converge in force FD test.");
    }

    const auto n_G = build_density_G(fft, scf.density);
    const IonicForceComponents forces = compute_ionic_forces(
        lattice,
        grid,
        basis,
        ions,
        n_G,
        projectors,
        scf.orbitals,
        scf.occupations.occ
    );

    SelfConsistentPoint result;
    result.energy = scf.variational_energy;
    result.force_on_ion0 = forces.total[0];
    result.scf_iterations = scf.iterations;
    result.final_drho = scf.final_density_residual;
    result.final_dE = scf.final_energy_change;

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
