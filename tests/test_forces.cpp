#include "forces.hpp"
#include "potentials.hpp"
#include "scf_modules.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestSystem {
    Lattice lattice;
    FFTGrid grid;
    FFTWorkspace fft;
    PlaneWaveBasis3D basis;

    TestSystem()
        : lattice(
              Eigen::Vector3d(9.0, 0.0, 0.0),
              Eigen::Vector3d(0.0, 9.0, 0.0),
              Eigen::Vector3d(0.0, 0.0, 9.0)),
          grid(21, 21, 21),
          fft(grid) {
        basis.generate(lattice, Eigen::Vector3d::Zero(), 3.0);
    }
};

double max_abs_component(const std::vector<Eigen::Vector3d>& v) {
    double result = 0.0;
    for (const auto& x : v) {
        result = std::max(result, x.cwiseAbs().maxCoeff());
    }
    return result;
}

void require_less(double value, double tolerance, const std::string& label) {
    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

void wrap_fractional(Eigen::Vector3d& s) {
    for (int a = 0; a < 3; ++a) {
        s[a] -= std::floor(s[a]);
    }
}

std::vector<Ion> displaced_ions(
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

std::vector<double> make_periodic_density(
    const TestSystem& system,
    const Eigen::Vector3d& center_frac,
    double nelec) {

    std::vector<double> rho(system.grid.ngrid, 0.0);
    const double mean = nelec / system.lattice.volume();

    for (int i = 0; i < system.grid.n1; ++i) {
        for (int j = 0; j < system.grid.n2; ++j) {
            for (int k = 0; k < system.grid.n3; ++k) {
                const int p = system.grid.index(i, j, k);
                const Eigen::Vector3d s = system.grid.frac_coord(i, j, k);
                const Eigen::Vector3d ds = s - center_frac;

                rho[p] = mean * (
                    1.0
                    + 0.20 * std::cos(2.0 * M_PI * ds[0])
                    + 0.13 * std::cos(2.0 * M_PI * ds[1])
                    + 0.09 * std::cos(2.0 * M_PI * ds[2])
                );
            }
        }
    }

    return rho;
}

Eigen::VectorXcd make_centered_orbital(
    const TestSystem& system,
    const Eigen::Vector3d& center_frac) {

    const Eigen::Vector3d R =
        system.lattice.cart_from_frac(center_frac);
    Eigen::VectorXcd c(system.basis.size());

    for (int ig = 0; ig < system.basis.size(); ++ig) {
        const Eigen::Vector3d G = system.basis.gvectors[ig].G_cart;
        const double phase = -G.dot(R);
        c[ig] =
            std::exp(-0.18 * G.squaredNorm())
            * std::complex<double>(std::cos(phase), std::sin(phase));
    }

    c.normalize();
    return c;
}

Ion make_ion(const Eigen::Vector3d& position) {
    Ion ion;
    ion.frac_position = position;
    ion.Z = 2.0;
    ion.sigma = 0.65;
    ion.short_amp = 0.8;
    ion.short_rc = 0.55;
    ion.beta_s_rc = 0.60;
    ion.beta_s_D = 0.45;
    ion.beta_p_rc = 0.50;
    ion.beta_p_D = -0.12;
    return ion;
}

Eigen::Vector3d finite_difference_force(
    const Lattice& lattice,
    const std::vector<Ion>& ions,
    int iion,
    double h,
    const std::function<double(const std::vector<Ion>&)>& energy) {

    Eigen::Vector3d force = Eigen::Vector3d::Zero();

    for (int a = 0; a < 3; ++a) {
        const auto plus = displaced_ions(lattice, ions, iion, a, h);
        const auto minus = displaced_ions(lattice, ions, iion, a, -h);
        force[a] = -(energy(plus) - energy(minus)) / (2.0 * h);
    }

    return force;
}

void test_single_ion_high_symmetry() {
    TestSystem system;
    const Eigen::Vector3d center(0.5, 0.5, 0.5);
    std::vector<Ion> ions{make_ion(center)};
    const auto rho = make_periodic_density(system, center, 2.0);
    const auto n_G = build_density_G(system.fft, rho);
    const auto projectors =
        build_gaussian_nonlocal_projectors(system.lattice, system.basis, ions);

    Eigen::MatrixXcd C(system.basis.size(), 1);
    C.col(0) = make_centered_orbital(system, center);
    const std::vector<double> occupations{2.0};

    const auto forces = compute_ionic_forces(
        system.lattice,
        system.grid,
        system.basis,
        ions,
        n_G,
        projectors,
        C,
        occupations
    );

    require_less(
        max_abs_component(forces.total),
        2.0e-12,
        "single-ion high-symmetry |F|max"
    );
}

void test_two_ion_action_reaction() {
    TestSystem system;
    std::vector<Ion> ions{
        make_ion(Eigen::Vector3d(0.34, 0.5, 0.5)),
        make_ion(Eigen::Vector3d(0.66, 0.5, 0.5))
    };

    // Use only s projectors so the complete setup is inversion symmetric.
    ions[0].beta_p_D = 0.0;
    ions[1].beta_p_D = 0.0;

    const Eigen::Vector3d center(0.5, 0.5, 0.5);
    const auto rho = make_periodic_density(system, center, 4.0);
    const auto n_G = build_density_G(system.fft, rho);
    const auto projectors =
        build_gaussian_nonlocal_projectors(system.lattice, system.basis, ions);

    Eigen::MatrixXcd C(system.basis.size(), 1);
    C.col(0) = make_centered_orbital(system, center);
    const std::vector<double> occupations{2.0};

    const auto forces = compute_ionic_forces(
        system.lattice,
        system.grid,
        system.basis,
        ions,
        n_G,
        projectors,
        C,
        occupations
    );

    require_less(
        (forces.total[0] + forces.total[1]).cwiseAbs().maxCoeff(),
        3.0e-12,
        "two-ion |F1+F2|max"
    );
}

void test_finite_differences() {
    TestSystem system;
    std::vector<Ion> ions{
        make_ion(Eigen::Vector3d(0.31, 0.43, 0.27))
    };

    const auto rho = make_periodic_density(
        system,
        Eigen::Vector3d(0.19, 0.61, 0.38),
        2.0
    );
    const auto n_G = build_density_G(system.fft, rho);
    const double dV =
        system.lattice.volume() / static_cast<double>(system.grid.ngrid);
    const double h = 2.0e-5;

    const auto F_local = compute_local_ionic_forces(
        system.lattice,
        system.grid,
        ions,
        n_G
    );
    const Eigen::Vector3d F_local_fd = finite_difference_force(
        system.lattice,
        ions,
        0,
        h,
        [&](const std::vector<Ion>& moved) {
            const auto potential = build_local_pseudopotential_components(
                system.lattice,
                system.fft,
                moved
            );
            return compute_external_energy(rho, potential.Vtotal, dV);
        }
    );

    require_less(
        (F_local[0] - F_local_fd).cwiseAbs().maxCoeff(),
        2.0e-8,
        "local-force finite-difference error"
    );

    std::vector<Ion> pair = ions;
    pair.push_back(make_ion(Eigen::Vector3d(0.72, 0.58, 0.81)));
    pair[1].Z = 1.4;
    pair[1].sigma = 0.48;

    const auto F_II = compute_smooth_ion_ion_forces(
        system.lattice,
        system.grid,
        pair
    );
    const Eigen::Vector3d F_II_fd = finite_difference_force(
        system.lattice,
        pair,
        0,
        h,
        [&](const std::vector<Ion>& moved) {
            return compute_smooth_ion_ion_energy(
                system.lattice,
                system.grid,
                moved
            );
        }
    );

    require_less(
        (F_II[0] - F_II_fd).cwiseAbs().maxCoeff(),
        2.0e-8,
        "ion-ion-force finite-difference error"
    );

    Eigen::MatrixXcd C(system.basis.size(), 1);
    C.col(0) = make_centered_orbital(
        system,
        Eigen::Vector3d(0.57, 0.22, 0.69)
    );
    const std::vector<double> occupations{1.6};
    const auto projectors =
        build_gaussian_nonlocal_projectors(system.lattice, system.basis, ions);
    const auto F_NL = compute_nonlocal_ionic_forces(
        system.basis,
        projectors,
        C,
        occupations,
        static_cast<int>(ions.size())
    );
    const Eigen::Vector3d F_NL_fd = finite_difference_force(
        system.lattice,
        ions,
        0,
        h,
        [&](const std::vector<Ion>& moved) {
            const auto moved_projectors = build_gaussian_nonlocal_projectors(
                system.lattice,
                system.basis,
                moved
            );
            return compute_nonlocal_energy(
                moved_projectors,
                C,
                occupations
            );
        }
    );

    require_less(
        (F_NL[0] - F_NL_fd).cwiseAbs().maxCoeff(),
        2.0e-8,
        "nonlocal-force finite-difference error"
    );
}

} // namespace

int main() {
    try {
        test_single_ion_high_symmetry();
        test_two_ion_action_reaction();
        test_finite_differences();
    } catch (const std::exception& error) {
        std::cerr << "Force test failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "All analytic-force tests passed.\n";
    return 0;
}
