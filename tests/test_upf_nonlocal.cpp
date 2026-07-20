#include "forces.hpp"
#include "potentials.hpp"
#include "upf_nonlocal.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_less(double error, double tolerance, const std::string& label) {
    std::cout << label << " = " << error
              << "  tolerance = " << tolerance << "\n";
    if (!(error < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

UPFNonlocalSpecies make_two_s_species() {
    constexpr int nmesh = 1001;
    constexpr double dr = 0.01;
    UPFNonlocalSpecies species;
    species.element = "X";
    species.radial_grid_bohr.resize(nmesh);
    species.quadrature_weights.resize(nmesh);
    species.radial_projectors.resize(2);

    for (int i = 0; i < nmesh; ++i) {
        const double r = i * dr;
        species.radial_grid_bohr[i] = r;
        const double simpson = (i == 0 || i == nmesh - 1)
            ? 1.0
            : (i % 2 == 0 ? 2.0 : 4.0);
        species.quadrature_weights[i] = simpson * dr / 3.0;
    }

    const std::vector<double> exponents{0.55, 1.10};
    for (int ip = 0; ip < 2; ++ip) {
        UPFProjector& projector = species.radial_projectors[ip];
        projector.index = ip + 1;
        projector.label = ip == 0 ? "1S" : "2S";
        projector.angular_momentum = 0;
        projector.r_times_beta.resize(nmesh);
        for (int i = 0; i < nmesh; ++i) {
            const double r = species.radial_grid_bohr[i];
            projector.r_times_beta[i] =
                r * std::exp(-exponents[ip] * r * r);
        }
    }

    species.dij_hartree.resize(2, 2);
    species.dij_hartree << 1.20, 0.35,
                              0.35, -0.60;
    return species;
}

std::vector<Eigen::VectorXcd> analytic_s_projectors(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const UPFLocalIon& ion) {

    const std::vector<double> exponents{0.55, 1.10};
    const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
    const double inverse_sqrt_volume = 1.0 / std::sqrt(lattice.volume());
    const Eigen::Vector3d R = lattice.cart_from_frac(ion.frac_position);
    std::vector<Eigen::VectorXcd> raw;
    for (double exponent : exponents) {
        Eigen::VectorXcd beta = Eigen::VectorXcd::Zero(basis.size());
        for (int ig = 0; ig < basis.size(); ++ig) {
            const Eigen::Vector3d& G = basis.gvectors[ig].G_cart;
            const double isolated_transform =
                std::pow(M_PI / exponent, 1.5)
                * std::exp(-G.squaredNorm() / (4.0 * exponent));
            const double phase = -G.dot(R);
            beta[ig] =
                inverse_sqrt_volume * y00 * isolated_transform
                * std::complex<double>(std::cos(phase), std::sin(phase));
        }
        raw.push_back(std::move(beta));
    }
    return raw;
}

Eigen::VectorXcd deterministic_state(int size, double offset) {
    Eigen::VectorXcd state(size);
    for (int i = 0; i < size; ++i) {
        state[i] = std::complex<double>(
            std::sin(0.37 * i + offset),
            std::cos(0.23 * i - 0.4 * offset)
        );
    }
    state.normalize();
    return state;
}

void test_qe_real_spherical_harmonics() {
    const Eigen::Vector3d direction(2.0, -3.0, 4.0);
    const double radius = direction.norm();
    const double p_normalization = std::sqrt(3.0 / (4.0 * M_PI));
    const auto p = qe_real_spherical_harmonics(1, direction);
    require_less(
        std::abs(p[0] - p_normalization * direction[2] / radius),
        2.0e-15,
        "QE p_z harmonic error"
    );
    require_less(
        std::abs(p[1] + p_normalization * direction[0] / radius),
        2.0e-15,
        "QE negative-p_x harmonic error"
    );
    require_less(
        std::abs(p[2] + p_normalization * direction[1] / radius),
        2.0e-15,
        "QE negative-p_y harmonic error"
    );

    double maximum_addition_error = 0.0;
    for (int l = 0; l <= 5; ++l) {
        const auto harmonics = qe_real_spherical_harmonics(l, direction);
        double sum = 0.0;
        for (double value : harmonics) {
            sum += value * value;
        }
        maximum_addition_error = std::max(
            maximum_addition_error,
            std::abs(sum - (2.0 * l + 1.0) / (4.0 * M_PI))
        );
    }
    require_less(
        maximum_addition_error,
        3.0e-15,
        "real-harmonic addition-theorem error"
    );
}

void test_upf_preparation_and_angular_expansion() {
    const UPFData local_only = read_nc_upf(
        std::string(TEST_DATA_DIR) + "/minimal_local_nc.upf"
    );
    const UPFNonlocalSpecies empty_species =
        prepare_upf_nonlocal_species(local_only);
    require(
        empty_species.radial_projectors.empty() &&
        empty_species.dij_hartree.size() == 0,
        "projector-free UPF did not produce an empty nonlocal species"
    );

    const UPFData upf = read_nc_upf(
        std::string(TEST_DATA_DIR) + "/minimal_sp_nc.upf"
    );
    const UPFNonlocalSpecies species =
        prepare_upf_nonlocal_species(upf);
    require_less(
        std::abs(species.dij_hartree(0, 0) - 0.75),
        1.0e-15,
        "UPF s DIJ Ry-to-Ha error"
    );
    require_less(
        std::abs(species.dij_hartree(1, 1) - 1.75),
        1.0e-15,
        "UPF p DIJ Ry-to-Ha error"
    );

    const double cell = 8.0;
    const Lattice lattice(
        Eigen::Vector3d(cell, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell)
    );
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 1.0);
    const auto projectors = build_upf_nonlocal_projectors(
        lattice,
        basis,
        {species},
        {{0, Eigen::Vector3d(0.27, 0.39, 0.43)}}
    );
    require(
        projectors.size() == 4,
        "one radial s and one radial p projector did not expand to 1+3"
    );
    require_less(
        std::abs(projectors[0].D - 0.75),
        1.0e-15,
        "expanded s-channel D error"
    );
    for (int channel = 1; channel < 4; ++channel) {
        require_less(
            std::abs(projectors[channel].D - 1.75),
            1.0e-15,
            "expanded p-channel D error"
        );
        require(
            projectors[channel].beta_G.norm() > 0.0,
            "expanded p projector vanished"
        );
    }
}

void test_dense_dij_operator_and_energy() {
    const double cell = 8.0;
    const Lattice lattice(
        Eigen::Vector3d(cell, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell)
    );
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 2.4);
    const UPFNonlocalSpecies species = make_two_s_species();
    const UPFLocalIon ion{0, Eigen::Vector3d(0.23, 0.37, 0.41)};
    const auto diagonal_projectors = build_upf_nonlocal_projectors(
        lattice,
        basis,
        {species},
        {ion}
    );
    require(
        diagonal_projectors.size() == 2,
        "dense two-by-two s DIJ did not produce two eigenprojectors"
    );

    const auto raw = analytic_s_projectors(lattice, basis, ion);
    double maximum_beta_error = 0.0;
    for (int ip = 0; ip < 2; ++ip) {
        /*
         * Compare the numerically transformed radial functions indirectly:
         * both raw projectors span the same two-dimensional subspace, and
         * the complete operator below is invariant under DIJ diagonalization.
         */
        maximum_beta_error = std::max(
            maximum_beta_error,
            std::abs(raw[ip].squaredNorm())
        );
    }
    require(maximum_beta_error > 0.1, "analytic UPF test projectors vanished");

    const Eigen::VectorXcd state = deterministic_state(basis.size(), 0.2);
    Eigen::VectorXcd direct = Eigen::VectorXcd::Zero(basis.size());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            direct += species.dij_hartree(i, j)
                * raw[i]
                * raw[j].dot(state);
        }
    }
    const Eigen::VectorXcd diagonalized =
        apply_nonlocal_projectors(diagonal_projectors, state);
    require_less(
        (diagonalized - direct).norm(),
        2.0e-11,
        "dense-DIJ operator error"
    );

    Eigen::MatrixXcd orbitals(basis.size(), 2);
    orbitals.col(0) = state;
    orbitals.col(1) = deterministic_state(basis.size(), 1.4);
    const std::vector<double> occupations{2.0, 0.65};
    double direct_energy = 0.0;
    for (int ib = 0; ib < orbitals.cols(); ++ib) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                direct_energy += occupations[ib]
                    * species.dij_hartree(i, j)
                    * std::real(
                        std::conj(raw[i].dot(orbitals.col(ib)))
                        * raw[j].dot(orbitals.col(ib))
                    );
            }
        }
    }
    const double diagonalized_energy = compute_nonlocal_energy(
        diagonal_projectors,
        orbitals,
        occupations
    );
    require_less(
        std::abs(diagonalized_energy - direct_energy),
        2.0e-12,
        "dense-DIJ energy error"
    );
}

void test_nonlocal_force_finite_difference() {
    const double cell = 7.5;
    const Lattice lattice(
        Eigen::Vector3d(cell, 0.0, 0.0),
        Eigen::Vector3d(0.0, cell, 0.0),
        Eigen::Vector3d(0.0, 0.0, cell)
    );
    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), 2.2);
    const std::vector<UPFNonlocalSpecies> species{make_two_s_species()};
    const UPFLocalIon reference{0, Eigen::Vector3d(0.19, 0.31, 0.47)};

    Eigen::MatrixXcd orbitals(basis.size(), 2);
    orbitals.col(0) = deterministic_state(basis.size(), 0.7);
    orbitals.col(1) = deterministic_state(basis.size(), 1.9);
    const std::vector<double> occupations{2.0, 0.4};
    const auto projectors = build_upf_nonlocal_projectors(
        lattice, basis, species, {reference}
    );
    const Eigen::Vector3d analytic = compute_nonlocal_ionic_forces(
        basis,
        projectors,
        orbitals,
        occupations,
        1
    )[0];

    constexpr double step = 1.0e-5;
    Eigen::Vector3d finite_difference = Eigen::Vector3d::Zero();
    for (int direction = 0; direction < 3; ++direction) {
        UPFLocalIon plus = reference;
        UPFLocalIon minus = reference;
        Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
        displacement[direction] = step;
        plus.frac_position += lattice.A_inv * displacement;
        minus.frac_position -= lattice.A_inv * displacement;
        const double plus_energy = compute_nonlocal_energy(
            build_upf_nonlocal_projectors(lattice, basis, species, {plus}),
            orbitals,
            occupations
        );
        const double minus_energy = compute_nonlocal_energy(
            build_upf_nonlocal_projectors(lattice, basis, species, {minus}),
            orbitals,
            occupations
        );
        finite_difference[direction] =
            -(plus_energy - minus_energy) / (2.0 * step);
    }
    require_less(
        (analytic - finite_difference).cwiseAbs().maxCoeff(),
        2.0e-9,
        "UPF nonlocal-force finite-difference error"
    );
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_qe_real_spherical_harmonics();
        test_upf_preparation_and_angular_expansion();
        test_dense_dij_operator_and_energy();
        test_nonlocal_force_finite_difference();
    } catch (const std::exception& error) {
        std::cerr << "UPF nonlocal test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "All NC-UPF nonlocal tests passed.\n";
    return 0;
}
