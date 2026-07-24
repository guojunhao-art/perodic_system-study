#include "forces.hpp"
#include "upf_local_potential.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double reference_sigma = 0.72;
constexpr double gaussian_amplitude = 1.25;
constexpr double gaussian_exponent = 0.60;

void require_less(double value, double tolerance, const std::string& label) {
    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

UPFData make_analytic_local_upf() {
    const int npoints = 6001;
    const double rmax = 12.0;
    const double h = rmax / static_cast<double>(npoints - 1);

    UPFData upf;
    upf.header.element = "X";
    upf.header.z_valence = 3.0;
    upf.header.mesh_size = npoints;
    upf.mesh.r.resize(npoints);
    upf.mesh.rab.assign(npoints, h);
    upf.local_potential_ry.resize(npoints);

    for (int i = 0; i < npoints; ++i) {
        const double r = h * static_cast<double>(i);
        upf.mesh.r[i] = r;
        const double local_hartree =
            -upf.header.z_valence
                * gaussian_coulomb_shape(r, reference_sigma)
            + gaussian_amplitude
                * std::exp(-gaussian_exponent * r * r);
        upf.local_potential_ry[i] =
            local_hartree / RYDBERG_TO_HARTREE;
    }
    return upf;
}

double analytic_gaussian_transform(double g) {
    return gaussian_amplitude
        * std::pow(M_PI / gaussian_exponent, 1.5)
        * std::exp(
            -g * g / (4.0 * gaussian_exponent)
        );
}

double analytic_full_kernel(double z, double g) {
    return -4.0 * M_PI * z
        * std::exp(-0.5 * reference_sigma * reference_sigma * g * g)
        / (g * g)
        + analytic_gaussian_transform(g);
}

void test_screening_split_and_sigma_invariance() {
    const UPFData upf = make_analytic_local_upf();
    const UPFLocalSpecies reference =
        prepare_upf_local_species(upf, reference_sigma);

    double max_correction_error = 0.0;
    double max_kernel_error = 0.0;
    for (double g : std::vector<double>{0.0, 0.2, 0.8, 1.7, 3.0}) {
        max_correction_error = std::max(
            max_correction_error,
            std::abs(
                upf_local_correction_transform(reference, g)
                - analytic_gaussian_transform(g)
            )
        );
        if (g > 0.0) {
            max_kernel_error = std::max(
                max_kernel_error,
                std::abs(
                    upf_local_kernel_G(reference, g)
                    - analytic_full_kernel(upf.header.z_valence, g)
                )
            );
        }
    }

    require_less(
        max_correction_error,
        2.0e-10,
        "screened local correction-transform error"
    );
    require_less(
        max_kernel_error,
        2.0e-10,
        "screened full local-kernel error"
    );
    const double analytic_g0 =
        analytic_gaussian_transform(0.0)
        + 2.0 * M_PI * upf.header.z_valence
            * reference_sigma * reference_sigma;
    require_less(
        std::abs(upf_local_kernel_G(reference, 0.0) - analytic_g0),
        2.0e-10,
        "standard local G=0 correction error"
    );

    double max_sigma_dependence = 0.0;
    for (double sigma : std::vector<double>{0.45, 1.05}) {
        const UPFLocalSpecies alternative =
            prepare_upf_local_species(upf, sigma);
        for (double g : std::vector<double>{0.0, 0.2, 0.8, 1.7, 3.0}) {
            max_sigma_dependence = std::max(
                max_sigma_dependence,
                std::abs(
                    upf_local_kernel_G(alternative, g)
                    - upf_local_kernel_G(reference, g)
                )
            );
        }
    }
    require_less(
        max_sigma_dependence,
        2.0e-9,
        "full-kernel screening-width dependence"
    );
}

double local_energy(
    double volume,
    const std::vector<std::complex<double>>& density_G,
    const std::vector<std::complex<double>>& potential_G) {

    double energy = 0.0;
    for (int p = 0; p < static_cast<int>(density_G.size()); ++p) {
        energy += volume
            * std::real(std::conj(density_G[p]) * potential_G[p]);
    }
    return energy;
}

std::vector<UPFLocalIon> displace_ion(
    const Lattice& lattice,
    const std::vector<UPFLocalIon>& ions,
    int ion_index,
    int direction,
    double displacement_bohr) {

    auto moved = ions;
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    displacement[direction] = displacement_bohr;
    moved[ion_index].frac_position += lattice.A_inv * displacement;
    return moved;
}

void test_periodic_coefficients_and_local_force() {
    const double a = 10.0;
    const Lattice lattice(
        Eigen::Vector3d(a, 0.0, 0.0),
        Eigen::Vector3d(0.0, a, 0.0),
        Eigen::Vector3d(0.0, 0.0, a)
    );
    const FFTGrid grid(7, 7, 7);
    const std::vector<UPFLocalSpecies> species{
        prepare_upf_local_species(make_analytic_local_upf(), reference_sigma)
    };
    const std::vector<UPFLocalIon> ions{
        {0, Eigen::Vector3d(0.23, 0.37, 0.41)},
        {0, Eigen::Vector3d(0.66, 0.58, 0.72)}
    };

    const UPFLocalReciprocalCache cache =
        build_upf_local_reciprocal_cache(
            lattice,
            grid,
            species,
            2
        );
    const auto potential_G =
        build_upf_local_potential_G(lattice, grid, species, ions);
    const auto cached_potential_G =
        build_upf_local_potential_G(lattice, cache, ions, 2);
    double maximum_cache_error = 0.0;
    for (int p = 0; p < grid.ngrid; ++p) {
        maximum_cache_error = std::max(
            maximum_cache_error,
            std::abs(cached_potential_G[p] - potential_G[p])
        );
    }
    require_less(
        maximum_cache_error,
        2.0e-13,
        "cached local-potential coefficient error"
    );
    const int zero_index = grid.index_from_freq(Eigen::Vector3i::Zero());
    const double expected_G0 =
        static_cast<double>(ions.size())
        * upf_local_kernel_G(species[0], 0.0)
        / lattice.volume();
    require_less(
        std::abs(potential_G[zero_index] - expected_G0),
        2.0e-13,
        "periodic local G=0 coefficient error"
    );

    double max_hermitian_error = 0.0;
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);
                const int minus_p = grid.index_from_freq(-n);
                max_hermitian_error = std::max(
                    max_hermitian_error,
                    std::abs(
                        potential_G[minus_p] - std::conj(potential_G[p])
                    )
                );
            }
        }
    }
    require_less(
        max_hermitian_error,
        2.0e-13,
        "periodic local Hermitian-symmetry error"
    );

    FFTWorkspace fft(grid);
    const auto potential_real = build_upf_local_potential_real(
        lattice,
        fft,
        species,
        ions
    );
    const int real_index = grid.index(2, 3, 1);
    const Eigen::Vector3d real_fractional =
        grid.frac_coord(2, 3, 1);
    std::complex<double> direct_sum(0.0, 0.0);
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);
                const double phase =
                    2.0 * M_PI * n.cast<double>().dot(real_fractional);
                direct_sum += potential_G[p] * std::complex<double>(
                    std::cos(phase),
                    std::sin(phase)
                );
            }
        }
    }
    require_less(
        std::abs(potential_real[real_index] - direct_sum.real()),
        2.0e-12,
        "UPF local inverse-FFT error"
    );

    std::vector<std::complex<double>> density_G(
        grid.ngrid,
        std::complex<double>(0.0, 0.0)
    );
    const std::vector<std::pair<Eigen::Vector3i, std::complex<double>>> modes{
        {Eigen::Vector3i(1, 0, 0), {0.014, 0.006}},
        {Eigen::Vector3i(0, 1, 1), {-0.009, 0.004}},
        {Eigen::Vector3i(1, -1, 0), {0.007, -0.005}}
    };
    for (const auto& mode : modes) {
        density_G[grid.index_from_freq(mode.first)] = mode.second;
        density_G[grid.index_from_freq(-mode.first)] =
            std::conj(mode.second);
    }

    const auto analytic_force = compute_upf_local_ionic_forces(
        lattice,
        grid,
        species,
        ions,
        density_G
    );
    const auto cached_force = compute_upf_local_ionic_forces(
        lattice,
        cache,
        ions,
        density_G,
        2
    );
    require_less(
        (cached_force[0] - analytic_force[0]).cwiseAbs().maxCoeff(),
        2.0e-13,
        "cached local-force error"
    );
    const double h = 1.0e-5;
    Eigen::Vector3d finite_difference = Eigen::Vector3d::Zero();
    for (int direction = 0; direction < 3; ++direction) {
        const auto plus_ions =
            displace_ion(lattice, ions, 0, direction, h);
        const auto minus_ions =
            displace_ion(lattice, ions, 0, direction, -h);
        const double plus_energy = local_energy(
            lattice.volume(),
            density_G,
            build_upf_local_potential_G(
                lattice, grid, species, plus_ions
            )
        );
        const double minus_energy = local_energy(
            lattice.volume(),
            density_G,
            build_upf_local_potential_G(
                lattice, grid, species, minus_ions
            )
        );
        finite_difference[direction] =
            -(plus_energy - minus_energy) / (2.0 * h);
    }

    require_less(
        (analytic_force[0] - finite_difference).cwiseAbs().maxCoeff(),
        2.0e-9,
        "UPF local-force finite-difference error"
    );
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_screening_split_and_sigma_invariance();
        test_periodic_coefficients_and_local_force();
    } catch (const std::exception& error) {
        std::cerr << "UPF local-potential test failed: "
                  << error.what() << "\n";
        return 1;
    }

    std::cout << "All screened UPF local-potential tests passed.\n";
    return 0;
}
