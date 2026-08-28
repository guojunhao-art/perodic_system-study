#include "scf_modules.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_less(double value, double tolerance, const std::string& label) {
    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

void require_true(bool condition, const std::string& label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

XCResult evaluate_on_line(
    const LibXCFunctional& functional,
    const std::vector<double>& density,
    double dV) {

    const double length = dV * static_cast<double>(density.size());
    const Lattice lattice(
        Eigen::Vector3d(length, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 1.0)
    );
    FFTWorkspace fft(
        FFTGrid(static_cast<int>(density.size()), 1, 1), 1
    );
    return functional.evaluate(lattice, fft, density, dV);
}

Lattice skew_lattice() {
    return Lattice(
        Eigen::Vector3d(4.1, 0.2, 0.1),
        Eigen::Vector3d(0.3, 3.7, 0.4),
        Eigen::Vector3d(0.1, 0.5, 4.4)
    );
}

void test_exchange_against_analytic_oracle() {
    const std::vector<double> density{
        0.0, 1.0e-12, 1.0e-6, 1.0e-3, 0.1, 1.0, 10.0
    };
    const double dV = 0.37;
    const LDAExchangeResult analytic = build_lda_exchange(density, dV);
    const LibXCFunctional libxc(XCFunctional::ExchangeOnly);
    const XCResult evaluated = evaluate_on_line(libxc, density, dV);

    double max_potential_error = 0.0;
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        max_potential_error = std::max(
            max_potential_error,
            std::abs(evaluated.Vxc[p] - analytic.Vx[p])
        );
    }
    require_less(
        max_potential_error,
        2.0e-13,
        "LibXC exchange-potential error"
    );
    require_less(
        std::abs(evaluated.exchange_energy - analytic.Ex),
        2.0e-13,
        "LibXC exchange-energy error"
    );
    require_less(
        std::abs(evaluated.correlation_energy),
        1.0e-15,
        "exchange-only correlation-energy error"
    );
}

void test_pz_potential_is_energy_derivative() {
    const LibXCFunctional libxc(XCFunctional::PerdewZunger);
    double max_error = 0.0;

    for (double density : std::vector<double>{0.01, 0.1, 1.0, 5.0}) {
        const double h = 1.0e-6 * density;
        const XCResult center = evaluate_on_line(libxc, {density}, 1.0);
        const double plus = evaluate_on_line(
            libxc, {density + h}, 1.0
        ).total_energy();
        const double minus = evaluate_on_line(
            libxc, {density - h}, 1.0
        ).total_energy();
        const double finite_difference = (plus - minus) / (2.0 * h);
        max_error = std::max(
            max_error,
            std::abs(center.Vxc[0] - finite_difference)
        );
        require_true(
            center.correlation_energy < 0.0,
            "PZ correlation energy should be negative."
        );
    }
    require_less(max_error, 2.0e-9, "LibXC XC variational-derivative error");
}

void test_density_validation() {
    const LibXCFunctional libxc(XCFunctional::PerdewZunger);
    const XCResult tiny_negative = evaluate_on_line(
        libxc, {-1.0e-14, 0.1}, 1.0
    );
    require_less(
        std::abs(tiny_negative.Vxc[0]),
        1.0e-15,
        "tiny-negative density cleanup error"
    );

    bool rejected = false;
    try {
        (void)evaluate_on_line(libxc, {-1.0e-4}, 1.0);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require_true(rejected, "A materially negative density was not rejected.");
}

void test_spin_unpolarized_limit_and_derivatives() {
    const std::vector<double> total_density{0.02, 0.1, 0.7, 2.0};
    std::vector<double> density_up(total_density.size(), 0.0);
    std::vector<double> density_down(total_density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(total_density.size()); ++p) {
        density_up[p] = 0.5 * total_density[p];
        density_down[p] = 0.5 * total_density[p];
    }

    const double dV = 0.31;
    const LibXCFunctional unpolarized(XCFunctional::PerdewZunger, 1);
    const LibXCFunctional polarized(XCFunctional::PerdewZunger, 2);
    const XCResult reference = evaluate_on_line(
        unpolarized, total_density, dV
    );
    const SpinXCResult spin = polarized.evaluate_spin(
        density_up, density_down, dV
    );

    double maximum_potential_error = 0.0;
    for (int p = 0; p < static_cast<int>(total_density.size()); ++p) {
        maximum_potential_error = std::max({
            maximum_potential_error,
            std::abs(spin.Vxc_up[p] - reference.Vxc[p]),
            std::abs(spin.Vxc_down[p] - reference.Vxc[p])
        });
    }
    require_less(
        maximum_potential_error,
        2.0e-12,
        "polarized/unpolarized XC-potential error"
    );
    require_less(
        std::abs(spin.exchange_energy - reference.exchange_energy),
        2.0e-12,
        "polarized/unpolarized exchange-energy error"
    );
    require_less(
        std::abs(spin.correlation_energy - reference.correlation_energy),
        2.0e-12,
        "polarized/unpolarized correlation-energy error"
    );

    const double up = 0.8;
    const double down = 0.2;
    const double h = 1.0e-6;
    const SpinXCResult center =
        polarized.evaluate_spin({up}, {down}, 1.0);
    const double up_derivative = (
        polarized.evaluate_spin({up + h}, {down}, 1.0).total_energy()
        - polarized.evaluate_spin({up - h}, {down}, 1.0).total_energy()
    ) / (2.0 * h);
    const double down_derivative = (
        polarized.evaluate_spin({up}, {down + h}, 1.0).total_energy()
        - polarized.evaluate_spin({up}, {down - h}, 1.0).total_energy()
    ) / (2.0 * h);
    require_less(
        std::abs(center.Vxc_up[0] - up_derivative),
        3.0e-9,
        "spin-up XC variational-derivative error"
    );
    require_less(
        std::abs(center.Vxc_down[0] - down_derivative),
        3.0e-9,
        "spin-down XC variational-derivative error"
    );
}

void test_spectral_gradient_on_skew_cell() {
    const Lattice lattice = skew_lattice();
    const FFTGrid grid(9, 10, 11);
    FFTWorkspace fft(grid, 1);
    const std::array<Eigen::Vector3i, 2> modes{{
        Eigen::Vector3i(1, -2, 1),
        Eigen::Vector3i(2, 1, -1)
    }};
    const std::array<double, 2> amplitudes{{0.071, -0.043}};
    const std::array<double, 2> phases{{0.23, -0.41}};

    std::vector<double> field(grid.ngrid, 0.0);
    std::array<std::vector<double>, 3> expected;
    for (std::vector<double>& component : expected) {
        component.assign(grid.ngrid, 0.0);
    }
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3d fractional =
                    grid.frac_coord(i, j, k);
                field[p] = 0.31;
                for (int mode = 0; mode < 2; ++mode) {
                    const double phase = 2.0 * M_PI
                        * modes[mode].cast<double>().dot(fractional)
                        + phases[mode];
                    const Eigen::Vector3d G =
                        lattice.gvector_from_freq(modes[mode]);
                    field[p] += amplitudes[mode] * std::cos(phase);
                    for (int direction = 0; direction < 3; ++direction) {
                        expected[direction][p] -= amplitudes[mode]
                            * G[direction] * std::sin(phase);
                    }
                }
            }
        }
    }

    const auto gradient = spectral_gradient(lattice, fft, field);
    double maximum_error = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
        for (int p = 0; p < grid.ngrid; ++p) {
            maximum_error = std::max(
                maximum_error,
                std::abs(gradient[direction][p]
                         - expected[direction][p])
            );
        }
    }
    require_less(maximum_error, 2.0e-13, "skew-cell gradient error");
}

void test_spectral_integration_by_parts_and_nyquist() {
    const Lattice lattice = skew_lattice();
    const FFTGrid grid(8, 7, 5);
    FFTWorkspace fft(grid, 1);
    std::vector<double> scalar(grid.ngrid, 0.0);
    std::array<std::vector<double>, 3> vector_field;
    for (std::vector<double>& component : vector_field) {
        component.resize(grid.ngrid);
    }
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3d s = grid.frac_coord(i, j, k);
                scalar[p] = 0.2
                    + 0.07 * std::cos(2.0 * M_PI * (s[0] - 2.0 * s[1]))
                    - 0.04 * std::sin(2.0 * M_PI * (s[1] + s[2]));
                vector_field[0][p] =
                    0.13 * std::sin(2.0 * M_PI * (s[0] + s[2]));
                vector_field[1][p] =
                    -0.09 * std::cos(2.0 * M_PI * (2.0 * s[1] - s[2]));
                vector_field[2][p] =
                    0.05 * std::sin(2.0 * M_PI * (s[0] - s[1]));
            }
        }
    }
    const auto gradient = spectral_gradient(lattice, fft, scalar);
    const std::vector<double> divergence =
        spectral_divergence(lattice, fft, vector_field);
    double left = 0.0;
    double right = 0.0;
    for (int p = 0; p < grid.ngrid; ++p) {
        left += scalar[p] * divergence[p];
        for (int direction = 0; direction < 3; ++direction) {
            right -= gradient[direction][p] * vector_field[direction][p];
        }
    }
    require_less(
        std::abs(left - right),
        2.0e-13,
        "spectral integration-by-parts error"
    );

    std::vector<double> nyquist(grid.ngrid, 0.0);
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                nyquist[grid.index(i, j, k)] = i % 2 == 0 ? 1.0 : -1.0;
            }
        }
    }
    const auto nyquist_gradient = spectral_gradient(lattice, fft, nyquist);
    double maximum_nyquist_gradient = 0.0;
    for (const std::vector<double>& component : nyquist_gradient) {
        for (double value : component) {
            maximum_nyquist_gradient = std::max(
                maximum_nyquist_gradient, std::abs(value)
            );
        }
    }
    require_less(
        maximum_nyquist_gradient,
        1.0e-14,
        "Nyquist gradient"
    );
}

void test_pbe_potential_is_discrete_energy_derivative() {
    const Lattice lattice = skew_lattice();
    const FFTGrid grid(9, 10, 11);
    FFTWorkspace fft(grid, 1);
    const double dV = lattice.volume() / static_cast<double>(grid.ngrid);
    std::vector<double> density(grid.ngrid, 0.0);
    std::vector<double> perturbation(grid.ngrid, 0.0);
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3d s = grid.frac_coord(i, j, k);
                density[p] = 0.18
                    + 0.035 * std::cos(
                        2.0 * M_PI * (s[0] - 2.0 * s[1] + s[2])
                    )
                    + 0.021 * std::sin(
                        2.0 * M_PI * (2.0 * s[0] + s[1] - s[2])
                    );
                perturbation[p] =
                    0.11 * std::cos(2.0 * M_PI * (s[0] + s[2]))
                    - 0.08 * std::sin(2.0 * M_PI * (s[1] - s[2]));
            }
        }
    }

    const LibXCFunctional pbe(
        XCFunctional::PerdewBurkeErnzerhof, 1
    );
    const XCResult center = pbe.evaluate(
        lattice, fft, density, dV
    );
    const double step = 2.0e-6;
    std::vector<double> plus = density;
    std::vector<double> minus = density;
    for (int p = 0; p < grid.ngrid; ++p) {
        plus[p] += step * perturbation[p];
        minus[p] -= step * perturbation[p];
    }
    const double finite_difference = (
        pbe.evaluate(lattice, fft, plus, dV).total_energy()
        - pbe.evaluate(lattice, fft, minus, dV).total_energy()
    ) / (2.0 * step);
    double potential_derivative = 0.0;
    for (int p = 0; p < grid.ngrid; ++p) {
        potential_derivative +=
            dV * center.Vxc[p] * perturbation[p];
    }
    require_less(
        std::abs(finite_difference - potential_derivative),
        3.0e-9,
        "PBE discrete variational-derivative error"
    );
    require_true(
        center.exchange_energy < 0.0 && center.correlation_energy < 0.0,
        "PBE exchange and correlation energies should be negative."
    );

    bool spin_pbe_rejected = false;
    try {
        const LibXCFunctional unsupported(
            XCFunctional::PerdewBurkeErnzerhof, 2
        );
        (void)unsupported;
    } catch (const std::runtime_error&) {
        spin_pbe_rejected = true;
    }
    require_true(
        spin_pbe_rejected,
        "Spin-polarized PBE was not rejected."
    );
}

void test_pseudopotential_functional_labels() {
    require_true(
        pseudopotential_functional_matches_xc(
            "PBE", XCFunctional::PerdewBurkeErnzerhof
        ),
        "A standard PBE UPF label was rejected."
    );
    require_true(
        pseudopotential_functional_matches_xc(
            "SLA-PW-PBX-PBC", XCFunctional::PerdewBurkeErnzerhof
        ),
        "The legacy HGH PBE component label was rejected."
    );
    require_true(
        !pseudopotential_functional_matches_xc(
            "PBESOL", XCFunctional::PerdewBurkeErnzerhof
        ),
        "A PBEsol UPF label was accepted as PBE."
    );
    require_true(
        pseudopotential_functional_matches_xc(
            "SLA PZ NOGX NOGC", XCFunctional::PerdewZunger
        ) &&
        !pseudopotential_functional_matches_xc(
            "PBE", XCFunctional::PerdewZunger
        ),
        "PZ-LDA UPF label matching is inconsistent."
    );
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12)
                  << "LibXC runtime version = "
                  << libxc_runtime_version() << "\n";
        test_exchange_against_analytic_oracle();
        test_pz_potential_is_energy_derivative();
        test_density_validation();
        test_spin_unpolarized_limit_and_derivatives();
        test_spectral_gradient_on_skew_cell();
        test_spectral_integration_by_parts_and_nyquist();
        test_pbe_potential_is_discrete_energy_derivative();
        test_pseudopotential_functional_labels();
    } catch (const std::exception& error) {
        std::cerr << "LibXC LDA/GGA test failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "All LibXC LDA/GGA tests passed.\n";
    return 0;
}
