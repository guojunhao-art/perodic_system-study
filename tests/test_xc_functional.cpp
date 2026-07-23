#include "scf_modules.hpp"
#include "xc_functional.hpp"

#include <algorithm>
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

void test_exchange_against_analytic_oracle() {
    const std::vector<double> density{
        0.0, 1.0e-12, 1.0e-6, 1.0e-3, 0.1, 1.0, 10.0
    };
    const double dV = 0.37;
    const LDAExchangeResult analytic = build_lda_exchange(density, dV);
    const LibXCLDAFunctional libxc(LDAFunctional::ExchangeOnly);
    const XCResult evaluated = libxc.evaluate(density, dV);

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
    const LibXCLDAFunctional libxc(LDAFunctional::PerdewZunger);
    double max_error = 0.0;

    for (double density : std::vector<double>{0.01, 0.1, 1.0, 5.0}) {
        const double h = 1.0e-6 * density;
        const XCResult center = libxc.evaluate({density}, 1.0);
        const double plus =
            libxc.evaluate({density + h}, 1.0).total_energy();
        const double minus =
            libxc.evaluate({density - h}, 1.0).total_energy();
        const double finite_difference = (plus - minus) / (2.0 * h);
        max_error = std::max(
            max_error,
            std::abs(center.Vxc[0] - finite_difference)
        );

        if (!(center.correlation_energy < 0.0)) {
            throw std::runtime_error("PZ correlation energy should be negative.");
        }
    }

    require_less(max_error, 2.0e-9, "LibXC XC variational-derivative error");
}

void test_density_validation() {
    const LibXCLDAFunctional libxc(LDAFunctional::PerdewZunger);
    const XCResult tiny_negative = libxc.evaluate({-1.0e-14, 0.1}, 1.0);
    require_less(
        std::abs(tiny_negative.Vxc[0]),
        1.0e-15,
        "tiny-negative density cleanup error"
    );

    bool rejected = false;
    try {
        (void)libxc.evaluate({-1.0e-4}, 1.0);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("A materially negative density was not rejected.");
    }
}

void test_spin_unpolarized_limit_and_derivatives() {
    const std::vector<double> total_density{
        0.02, 0.1, 0.7, 2.0
    };
    std::vector<double> density_up(total_density.size(), 0.0);
    std::vector<double> density_down(total_density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(total_density.size()); ++p) {
        density_up[p] = 0.5 * total_density[p];
        density_down[p] = 0.5 * total_density[p];
    }

    const double dV = 0.31;
    const LibXCLDAFunctional unpolarized(
        LDAFunctional::PerdewZunger, 1
    );
    const LibXCLDAFunctional polarized(
        LDAFunctional::PerdewZunger, 2
    );
    const XCResult reference =
        unpolarized.evaluate(total_density, dV);
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
        std::abs(
            spin.exchange_energy - reference.exchange_energy
        ),
        2.0e-12,
        "polarized/unpolarized exchange-energy error"
    );
    require_less(
        std::abs(
            spin.correlation_energy - reference.correlation_energy
        ),
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

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12)
                  << "LibXC runtime version = " << libxc_runtime_version() << "\n";
        test_exchange_against_analytic_oracle();
        test_pz_potential_is_energy_derivative();
        test_density_validation();
        test_spin_unpolarized_limit_and_derivatives();
    } catch (const std::exception& error) {
        std::cerr << "LibXC LDA test failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "All LibXC LDA tests passed.\n";
    return 0;
}
