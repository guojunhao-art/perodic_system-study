#include "xc_functional.hpp"

#include <xc.h>
#include <xc_funcs.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double negative_density_tolerance = 1.0e-12;

void initialize_functional(xc_func_type& functional, int identifier) {
    if (xc_func_init(&functional, identifier, XC_UNPOLARIZED) != 0) {
        throw std::runtime_error(
            "LibXC failed to initialize functional id "
            + std::to_string(identifier) + "."
        );
    }
}

void evaluate_one_functional(
    const xc_func_type& functional,
    const std::vector<double>& density,
    double dV,
    std::vector<double>& potential,
    double& energy) {

    if (density.empty()) {
        energy = 0.0;
        return;
    }

    std::vector<double> energy_per_particle(density.size(), 0.0);
    std::vector<double> local_potential(density.size(), 0.0);
    xc_lda_exc_vxc(
        &functional,
        density.size(),
        density.data(),
        energy_per_particle.data(),
        local_potential.data()
    );

    energy = 0.0;
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        energy += dV * density[p] * energy_per_particle[p];
        potential[p] += local_potential[p];
    }
}

} // namespace

struct LibXCLDAFunctional::Impl {
    xc_func_type exchange{};
    xc_func_type correlation{};
    bool exchange_initialized = false;
    bool correlation_initialized = false;

    ~Impl() {
        if (correlation_initialized) {
            xc_func_end(&correlation);
        }
        if (exchange_initialized) {
            xc_func_end(&exchange);
        }
    }
};

LibXCLDAFunctional::LibXCLDAFunctional(LDAFunctional functional)
    : impl_(std::make_unique<Impl>()) {

    initialize_functional(impl_->exchange, XC_LDA_X);
    impl_->exchange_initialized = true;

    if (functional == LDAFunctional::PerdewZunger) {
        initialize_functional(impl_->correlation, XC_LDA_C_PZ);
        impl_->correlation_initialized = true;
    } else if (functional != LDAFunctional::ExchangeOnly) {
        throw std::runtime_error("Unknown LDA functional selection.");
    }
}

LibXCLDAFunctional::~LibXCLDAFunctional() = default;

LibXCLDAFunctional::LibXCLDAFunctional(
    LibXCLDAFunctional&&) noexcept = default;

LibXCLDAFunctional& LibXCLDAFunctional::operator=(
    LibXCLDAFunctional&&) noexcept = default;

XCResult LibXCLDAFunctional::evaluate(
    const std::vector<double>& density,
    double dV) const {

    if (!std::isfinite(dV) || dV <= 0.0) {
        throw std::runtime_error("XC grid-cell volume must be positive and finite.");
    }

    std::vector<double> sanitized_density(density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        const double value = density[p];
        if (!std::isfinite(value)) {
            throw std::runtime_error("XC density contains a non-finite value.");
        }
        if (value < -negative_density_tolerance) {
            throw std::runtime_error(
                "XC density contains a physically invalid negative value."
            );
        }
        sanitized_density[p] = std::max(value, 0.0);
    }

    XCResult result;
    result.Vxc.assign(density.size(), 0.0);
    evaluate_one_functional(
        impl_->exchange,
        sanitized_density,
        dV,
        result.Vxc,
        result.exchange_energy
    );

    if (impl_->correlation_initialized) {
        evaluate_one_functional(
            impl_->correlation,
            sanitized_density,
            dV,
            result.Vxc,
            result.correlation_energy
        );
    }

    return result;
}

std::string lda_functional_name(LDAFunctional functional) {
    if (functional == LDAFunctional::ExchangeOnly) {
        return "LDA_X";
    }
    if (functional == LDAFunctional::PerdewZunger) {
        return "LDA_X + LDA_C_PZ";
    }
    throw std::runtime_error("Unknown LDA functional selection.");
}

std::string libxc_runtime_version() {
    return xc_version_string();
}
