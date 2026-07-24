#include "xc_functional.hpp"

#include <xc.h>
#include <xc_funcs.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double negative_density_tolerance = 1.0e-12;

void initialize_functional(
    xc_func_type& functional,
    int identifier,
    int polarization) {

    if (xc_func_init(&functional, identifier, polarization) != 0) {
        throw std::runtime_error(
            "LibXC failed to initialize functional id "
            + std::to_string(identifier) + "."
        );
    }
}

std::vector<double> sanitize_density(
    const std::vector<double>& density) {

    std::vector<double> sanitized(density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        const double value = density[p];
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "XC density contains a non-finite value."
            );
        }
        if (value < -negative_density_tolerance) {
            throw std::runtime_error(
                "XC density contains a physically invalid negative value."
            );
        }
        sanitized[p] = std::max(value, 0.0);
    }
    return sanitized;
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

void evaluate_one_spin_functional(
    const xc_func_type& functional,
    const std::vector<double>& interleaved_density,
    const std::vector<double>& total_density,
    double dV,
    std::vector<double>& potential_up,
    std::vector<double>& potential_down,
    double& energy) {

    const int point_count = static_cast<int>(total_density.size());
    if (point_count == 0) {
        energy = 0.0;
        return;
    }

    std::vector<double> energy_per_particle(point_count, 0.0);
    std::vector<double> local_potential(2 * point_count, 0.0);
    xc_lda_exc_vxc(
        &functional,
        point_count,
        interleaved_density.data(),
        energy_per_particle.data(),
        local_potential.data()
    );

    energy = 0.0;
    for (int p = 0; p < point_count; ++p) {
        energy += dV * total_density[p] * energy_per_particle[p];
        potential_up[p] += local_potential[2 * p];
        potential_down[p] += local_potential[2 * p + 1];
    }
}

} // namespace

struct LibXCLDAFunctional::Impl {
    xc_func_type exchange{};
    xc_func_type correlation{};
    bool exchange_initialized = false;
    bool correlation_initialized = false;
    int nspin = 1;

    ~Impl() {
        if (correlation_initialized) {
            xc_func_end(&correlation);
        }
        if (exchange_initialized) {
            xc_func_end(&exchange);
        }
    }
};

LibXCLDAFunctional::LibXCLDAFunctional(
    LDAFunctional functional,
    int nspin)
    : impl_(std::make_unique<Impl>()) {

    if (nspin != 1 && nspin != 2) {
        throw std::runtime_error(
            "LibXC LDA supports nspin = 1 or 2."
        );
    }
    impl_->nspin = nspin;
    const int polarization =
        nspin == 1 ? XC_UNPOLARIZED : XC_POLARIZED;
    initialize_functional(
        impl_->exchange, XC_LDA_X, polarization
    );
    impl_->exchange_initialized = true;

    if (functional == LDAFunctional::PerdewZunger) {
        initialize_functional(
            impl_->correlation, XC_LDA_C_PZ, polarization
        );
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

    if (impl_->nspin != 1) {
        throw std::runtime_error(
            "Use evaluate_spin() for a spin-polarized LibXC functional."
        );
    }
    if (!std::isfinite(dV) || dV <= 0.0) {
        throw std::runtime_error("XC grid-cell volume must be positive and finite.");
    }

    const std::vector<double> sanitized_density =
        sanitize_density(density);

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

SpinXCResult LibXCLDAFunctional::evaluate_spin(
    const std::vector<double>& density_up,
    const std::vector<double>& density_down,
    double dV) const {

    if (impl_->nspin != 2) {
        throw std::runtime_error(
            "Use evaluate() for an unpolarized LibXC functional."
        );
    }
    if (!std::isfinite(dV) || dV <= 0.0) {
        throw std::runtime_error(
            "XC grid-cell volume must be positive and finite."
        );
    }
    if (density_up.size() != density_down.size()) {
        throw std::runtime_error(
            "Spin-density grids must have the same size."
        );
    }

    const std::vector<double> sanitized_up =
        sanitize_density(density_up);
    const std::vector<double> sanitized_down =
        sanitize_density(density_down);
    const int point_count = static_cast<int>(density_up.size());
    std::vector<double> interleaved_density(2 * point_count, 0.0);
    std::vector<double> total_density(point_count, 0.0);
    for (int p = 0; p < point_count; ++p) {
        interleaved_density[2 * p] = sanitized_up[p];
        interleaved_density[2 * p + 1] = sanitized_down[p];
        total_density[p] = sanitized_up[p] + sanitized_down[p];
    }

    SpinXCResult result;
    result.Vxc_up.assign(point_count, 0.0);
    result.Vxc_down.assign(point_count, 0.0);
    evaluate_one_spin_functional(
        impl_->exchange,
        interleaved_density,
        total_density,
        dV,
        result.Vxc_up,
        result.Vxc_down,
        result.exchange_energy
    );
    if (impl_->correlation_initialized) {
        evaluate_one_spin_functional(
            impl_->correlation,
            interleaved_density,
            total_density,
            dV,
            result.Vxc_up,
            result.Vxc_down,
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
