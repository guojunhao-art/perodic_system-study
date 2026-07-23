#include "scf_convergence.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

bool scf_energy_changes_converged(
    double total_energy_change,
    double band_energy_change,
    double energy_tolerance) noexcept {

    return std::isfinite(total_energy_change)
        && std::isfinite(band_energy_change)
        && std::isfinite(energy_tolerance)
        && energy_tolerance > 0.0
        && std::abs(total_energy_change) < energy_tolerance
        && std::abs(band_energy_change) < energy_tolerance;
}

EigensolverToleranceSchedule::EigensolverToleranceSchedule(
    double initial_tolerance,
    double final_tolerance,
    double density_tolerance)
    : initial_tolerance_(initial_tolerance),
      final_tolerance_(final_tolerance),
      density_tolerance_(density_tolerance),
      current_tolerance_(initial_tolerance) {

    if (!std::isfinite(initial_tolerance_) || initial_tolerance_ <= 0.0 ||
        !std::isfinite(final_tolerance_) || final_tolerance_ <= 0.0 ||
        !std::isfinite(density_tolerance_) || density_tolerance_ <= 0.0) {
        throw std::runtime_error(
            "Eigensolver and density tolerances must be finite and positive."
        );
    }
    if (initial_tolerance_ < final_tolerance_) {
        throw std::runtime_error(
            "Initial eigensolver tolerance must be no tighter than the final tolerance."
        );
    }
}

double EigensolverToleranceSchedule::current() const noexcept {
    return current_tolerance_;
}

bool EigensolverToleranceSchedule::at_final_tolerance() const noexcept {
    return current_tolerance_ <= final_tolerance_;
}

void EigensolverToleranceSchedule::advance(double density_residual) {
    if (!std::isfinite(density_residual) || density_residual < 0.0) {
        throw std::runtime_error(
            "SCF density residual must be finite and non-negative."
        );
    }

    const double desired = std::clamp(
        final_tolerance_
            * std::sqrt(density_residual / density_tolerance_),
        final_tolerance_,
        initial_tolerance_
    );
    const double one_decade_tighter = std::max(
        final_tolerance_, current_tolerance_ / 10.0
    );

    /* Never loosen, and normally tighten by at most one decade per step. */
    current_tolerance_ = std::min(
        current_tolerance_,
        std::max(desired, one_decade_tighter)
    );
}

void EigensolverToleranceSchedule::force_final_tolerance() noexcept {
    current_tolerance_ = final_tolerance_;
}
