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

std::vector<double> davidson_band_residual_tolerances(
    int nbands,
    double strict_tolerance,
    const std::vector<double>& previous_occupations) {

    if (nbands <= 0) {
        throw std::runtime_error(
            "Davidson band-residual policy requires a positive band count."
        );
    }
    if (!std::isfinite(strict_tolerance) || strict_tolerance <= 0.0) {
        throw std::runtime_error(
            "Strict Davidson residual tolerance must be positive and finite."
        );
    }
    if (!previous_occupations.empty() &&
        static_cast<int>(previous_occupations.size()) != nbands) {
        throw std::runtime_error(
            "Previous occupation count must match the Davidson band count."
        );
    }

    std::vector<double> tolerances(nbands, strict_tolerance);
    if (previous_occupations.empty()) {
        return tolerances;
    }

    const double empty_tolerance = std::max(
        DAVIDSON_EMPTY_TOLERANCE_FACTOR * strict_tolerance,
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA
    );
    for (int band = 0; band < nbands; ++band) {
        const double occupation = previous_occupations[band];
        if (!std::isfinite(occupation) || occupation < 0.0) {
            throw std::runtime_error(
                "Previous band occupations must be finite and non-negative."
            );
        }
        if (occupation < DAVIDSON_EMPTY_OCCUPATION_THRESHOLD) {
            tolerances[band] = empty_tolerance;
        }
    }
    return tolerances;
}

bool davidson_occupation_refinement_needed(
    const std::vector<double>& previous_occupations,
    const std::vector<double>& current_occupations) {

    if (previous_occupations.empty()) {
        return false;
    }
    if (previous_occupations.size() != current_occupations.size()) {
        throw std::runtime_error(
            "Previous and current Davidson occupations must have the same "
            "size."
        );
    }
    for (int band = 0;
         band < static_cast<int>(current_occupations.size());
         ++band) {
        const double previous = previous_occupations[band];
        const double current = current_occupations[band];
        if (!std::isfinite(previous) || previous < 0.0 ||
            !std::isfinite(current) || current < 0.0) {
            throw std::runtime_error(
                "Davidson occupations must be finite and non-negative."
            );
        }
        if (previous < DAVIDSON_EMPTY_OCCUPATION_THRESHOLD &&
            current >= DAVIDSON_EMPTY_OCCUPATION_THRESHOLD) {
            return true;
        }
    }
    return false;
}

DavidsonResidualAssessment assess_davidson_residuals(
    const std::vector<double>& residual_norms,
    const std::vector<double>& band_tolerances,
    double strict_tolerance) {

    if (residual_norms.empty() ||
        residual_norms.size() != band_tolerances.size()) {
        throw std::runtime_error(
            "Davidson residuals and band tolerances must have the same "
            "nonzero size."
        );
    }
    if (!std::isfinite(strict_tolerance) || strict_tolerance <= 0.0) {
        throw std::runtime_error(
            "Strict Davidson residual tolerance must be positive and finite."
        );
    }

    DavidsonResidualAssessment assessment;
    assessment.acceptable = true;
    assessment.strict_bands_converged = true;
    const int maximum_relaxed_failures = std::max(
        1,
        std::min(
            5,
            static_cast<int>(residual_norms.size()) / 4
        )
    );

    for (int band = 0;
         band < static_cast<int>(residual_norms.size());
         ++band) {
        const double residual = residual_norms[band];
        const double tolerance = band_tolerances[band];
        if (!std::isfinite(residual) || residual < 0.0 ||
            !std::isfinite(tolerance) || tolerance <= 0.0) {
            throw std::runtime_error(
                "Davidson residuals and tolerances must be finite and "
                "non-negative/positive."
            );
        }
        if (residual <= tolerance) {
            continue;
        }

        const bool strict_band =
            tolerance <= strict_tolerance;
        if (strict_band) {
            assessment.acceptable = false;
            assessment.strict_bands_converged = false;
            if (residual > assessment.maximum_strict_residual) {
                assessment.maximum_strict_residual = residual;
                assessment.worst_strict_band = band;
            }
            continue;
        }

        ++assessment.relaxed_band_failures;
        if (residual > assessment.maximum_relaxed_residual) {
            assessment.maximum_relaxed_residual = residual;
            assessment.worst_relaxed_band = band;
            assessment.maximum_accepted_relaxed_residual =
                DAVIDSON_EMPTY_FAILURE_GRACE_FACTOR * tolerance;
        }
        if (residual >
            DAVIDSON_EMPTY_FAILURE_GRACE_FACTOR * tolerance) {
            assessment.acceptable = false;
        }
    }

    if (assessment.relaxed_band_failures >
        maximum_relaxed_failures) {
        assessment.acceptable = false;
    }
    return assessment;
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
