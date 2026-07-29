#pragma once

#include <vector>

/*
 * VASP-style outer SCF convergence: both the variational-energy change and
 * occupied band-energy change must be below the same energy threshold.
 */
bool scf_energy_changes_converged(
    double total_energy_change,
    double band_energy_change,
    double energy_tolerance) noexcept;

/*
 * Residual convergence remains strict for occupied and partially occupied
 * states. Once occupations from a preceding SCF iteration are available,
 * effectively empty bands may use a looser residual target because they do
 * not materially contribute to the density, total energy, or forces.
 */
constexpr double DAVIDSON_EMPTY_OCCUPATION_THRESHOLD = 1.0e-2;
constexpr double DAVIDSON_EMPTY_TOLERANCE_FACTOR = 5.0;
constexpr double DAVIDSON_DEFAULT_EMPTY_TOLERANCE_HA = 1.0e-6;
constexpr double DAVIDSON_EMPTY_FAILURE_GRACE_FACTOR = 5.0;

std::vector<double> davidson_band_residual_tolerances(
    int nbands,
    double strict_tolerance,
    const std::vector<double>& previous_occupations = {},
    double empty_tolerance = DAVIDSON_DEFAULT_EMPTY_TOLERANCE_HA);

/*
 * If a band classified as empty from the preceding iteration becomes
 * occupied now, require another SCF iteration so it is solved with the
 * strict residual target before convergence can be accepted.
 */
bool davidson_occupation_refinement_needed(
    const std::vector<double>& previous_occupations,
    const std::vector<double>& current_occupations);

struct DavidsonResidualAssessment {
    bool acceptable = false;
    bool strict_bands_converged = false;
    int worst_strict_band = -1;
    double maximum_strict_residual = 0.0;
    int relaxed_band_failures = 0;
    int worst_relaxed_band = -1;
    double maximum_relaxed_residual = 0.0;
    double maximum_accepted_relaxed_residual = 0.0;
};

/*
 * A result above the relaxed target may still be retained if every strict
 * band converged, only a bounded number of empty bands missed their targets,
 * and each remains within a narrow numerical grace margin.
 */
DavidsonResidualAssessment assess_davidson_residuals(
    const std::vector<double>& residual_norms,
    const std::vector<double>& band_tolerances,
    double strict_tolerance);

/*
 * Match the accuracy of the inner Davidson solve to the current accuracy of
 * the outer SCF iteration. Tolerances use Hartree atomic units.
 */
class EigensolverToleranceSchedule {
public:
    EigensolverToleranceSchedule(
        double initial_tolerance,
        double final_tolerance,
        double density_tolerance);

    double current() const noexcept;
    bool at_final_tolerance() const noexcept;

    /* Select the tolerance for the next SCF iteration. */
    void advance(double density_residual);

    /* Request one fully converged eigensolve before accepting SCF convergence. */
    void force_final_tolerance() noexcept;

private:
    double initial_tolerance_ = 0.0;
    double final_tolerance_ = 0.0;
    double density_tolerance_ = 0.0;
    double current_tolerance_ = 0.0;
};
