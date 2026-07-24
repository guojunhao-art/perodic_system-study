#pragma once

/*
 * VASP-style outer SCF convergence: both the variational-energy change and
 * occupied band-energy change must be below the same energy threshold.
 */
bool scf_energy_changes_converged(
    double total_energy_change,
    double band_energy_change,
    double energy_tolerance) noexcept;

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
