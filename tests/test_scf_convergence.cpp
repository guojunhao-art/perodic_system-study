#include "scf_convergence.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& label) {

    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + ": actual = " + std::to_string(actual)
            + ", expected = " + std::to_string(expected)
        );
    }
}

void test_band_tolerances() {
    const double strict_tolerance = 1.0e-9;
    const std::vector<double> occupations{2.0, 0.4, 0.009, 0.0};
    const std::vector<double> tolerances =
        davidson_band_residual_tolerances(
            static_cast<int>(occupations.size()),
            strict_tolerance,
            occupations
        );

    require_close(tolerances[0], strict_tolerance, 0.0,
                  "Occupied-band Davidson tolerance");
    require_close(tolerances[1], strict_tolerance, 0.0,
                  "Partially occupied-band Davidson tolerance");
    require_close(
        tolerances[2],
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA,
        0.0,
        "Effectively empty-band Davidson tolerance"
    );
    require_close(
        tolerances[3],
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA,
        0.0,
        "Empty-band Davidson tolerance"
    );

    const std::vector<double> first_iteration =
        davidson_band_residual_tolerances(
            4,
            strict_tolerance
        );
    for (double tolerance : first_iteration) {
        require_close(
            tolerance,
            strict_tolerance,
            0.0,
            "First-iteration Davidson tolerance"
        );
    }

    const std::vector<double> loose_strict =
        davidson_band_residual_tolerances(
            2,
            1.0e-6,
            {2.0, 0.0}
        );
    require_close(loose_strict[1], 5.0e-6, 1.0e-20,
                  "Scaled empty-band Davidson tolerance");

    if (!davidson_occupation_refinement_needed(
            {2.0, 0.0},
            {2.0, 0.02})) {
        throw std::runtime_error(
            "A newly occupied band did not request strict refinement."
        );
    }
    if (davidson_occupation_refinement_needed(
            {2.0, 0.02},
            {2.0, 0.0})) {
        throw std::runtime_error(
            "A newly empty band incorrectly requested strict refinement."
        );
    }
}

void test_residual_assessment() {
    const double strict_tolerance = 1.0e-9;
    const std::vector<double> tolerances{
        strict_tolerance,
        strict_tolerance,
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA,
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA
    };

    DavidsonResidualAssessment assessment =
        assess_davidson_residuals(
            {5.0e-10, 8.0e-10, 2.0e-7, 8.0e-8},
            tolerances,
            strict_tolerance
        );
    if (!assessment.acceptable ||
        !assessment.strict_bands_converged ||
        assessment.relaxed_band_failures != 1) {
        throw std::runtime_error(
            "A small residual miss confined to one empty band was rejected."
        );
    }

    assessment = assess_davidson_residuals(
        {2.0e-9, 8.0e-10, 8.0e-8, 8.0e-8},
        tolerances,
        strict_tolerance
    );
    if (assessment.acceptable ||
        assessment.strict_bands_converged ||
        assessment.worst_strict_band != 0) {
        throw std::runtime_error(
            "An unconverged occupied Davidson band was accepted."
        );
    }

    assessment = assess_davidson_residuals(
        {5.0e-10, 8.0e-10, 6.0e-7, 8.0e-8},
        tolerances,
        strict_tolerance
    );
    if (assessment.acceptable) {
        throw std::runtime_error(
            "A materially unconverged empty Davidson band was accepted."
        );
    }

    const std::vector<double> many_empty_tolerances(
        24,
        DAVIDSON_EMPTY_TOLERANCE_FLOOR_HA
    );
    const std::vector<double> many_empty_residuals(24, 2.0e-7);
    assessment = assess_davidson_residuals(
        many_empty_residuals,
        many_empty_tolerances,
        strict_tolerance
    );
    if (assessment.acceptable) {
        throw std::runtime_error(
            "Too many unconverged empty Davidson bands were accepted."
        );
    }
}

} // namespace

int main() {
    try {
        test_band_tolerances();
        test_residual_assessment();
        std::cout << "SCF convergence-policy tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SCF convergence-policy test failed: "
                  << error.what() << "\n";
        return 1;
    }
}
