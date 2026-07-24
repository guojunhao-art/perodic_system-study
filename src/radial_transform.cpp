#include "radial_transform.hpp"

#include <cmath>
#include <stdexcept>

namespace {

double odd_double_factorial(int n) {
    double value = 1.0;
    for (int k = 1; k <= n; k += 2) {
        value *= static_cast<double>(k);
    }
    return value;
}

double small_x_spherical_bessel(int l, double x) {
    const double x2 = x * x;
    const double leading =
        std::pow(x, l) / odd_double_factorial(2 * l + 1);
    const double correction2 =
        -x2 / (2.0 * static_cast<double>(2 * l + 3));
    const double correction4 =
        x2 * x2 /
        (8.0 * static_cast<double>((2 * l + 3) * (2 * l + 5)));
    const double correction6 =
        -x2 * x2 * x2 /
        (48.0 * static_cast<double>(
            (2 * l + 3) * (2 * l + 5) * (2 * l + 7)
        ));
    return leading * (1.0 + correction2 + correction4 + correction6);
}

void validate_radial_data(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& values) {

    if (l < 0) {
        throw std::runtime_error("Angular momentum l cannot be negative.");
    }
    if (g < 0.0) {
        throw std::runtime_error("Reciprocal radius g cannot be negative.");
    }
    if (radial_grid.empty()) {
        throw std::runtime_error("Radial grid cannot be empty.");
    }
    if (radial_grid.size() != quadrature_weights.size() ||
        radial_grid.size() != values.size()) {
        throw std::runtime_error("Radial arrays must have identical sizes.");
    }
    for (int i = 0; i < static_cast<int>(radial_grid.size()); ++i) {
        if (radial_grid[i] < 0.0) {
            throw std::runtime_error("Radial grid values cannot be negative.");
        }
        if (i > 0 && radial_grid[i] <= radial_grid[i - 1]) {
            throw std::runtime_error("Radial grid must be strictly increasing.");
        }
    }
}

template <typename RadialFactor>
double transform_impl_unchecked(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& values,
    const RadialFactor& radial_factor) {

    double integral = 0.0;
    for (int i = 0; i < static_cast<int>(radial_grid.size()); ++i) {
        const double r = radial_grid[i];
        integral += quadrature_weights[i]
            * radial_factor(r)
            * spherical_bessel_j(l, g * r)
            * values[i];
    }

    return 4.0 * M_PI * integral;
}

} // namespace

std::vector<double> make_upf_simpson_weights(
    const std::vector<double>& rab) {

    if (rab.size() < 3 || rab.size() % 2 == 0) {
        throw std::runtime_error(
            "UPF Simpson quadrature requires an odd mesh with at least 3 points."
        );
    }

    std::vector<double> weights(rab.size(), 0.0);
    for (int i = 0; i < static_cast<int>(rab.size()); ++i) {
        if (!std::isfinite(rab[i]) || rab[i] <= 0.0) {
            throw std::runtime_error(
                "UPF PP_RAB factors must be positive and finite."
            );
        }
        const double simpson_factor = (i == 0 || i + 1 == static_cast<int>(rab.size()))
            ? 1.0
            : (i % 2 == 0 ? 2.0 : 4.0);
        weights[i] = simpson_factor * rab[i] / 3.0;
    }
    return weights;
}

double spherical_bessel_j(int l, double x) {
    if (l < 0) {
        throw std::runtime_error("Angular momentum l cannot be negative.");
    }

    const double ax = std::abs(x);
    if (ax <= 0.10) {
        return small_x_spherical_bessel(l, x);
    }

    const double j0 = std::sin(x) / x;
    if (l == 0) {
        return j0;
    }

    const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
    if (l == 1) {
        return j1;
    }

    double j_previous = j0;
    double j_current = j1;
    for (int order = 1; order < l; ++order) {
        const double j_next =
            (2.0 * static_cast<double>(order) + 1.0) * j_current / x
            - j_previous;
        j_previous = j_current;
        j_current = j_next;
    }

    return j_current;
}

double radial_fourier_bessel_transform(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& radial_function) {

    validate_radial_data(
        l,
        g,
        radial_grid,
        quadrature_weights,
        radial_function
    );
    return radial_fourier_bessel_transform_unchecked(
        l,
        g,
        radial_grid,
        quadrature_weights,
        radial_function
    );
}

double radial_fourier_bessel_transform_unchecked(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& radial_function) {

    return transform_impl_unchecked(
        l,
        g,
        radial_grid,
        quadrature_weights,
        radial_function,
        [](double r) { return r * r; }
    );
}

double radial_fourier_bessel_transform_from_r_times_function(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& r_times_radial_function) {

    validate_radial_data(
        l,
        g,
        radial_grid,
        quadrature_weights,
        r_times_radial_function
    );
    return radial_fourier_bessel_transform_from_r_times_function_unchecked(
        l,
        g,
        radial_grid,
        quadrature_weights,
        r_times_radial_function
    );
}

double radial_fourier_bessel_transform_from_r_times_function_unchecked(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& r_times_radial_function) {

    return transform_impl_unchecked(
        l,
        g,
        radial_grid,
        quadrature_weights,
        r_times_radial_function,
        [](double r) { return r; }
    );
}
