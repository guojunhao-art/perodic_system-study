#include "radial_transform.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct UniformSimpsonGrid {
    std::vector<double> r;
    std::vector<double> weights;
};

UniformSimpsonGrid make_uniform_simpson_grid(double rmax, int npoints) {
    if (npoints < 3 || npoints % 2 == 0) {
        throw std::runtime_error("Simpson grid needs an odd number of points.");
    }

    UniformSimpsonGrid grid;
    grid.r.resize(npoints);
    grid.weights.resize(npoints);
    const double h = rmax / static_cast<double>(npoints - 1);

    for (int i = 0; i < npoints; ++i) {
        grid.r[i] = h * static_cast<double>(i);
        const double simpson_factor = (i == 0 || i == npoints - 1)
            ? 1.0
            : (i % 2 == 0 ? 2.0 : 4.0);
        grid.weights[i] = h * simpson_factor / 3.0;
    }

    return grid;
}

void require_less(double value, double tolerance, const std::string& label) {
    std::cout << label << " = " << value
              << "  tolerance = " << tolerance << "\n";
    if (!(value < tolerance)) {
        throw std::runtime_error(label + " exceeds tolerance.");
    }
}

double gaussian_transform_3d(double alpha, double g) {
    return std::pow(M_PI / alpha, 1.5)
        * std::exp(-g * g / (4.0 * alpha));
}

void test_spherical_bessel_values() {
    double max_error = 0.0;
    const std::vector<double> x_values{
        0.0, 1.0e-8, 0.03, 0.10, 0.24, 0.30, 1.7, 5.2
    };

    for (int l = 0; l <= 4; ++l) {
        for (double x : x_values) {
            const double expected = x == 0.0
                ? (l == 0 ? 1.0 : 0.0)
                : std::sph_bessel(static_cast<unsigned int>(l), x);
            max_error = std::max(
                max_error,
                std::abs(spherical_bessel_j(l, x) - expected)
            );

            const double parity = l % 2 == 0 ? 1.0 : -1.0;
            max_error = std::max(
                max_error,
                std::abs(spherical_bessel_j(l, -x) - parity * expected)
            );
        }
    }

    require_less(max_error, 2.0e-10, "spherical-Bessel value error");
}

void test_s_gaussian_transform() {
    const double alpha = 0.70;
    const auto grid = make_uniform_simpson_grid(10.0, 12001);
    std::vector<double> gaussian(grid.r.size());
    std::vector<double> r_times_gaussian(grid.r.size());

    for (int i = 0; i < static_cast<int>(grid.r.size()); ++i) {
        gaussian[i] = std::exp(-alpha * grid.r[i] * grid.r[i]);
        r_times_gaussian[i] = grid.r[i] * gaussian[i];
    }

    double max_error = 0.0;
    double max_upf_form_error = 0.0;
    for (double g : std::vector<double>{0.0, 0.2, 1.0, 3.0, 6.0}) {
        const double expected = gaussian_transform_3d(alpha, g);
        const double transformed = radial_fourier_bessel_transform(
            0, g, grid.r, grid.weights, gaussian
        );
        const double transformed_upf_form =
            radial_fourier_bessel_transform_from_r_times_function(
                0, g, grid.r, grid.weights, r_times_gaussian
            );
        max_error = std::max(max_error, std::abs(transformed - expected));
        max_upf_form_error = std::max(
            max_upf_form_error,
            std::abs(transformed_upf_form - expected)
        );
    }

    require_less(max_error, 2.0e-11, "s-Gaussian transform error");
    require_less(
        max_upf_form_error,
        2.0e-11,
        "UPF r-times-function transform error"
    );
}

void test_p_gaussian_transform() {
    const double alpha = 0.70;
    const auto grid = make_uniform_simpson_grid(10.0, 12001);
    std::vector<double> p_radial(grid.r.size());

    for (int i = 0; i < static_cast<int>(grid.r.size()); ++i) {
        const double r = grid.r[i];
        p_radial[i] = r * std::exp(-alpha * r * r);
    }

    double max_error = 0.0;
    for (double g : std::vector<double>{0.0, 0.2, 1.0, 3.0, 6.0}) {
        const double expected =
            g * gaussian_transform_3d(alpha, g) / (2.0 * alpha);
        const double transformed = radial_fourier_bessel_transform(
            1, g, grid.r, grid.weights, p_radial
        );
        max_error = std::max(max_error, std::abs(transformed - expected));
    }

    require_less(max_error, 2.0e-11, "p-Gaussian transform error");
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_spherical_bessel_values();
        test_s_gaussian_transform();
        test_p_gaussian_transform();
    } catch (const std::exception& error) {
        std::cerr << "Radial-transform test failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "All radial Fourier-Bessel tests passed.\n";
    return 0;
}
