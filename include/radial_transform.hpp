#pragma once

#include <vector>

/*
 * Convert UPF's raw PP_RAB factors into complete Simpson quadrature weights.
 * For an odd mesh, w_i = c_i PP_RAB_i / 3 with c = 1,4,2,...,4,1.
 */
std::vector<double> make_upf_simpson_weights(
    const std::vector<double>& rab);

/*
 * Spherical Bessel function j_l(x) for non-negative angular momentum l.
 * The implementation uses explicit small-x series to avoid cancellation.
 */
double spherical_bessel_j(int l, double x);

/*
 * Return
 *
 *   4 pi integral r^2 j_l(g r) f_l(r) dr.
 *
 * quadrature_weights must already contain the complete radial integration
 * weights, i.e. integral h(r) dr ~= sum_i w_i h(r_i).
 */
double radial_fourier_bessel_transform(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& radial_function);

/*
 * UPF stores u_l(r) = r f_l(r), rather than f_l(r). This equivalent form
 * evaluates
 *
 *   4 pi integral r j_l(g r) u_l(r) dr
 *
 * without dividing by r at the origin.
 */
double radial_fourier_bessel_transform_from_r_times_function(
    int l,
    double g,
    const std::vector<double>& radial_grid,
    const std::vector<double>& quadrature_weights,
    const std::vector<double>& r_times_radial_function);
