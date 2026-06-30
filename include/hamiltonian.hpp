#pragma once

#include "core.hpp"

#include <Eigen/Dense>
#include <complex>
#include <vector>

void scatter_coeffs_to_fft_grid(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& coeffs,
    std::vector<std::complex<double>>& reciprocal_grid);

void gather_coeffs_from_fft_grid(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& reciprocal_grid,
    std::vector<std::complex<double>>& coeffs_out);

std::vector<std::complex<double>> apply_local_potential_fft(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const std::vector<std::complex<double>>& coeffs);

std::vector<std::complex<double>> apply_hamiltonian_fft(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const std::vector<std::complex<double>>& coeffs);

std::vector<std::complex<double>> forward_fft_real_potential(
    FFTWorkspace& fft,
    const std::vector<double>& V_r);

bool fft_grid_can_represent_freq(
    const Eigen::Vector3i& n,
    const FFTGrid& grid);

Eigen::MatrixXcd build_explicit_hamiltonian(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r);

double hermiticity_error(const Eigen::MatrixXcd& H);

std::vector<std::complex<double>> eigen_to_std_vector(
    const Eigen::VectorXcd& x);

Eigen::VectorXcd std_to_eigen_vector(
    const std::vector<std::complex<double>>& x);

Eigen::VectorXcd apply_hamiltonian_eigen(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::VectorXcd& c,
    const std::vector<NonlocalProjector>* projectors = nullptr);

Eigen::MatrixXcd apply_hamiltonian_to_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& C,
    const std::vector<NonlocalProjector>* projectors = nullptr);
