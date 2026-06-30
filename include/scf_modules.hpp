#pragma once

#include "core.hpp"

#include <Eigen/Dense>
#include <vector>

enum class OccupationMode {
    Fixed,
    DegeneracyAwareZeroT,
    FermiDirac
};

struct OccupationResult {
    std::vector<double> occ;
    double mu = 0.0;
    double nelec_sum = 0.0;
    double entropy = 0.0;
};

struct LDAExchangeResult {
    std::vector<double> Vx;
    double Ex = 0.0;
};

struct EnergyTerms {
    double kinetic = 0.0;
    double external = 0.0;
    double hartree = 0.0;
    double exchange = 0.0;
    double total = 0.0;
    double electronic_entropy = 0.0;
    double entropy_correction = 0.0;
    double free_energy = 0.0;
    double sigma0_estimate = 0.0;
    double ion_smooth = 0.0;
    double total_with_ion_smooth = 0.0;
};

std::vector<double> build_density_from_orbitals(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    double volume);

double integrate_real_grid(const std::vector<double>& f, double dV);

double electron_number_from_density(const std::vector<double>& rho, double dV);

std::vector<double> build_hartree_potential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& rho);

LDAExchangeResult build_lda_exchange(
    const std::vector<double>& rho,
    double dV);

std::vector<double> build_toy_ionic_potential(
    const FFTGrid& grid,
    double V0);

std::vector<double> combine_effective_potential(
    const std::vector<double>& Vion,
    const std::vector<double>& VH,
    const std::vector<double>& Vx);

double sum_occupations(const std::vector<double>& occ);

OccupationResult degeneracy_aware_zeroT_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    double occ_max = 2.0,
    double deg_tol = 1.0e-8);

double fermi_dirac_occ_single(
    double eps,
    double mu,
    double sigma,
    double occ_max = 2.0);

OccupationResult fermi_dirac_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    double sigma,
    double occ_max = 2.0);

OccupationResult compute_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    OccupationMode mode,
    const std::vector<double>& fixed_occ,
    double sigma,
    double deg_tol);

double compute_kinetic_energy(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations);

EnergyTerms compute_total_energy(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    const std::vector<double>& rho,
    const std::vector<double>& Vion,
    const std::vector<double>& VH,
    double Ex,
    double dV,
    double electronic_entropy = 0.0,
    double sigma = 0.0);

double mix_density(
    std::vector<double>& rho,
    const std::vector<double>& rho_out,
    double alpha,
    double dV,
    double target_nelec);

std::vector<std::complex<double>> build_density_G(
    FFTWorkspace& fft,
    const std::vector<double>& rho);

double compute_external_energy(
    const std::vector<double>& rho,
    const std::vector<double>& V,
    double dV);