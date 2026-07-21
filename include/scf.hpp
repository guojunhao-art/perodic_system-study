#pragma once

#include "core.hpp"
#include "scf_modules.hpp"
#include "xc_functional.hpp"

#include <Eigen/Dense>

#include <iosfwd>
#include <vector>

enum class SCFVerbosity {
    Silent,
    Compact,
    Detailed
};

/*
 * Numerical controls for one self-consistent Kohn--Sham calculation.
 * Energy-like quantities and tolerances use Hartree atomic units.
 */
struct SCFOptions {
    double nelec = 2.0;
    int nbands = 10;

    LDAFunctional lda_functional = LDAFunctional::PerdewZunger;

    OccupationMode occupation_mode = OccupationMode::FermiDirac;
    std::vector<double> fixed_occupations;
    double smearing_sigma = 0.0;
    double degeneracy_tolerance = 1.0e-8;

    int max_iterations = 200;
    double density_tolerance = 1.0e-7;
    double energy_tolerance = 1.0e-9;

    int eigensolver_max_iterations = 80;
    int eigensolver_max_subspace = 0;
    double eigensolver_tolerance = 1.0e-10;
    double eigensolver_denom_floor = 1.0e-6;

    double mixing_alpha = 0.10;
    int pulay_max_history = 6;
    int pulay_min_history = 2;
    double pulay_regularization = 1.0e-12;

    SCFVerbosity verbosity = SCFVerbosity::Compact;

    /* Used only by Detailed output. */
    int bands_to_print = 8;
    int band_print_interval = 5;
};

/*
 * Both fields are optional. Empty fields select the original defaults:
 * a uniform density and low-kinetic-energy plane waves.
 */
struct SCFInitialGuess {
    std::vector<double> density;
    Eigen::MatrixXcd orbitals;
};

struct SCFResult {
    bool converged = false;
    int iterations = 0;
    double final_density_residual = 0.0;
    double final_energy_change = 0.0;

    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXcd orbitals;
    OccupationResult occupations;
    std::vector<double> density;
    EnergyTerms energy;

    double electron_number_from_density = 0.0;

    /* Aggregate eigensolver work over all SCF iterations. */
    long long eigensolver_hamiltonian_applications = 0;
    long long eigensolver_hamiltonian_block_calls = 0;
    long long eigensolver_iterations = 0;
    long long eigensolver_restarts = 0;
    double eigensolver_hamiltonian_seconds = 0.0;
    double eigensolver_subspace_seconds = 0.0;
    double wall_time_seconds = 0.0;

    /*
     * E at zero T, or the Mermin free energy at finite smearing, including
     * the supplied ion--ion energy. This is the energy to differentiate
     * in a later geometry optimization.
     */
    double variational_energy = 0.0;
};

/*
 * Solve the electronic SCF problem for a fixed ionic configuration.
 *
 * ionic_potential and ion_ion_energy are constructed outside this routine,
 * so a geometry driver can rebuild them after moving ions. Passing a null
 * log_stream makes the solver silent.
 */
SCFResult run_scf(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& ionic_potential,
    double ion_ion_energy,
    const std::vector<NonlocalProjector>& projectors,
    const SCFOptions& options,
    const SCFInitialGuess& initial_guess = {},
    std::ostream* log_stream = nullptr);
