#include "core.hpp"
#include "eigensolver.hpp"
#include "mixing.hpp"
#include "scf_modules.hpp"
#include "potentials.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    
    double nelec = 2.0;
    if (argc > 1) {
        nelec = std::atol(argv[1]); //电子数      
    }
    const int nbands_solve = std::max(3 * nelec, 10.0);

    OccupationMode occ_mode = OccupationMode::FermiDirac;//
    if (argc > 2) {
        if(std::atoi(argv[2]) == 1) occ_mode = OccupationMode::FermiDirac;
        else occ_mode = OccupationMode::DegeneracyAwareZeroT;        
    }

    double sigma_eV = 0.05;///
    if (argc > 3) {
        sigma_eV = std::atof(argv[3]); //sigma_ev
    }
    std::cout << std::setprecision(16);

    const double a = 10.0;

    Lattice lattice(
        Eigen::Vector3d(a, 0.0, 0.0),
        Eigen::Vector3d(0.0, a, 0.0),
        Eigen::Vector3d(0.0, 0.0, a)
    );

    FFTGrid grid(32, 32, 32);
    FFTWorkspace fft(grid);

    Eigen::Vector3d k_cart(0.0, 0.0, 0.0);

    double ecut = 3.0;
    if (argc > 4) {
        ecut = std::atof(argv[4]); //截断能
    }

    PlaneWaveBasis3D basis;
    basis.generate(lattice, k_cart, ecut);

    std::cout << "Basis size = " << basis.size() << "\n";

    const double volume = lattice.volume();
    const double dV =
        volume / static_cast<double>(grid.ngrid);

    const double deg_tol = 1.0e-8;

    /*
     * Fermi-Dirac sigma.
     * Only used if occ_mode == FermiDirac.
     */
    const double sigma_Ha = sigma_eV / 27.211386245988;

    std::vector<double> fixed_occ(nbands_solve, 0.0);
    /*
    * External potential.
    * Negative V0 attracts electrons.
    */
    std::vector<Ion> ions;

    Ion ion;
    ion.frac_position = Eigen::Vector3d(0.5, 0.5, 0.5);
    ion.Z = nelec;
    ion.sigma = 0.7;

    ion.short_amp = 1.0;
    ion.short_rc = 0.5;
    ion.beta_D = 1;
    if (argc > 5) {
        ion.sigma = std::atof(argv[5]); //平滑因子
    }
    ions.push_back(ion);

    LocalPotentialComponents local_pot =
        build_local_pseudopotential_components(
            lattice,
            fft,
            ions
        );

    auto projectors = build_s_gaussian_projectors(lattice, basis, ions);
    std::vector<double> Vion = local_pot.Vtotal;

    const double Eion_smooth = local_pot.Eion_smooth; 

    std::cout << "Smooth ion-ion energy = "
              << Eion_smooth << "\n";

 

    /*
     * Initial density: uniform.
     */
    std::vector<double> rho(
        grid.ngrid,
        nelec / volume
    );

    /*
     * Initial Davidson trial vectors.
     */
    const int ntrial_scf =
        std::min(basis.size(), nbands_solve + 4);
    
    Eigen::MatrixXcd C_guess =
        initial_low_kinetic_trials(
            basis.size(),
            ntrial_scf
        );

    const int scf_max_iter = 200;
    const double scf_rho_tol = 1.0e-7;
    const double eig_tol = 1.0e-10;

    double E_old = 0.0;

    PulayMixer pulay;
    pulay.alpha = 0.10;
    pulay.max_history = 6;
    pulay.min_history = 2;
    pulay.regularization = 1.0e-12;

    for (int scf_iter = 0; scf_iter < scf_max_iter; ++scf_iter) {
        /*
         * 1. Build potentials from input rho.
         */
        std::vector<double> VH =
            build_hartree_potential(
            lattice,
            fft,
            rho
            );

        LDAExchangeResult xres =
            build_lda_exchange(
                rho,
                dV
            );

        std::vector<double> Veff =
            combine_effective_potential(
                Vion,
                VH,
                xres.Vx
            );

        /*
         * 2. Solve Kohn-Sham equations for current Veff.
         * First version: Davidson.
         */
        DavidsonResult ks =
            davidson_lowest_eigenstates(
                basis,
                fft,
                Veff,
                nbands_solve,
                C_guess,
                80,
                std::min(basis.size(), 4 * nbands_solve + 8),
                eig_tol,
                1.0e-6,
                &projectors
            );

        

        /*
         * Compute occupations from eigenvalues.
         */
        OccupationResult occ =
            compute_occupations(
                ks.eigenvalues,
                nelec,
                occ_mode,
                fixed_occ,
                sigma_Ha,
                deg_tol
            );


        /*
         * 3. Build output density from occupied orbitals.
         */
        std::vector<double> rho_out =
            build_density_from_orbitals(
                basis,
                fft,
                ks.eigenvectors.leftCols(nbands_solve),
                occ.occ,
                volume
            );


        const double nelec_out =
            electron_number_from_density(rho_out, dV);

        const double nelec_occ = occ.nelec_sum;

        /*
         * 4. For energy, recompute VH and Ex using rho_out.
         * This makes energy more consistent with output density.
         */
        std::vector<double> VH_out =
            build_hartree_potential(
                lattice,
                fft,
                rho_out
            );

        LDAExchangeResult xout =
            build_lda_exchange(
                rho_out,
                dV
            );

        const double entropy_for_energy =
            (occ_mode == OccupationMode::FermiDirac)
            ? occ.entropy
            : 0.0;

        const double sigma_for_energy =
            (occ_mode == OccupationMode::FermiDirac)
            ? sigma_Ha
            : 0.0;

        EnergyTerms E =
            compute_total_energy(
                basis,
                ks.eigenvectors.leftCols(nbands_solve),
                occ.occ,
                rho_out,
                Vion,
                VH_out,
                xout.Ex,
                dV,
                entropy_for_energy,
                sigma_for_energy
            );
        E.ion_smooth = Eion_smooth;
        E.total_with_ion_smooth = E.total + Eion_smooth;

        std::cout << "E_electronic = " << E.total
              << "  E_ion_smooth = " << E.ion_smooth
              << "  E_total_smooth = " << E.total_with_ion_smooth
              << "\n";

        /*
         * 5. Mix density.
         */

        const double drho =
            pulay_mix_density(
                pulay,
                rho,
                rho_out,
                dV,
                nelec
            );

        const double energy_for_convergence =
            (occ_mode == OccupationMode::FermiDirac)
            ? E.free_energy
            : E.total;

        const double dE =
            (scf_iter == 0)
            ? 0.0
            : std::abs(energy_for_convergence - E_old);

        std::cout << "SCF iter "
                  << std::setw(3) << scf_iter + 1
                  << "  E = " << std::setw(20) << E.total
                  << "  F = " << std::setw(20) << E.free_energy
                  << "  E0est = " << std::setw(20) << E.sigma0_estimate
                  << "  sigmaS = " << std::setw(12) << E.entropy_correction  
                  << "  dE = " << std::setw(12) << dE
                  << "  drho = " << std::setw(12) << drho
                  <<  "  pulay_hist = "<< pulay.history_size()
                  << "  Ne_occ = " << nelec_occ
                  << "  Ne_out = " << nelec_out
                  << "  mu = " << occ.mu
                  << "  eps0 = " << ks.eigenvalues[0]
                  << "\n";


        if (scf_iter % 5 == 0 || scf_iter < 5) {
            const int nprint = std::min(8, nbands_solve);

            for (int ib = 0; ib < nprint; ++ib) {
                std::cout << "  band " << ib
                      << "  eps = " << std::setw(20)
                      << ks.eigenvalues[ib]
                      << "  occ = " << occ.occ[ib]
                      << "\n";
            }
        }

        if (scf_iter > 0 &&
            drho < scf_rho_tol &&
            dE < 1.0e-9) {
            std::cout << "SCF converged.\n";
            break;
        }

        E_old = energy_for_convergence;

        /*
         * 6. Use current eigenvectors as next initial guess.
         */
        C_guess = ks.eigenvectors.leftCols(nbands_solve);
    }


    return 0;
}