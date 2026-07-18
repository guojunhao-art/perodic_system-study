#include "core.hpp"
#include "forces.hpp"
#include "potentials.hpp"
#include "scf.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    double nelec = 2.0;
    if (argc > 1) {
        nelec = std::atof(argv[1]);
    }

    OccupationMode occupation_mode = OccupationMode::FermiDirac;
    if (argc > 2 && std::atoi(argv[2]) != 1) {
        occupation_mode = OccupationMode::DegeneracyAwareZeroT;
    }

    double sigma_eV = 0.05;
    if (argc > 3) {
        sigma_eV = std::atof(argv[3]);
    }

    double ecut = 3.0;
    if (argc > 4) {
        ecut = std::atof(argv[4]);
    }

    const double a = 10.0;
    const Lattice lattice(
        Eigen::Vector3d(a, 0.0, 0.0),
        Eigen::Vector3d(0.0, a, 0.0),
        Eigen::Vector3d(0.0, 0.0, a)
    );
    const FFTGrid grid(32, 32, 32);
    FFTWorkspace fft(grid);

    PlaneWaveBasis3D basis;
    basis.generate(lattice, Eigen::Vector3d::Zero(), ecut);

    Ion ion;
    ion.frac_position = Eigen::Vector3d(0.5, 0.5, 0.5);
    ion.Z = nelec;
    ion.sigma = 0.7;
    ion.short_amp = 1.0;
    ion.beta_s_rc = 0.5;
    ion.beta_p_rc = 0.5;

    if (argc > 5) {
        ion.sigma = std::atof(argv[5]);
    }
    if (argc > 6) {
        ion.beta_s_D = std::atof(argv[6]);
    }
    if (argc > 7) {
        ion.beta_p_D = std::atof(argv[7]);
    }

    const std::vector<Ion> ions{ion};
    const LocalPotentialComponents local_potential =
        build_local_pseudopotential_components(lattice, fft, ions);
    const auto projectors =
        build_gaussian_nonlocal_projectors(lattice, basis, ions);

    SCFOptions options;
    options.nelec = nelec;
    options.nbands = static_cast<int>(std::max(3.0 * nelec, 10.0));
    options.occupation_mode = occupation_mode;
    options.fixed_occupations.assign(options.nbands, 0.0);
    options.smearing_sigma = sigma_eV / 27.211386245988;
    options.max_iterations = 200;
    options.density_tolerance = 1.0e-7;
    options.energy_tolerance = 1.0e-9;
    options.eigensolver_max_iterations = 80;
    options.eigensolver_tolerance = 1.0e-10;
    options.mixing_alpha = 0.10;
    options.pulay_max_history = 6;
    options.pulay_min_history = 2;
    options.pulay_regularization = 1.0e-12;
    options.verbosity = SCFVerbosity::Compact;

    std::cout << "\n"
              << " PWDFT toy code: Gamma-point plane-wave calculation\n"
              << " -------------------------------------------------------------------------------\n"
              << "  NPLWV  = " << basis.size()
              << "    NGX = " << grid.n1
              << "    NGY = " << grid.n2
              << "    NGZ = " << grid.n3 << "\n"
              << "  NELECT = " << nelec
              << "    NBANDS = " << options.nbands
              << "    ENCUT = " << ecut << " Ha\n"
              << "  NIONS  = " << ions.size()
              << "    EDIFF = " << options.energy_tolerance
              << "    EDIFFRHO = " << options.density_tolerance << "\n"
              << "  XC     = " << lda_functional_name(options.lda_functional)
              << "    LibXC " << libxc_runtime_version() << "\n";

    const SCFResult scf = run_scf(
        lattice,
        basis,
        fft,
        local_potential.Vtotal,
        local_potential.Eion_smooth,
        projectors,
        options,
        {},
        &std::cout
    );

    if (!scf.converged) {
        std::cerr << "Warning: SCF reached the iteration limit without convergence.\n";
    }

    const auto density_G = build_density_G(fft, scf.density);
    const IonicForceComponents forces = compute_ionic_forces(
        lattice,
        grid,
        basis,
        ions,
        density_G,
        projectors,
        scf.orbitals,
        scf.occupations.occ
    );

    const double energy_without_entropy =
        scf.energy.total + scf.energy.ion_smooth;
    const double sigma0_energy =
        scf.energy.sigma0_estimate + scf.energy.ion_smooth;

    std::cout << "\n"
              << "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (Ha)\n"
              << "  ---------------------------------------------------\n"
              << std::setprecision(12) << std::scientific
              << "  free  energy   TOTEN  = " << std::setw(20)
              << scf.variational_energy << "\n"
              << "  energy  without entropy = " << std::setw(18)
              << energy_without_entropy
              << "  energy(sigma->0) = " << std::setw(18)
              << sigma0_energy << "\n\n";

    std::cout << "  POSITION (Bohr)" << std::setw(45)
              << "TOTAL-FORCE (Ha/Bohr)\n"
              << "  -----------------------------------------------------------------------------------\n";
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        const Eigen::Vector3d position =
            lattice.cart_from_frac(ions[iion].frac_position);
        std::cout << "  " << std::fixed << std::setprecision(8)
                  << std::setw(14) << position[0]
                  << std::setw(14) << position[1]
                  << std::setw(14) << position[2]
                  << "    "
                  << std::setw(14) << forces.total[iion][0]
                  << std::setw(14) << forces.total[iion][1]
                  << std::setw(14) << forces.total[iion][2]
                  << "\n";
    }
    std::cout << "  -----------------------------------------------------------------------------------\n"
              << "  reached required accuracy: "
              << (scf.converged ? "electronic convergence achieved" : "no")
              << "\n";

    return scf.converged ? 0 : 1;
}
