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

    std::cout << std::setprecision(16);
    std::cout << "Basis size = " << basis.size() << "\n";

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

    std::cout << "Smooth ion-ion energy = "
              << local_potential.Eion_smooth << "\n";

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

    std::cout << "Ionic Hellmann-Feynman forces (Ha/Bohr):\n";
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        std::cout << "  ion " << iion
                  << "  F_loc = " << forces.local[iion].transpose()
                  << "  F_II = " << forces.ion_ion[iion].transpose()
                  << "  F_NL = " << forces.nonlocal[iion].transpose()
                  << "  F_total = " << forces.total[iion].transpose()
                  << "\n";
    }

    return scf.converged ? 0 : 1;
}
