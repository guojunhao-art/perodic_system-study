#include "core.hpp"
#include "ewald.hpp"
#include "forces.hpp"
#include "scf.hpp"
#include "upf_local_potential.hpp"
#include "upf_nonlocal.hpp"
#include "upf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double hartree_to_eV = 27.211386245988;

bool is_perdew_zunger_lda(const std::string& functional) {
    std::string uppercase = functional;
    std::transform(
        uppercase.begin(),
        uppercase.end(),
        uppercase.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        }
    );
    return uppercase.find("PZ") != std::string::npos;
}

void require_fft_grid_for_basis_products(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid) {

    if (basis.gvectors.empty()) {
        throw std::runtime_error("The plane-wave basis is empty.");
    }

    Eigen::Vector3i minimum_frequency = basis.gvectors.front().n;
    Eigen::Vector3i maximum_frequency = basis.gvectors.front().n;
    for (const GVector& gvector : basis.gvectors) {
        minimum_frequency = minimum_frequency.cwiseMin(gvector.n);
        maximum_frequency = maximum_frequency.cwiseMax(gvector.n);
    }

    const Eigen::Vector3i grid_sizes(grid.n1, grid.n2, grid.n3);
    for (int direction = 0; direction < 3; ++direction) {
        const int largest_product_frequency =
            maximum_frequency[direction] - minimum_frequency[direction];
        const int largest_unaliased_frequency =
            (grid_sizes[direction] - 1) / 2;
        if (largest_product_frequency > largest_unaliased_frequency) {
            throw std::runtime_error(
                "FFT_N is too small for products of cutoff wavefunctions; "
                "increase the real-space FFT grid."
            );
        }
    }
}

Ion make_ewald_ion(
    const UPFLocalIon& upf_ion,
    double valence_charge,
    double gaussian_width_bohr) {

    Ion ion;
    ion.frac_position = upf_ion.frac_position;
    ion.Z = valence_charge;
    ion.sigma = gaussian_width_bohr;
    return ion;
}

IonicForceComponents assemble_upf_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFLocalSpecies>& local_species,
    const std::vector<UPFLocalIon>& upf_ions,
    const std::vector<Ion>& ewald_ions,
    double ewald_width_bohr,
    const std::vector<std::complex<double>>& density_G,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& orbitals,
    const std::vector<double>& occupations) {

    IonicForceComponents forces;
    forces.local = compute_upf_local_ionic_forces(
        lattice,
        grid,
        local_species,
        upf_ions,
        density_G
    );
    forces.ion_ion = compute_point_ion_ion_ewald_forces(
        lattice,
        grid,
        ewald_ions,
        ewald_width_bohr
    );
    forces.nonlocal = compute_nonlocal_ionic_forces(
        basis,
        projectors,
        orbitals,
        occupations,
        static_cast<int>(upf_ions.size())
    );
    forces.total.resize(upf_ions.size(), Eigen::Vector3d::Zero());
    for (int iion = 0; iion < static_cast<int>(upf_ions.size()); ++iion) {
        forces.total[iion] =
            forces.local[iion]
            + forces.ion_ion[iion]
            + forces.nonlocal[iion];
    }
    return forces;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr
            << "Usage: fft PSEUDO.UPF [ECUT_HA] [CELL_BOHR] [FFT_N] "
               "[SMEARING_EV]\n";
        return 2;
    }

    try {
        const UPFData upf = read_nc_upf(argv[1]);
        if (!is_perdew_zunger_lda(upf.header.functional)) {
            throw std::runtime_error(
                "The UPF functional is not PZ-LDA, while this driver uses "
                "LibXC LDA_X + LDA_C_PZ. Use a matching PZ NC-UPF."
            );
        }
        const double recommended_cutoff_hartree =
            RYDBERG_TO_HARTREE * upf.header.wavefunction_cutoff_ry;
        const double ecut_hartree = argc >= 3
            ? std::stod(argv[2])
            : (recommended_cutoff_hartree > 0.0
                ? recommended_cutoff_hartree
                : 10.0);
        const double cell_length_bohr =
            argc >= 4 ? std::stod(argv[3]) : 12.0;
        const int fft_size = argc >= 5 ? std::stoi(argv[4]) : 36;
        const double smearing_eV =
            argc >= 6 ? std::stod(argv[5]) : 0.05;
        const double ewald_width_bohr = 0.70;

        if (!std::isfinite(ecut_hartree) || ecut_hartree <= 0.0 ||
            !std::isfinite(cell_length_bohr) || cell_length_bohr <= 0.0 ||
            fft_size < 4 || !std::isfinite(smearing_eV) ||
            smearing_eV < 0.0) {
            throw std::runtime_error(
                "Cutoff, cell length, FFT size, or smearing is invalid."
            );
        }

        const Lattice lattice(
            Eigen::Vector3d(cell_length_bohr, 0.0, 0.0),
            Eigen::Vector3d(0.0, cell_length_bohr, 0.0),
            Eigen::Vector3d(0.0, 0.0, cell_length_bohr)
        );
        const FFTGrid grid(fft_size, fft_size, fft_size);
        FFTWorkspace fft(grid);
        PlaneWaveBasis3D basis;
        basis.generate(lattice, Eigen::Vector3d::Zero(), ecut_hartree);
        require_fft_grid_for_basis_products(basis, grid);

        const UPFLocalSpecies local =
            prepare_upf_local_species(upf, ewald_width_bohr);
        const UPFNonlocalSpecies nonlocal =
            prepare_upf_nonlocal_species(upf);
        const std::vector<UPFLocalSpecies> local_species{local};
        const std::vector<UPFNonlocalSpecies> nonlocal_species{nonlocal};
        const std::vector<UPFLocalIon> upf_ions{
            {0, Eigen::Vector3d(0.5, 0.5, 0.5)}
        };
        const std::vector<Ion> ewald_ions{
            make_ewald_ion(
                upf_ions[0],
                upf.header.z_valence,
                ewald_width_bohr
            )
        };

        const std::vector<double> local_potential =
            build_upf_local_potential_real(
                lattice,
                fft,
                local_species,
                upf_ions
            );
        const std::vector<NonlocalProjector> projectors =
            build_upf_nonlocal_projectors(
                lattice,
                basis,
                nonlocal_species,
                upf_ions
            );
        const EwaldEnergyComponents ewald =
            compute_point_ion_ion_ewald_energy(
                lattice,
                grid,
                ewald_ions,
                ewald_width_bohr
            );

        SCFOptions options;
        options.nelec = upf.header.z_valence;
        options.nbands = std::max(
            8,
            static_cast<int>(std::ceil(0.5 * options.nelec)) + 4
        );
        if (basis.size() < options.nbands) {
            throw std::runtime_error(
                "The plane-wave basis has fewer vectors than requested bands."
            );
        }
        options.occupation_mode = smearing_eV > 0.0
            ? OccupationMode::FermiDirac
            : OccupationMode::DegeneracyAwareZeroT;
        options.fixed_occupations.assign(options.nbands, 0.0);
        options.smearing_sigma = smearing_eV / hartree_to_eV;
        options.max_iterations = 240;
        options.density_tolerance = 1.0e-7;
        options.energy_tolerance = 1.0e-9;
        options.eigensolver_max_iterations = 100;
        options.eigensolver_tolerance = 1.0e-10;
        options.mixing_alpha = 0.10;
        options.pulay_max_history = 8;
        options.pulay_min_history = 2;
        options.pulay_regularization = 1.0e-12;
        options.verbosity = SCFVerbosity::Compact;

        std::cout << "\n"
                  << " PWDFT: Gamma-point NC-UPF calculation\n"
                  << " -------------------------------------------------------------------------------\n"
                  << "  PSEUDO = " << argv[1]
                  << "    ELEMENT = " << upf.header.element << "\n"
                  << "  NPLWV  = " << basis.size()
                  << "    NGX = " << grid.n1
                  << "    NGY = " << grid.n2
                  << "    NGZ = " << grid.n3 << "\n"
                  << "  NELECT = " << options.nelec
                  << "    NBANDS = " << options.nbands
                  << "    ENCUT = " << ecut_hartree << " Ha\n"
                  << "  NIONS  = " << upf_ions.size()
                  << "    NPROJ(radial/expanded) = "
                  << upf.projectors.size() << "/"
                  << projectors.size() << "\n"
                  << "  EDIFF  = " << options.energy_tolerance
                  << "    EDIFFRHO = " << options.density_tolerance << "\n"
                  << "  XC     = " << lda_functional_name(options.lda_functional)
                  << "    LibXC " << libxc_runtime_version() << "\n";

        const SCFResult scf = run_scf(
            lattice,
            basis,
            fft,
            local_potential,
            ewald.total,
            projectors,
            options,
            {},
            &std::cout
        );
        if (!scf.converged) {
            std::cerr
                << "Warning: SCF reached the iteration limit without convergence.\n";
        }

        const auto density_G = build_density_G(fft, scf.density);
        const IonicForceComponents forces = assemble_upf_forces(
            lattice,
            grid,
            basis,
            local_species,
            upf_ions,
            ewald_ions,
            ewald_width_bohr,
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
                  << sigma0_energy << "\n"
                  << "  E_NL = " << std::setw(20)
                  << scf.energy.nonlocal
                  << "  E_II(Ewald) = " << std::setw(20)
                  << scf.energy.ion_smooth << "\n\n";

        std::cout << "  POSITION (Bohr)" << std::setw(45)
                  << "TOTAL-FORCE (Ha/Bohr)\n"
                  << "  -----------------------------------------------------------------------------------\n";
        for (int iion = 0; iion < static_cast<int>(upf_ions.size()); ++iion) {
            const Eigen::Vector3d position =
                lattice.cart_from_frac(upf_ions[iion].frac_position);
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
                  << "  force split max |F_loc| / |F_II| / |F_NL| = "
                  << std::scientific << std::setprecision(5)
                  << forces.local[0].cwiseAbs().maxCoeff() << " / "
                  << forces.ion_ion[0].cwiseAbs().maxCoeff() << " / "
                  << forces.nonlocal[0].cwiseAbs().maxCoeff()
                  << " Ha/Bohr\n"
                  << "  reached required accuracy: "
                  << (scf.converged
                      ? "electronic convergence achieved"
                      : "no")
                  << "\n";

        return scf.converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NC-UPF calculation failed: " << error.what() << "\n";
        return 1;
    }
}
