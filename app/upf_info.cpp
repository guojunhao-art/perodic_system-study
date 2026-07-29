#include "upf_reader.hpp"
#include "upf_local_potential.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: upf_info FILE.UPF [GAUSSIAN_WIDTH_BOHR]\n";
        return 2;
    }

    try {
        const UPFData upf = read_nc_upf(argv[1]);

        std::cout << "UPF summary\n"
                  << "-----------\n"
                  << "file         : " << upf.source_path << "\n"
                  << "version      : " << upf.header.version << "\n"
                  << "element      : " << upf.header.element << "\n"
                  << "type         : " << upf.header.pseudo_type << "\n"
                  << "functional   : " << upf.header.functional << "\n"
                  << "relativistic : " << upf.header.relativistic << "\n"
                  << "z valence    : " << upf.header.z_valence << "\n"
                  << "mesh points  : " << upf.header.mesh_size << "\n"
                  << "projectors   : " << upf.header.number_of_projectors << "\n"
                  << "atomic wfc   : " << upf.header.number_of_wavefunctions << "\n";

        for (const UPFProjector& projector : upf.projectors) {
            std::cout << "  beta " << projector.index
                      << "  label=" << projector.label
                      << "  l=" << projector.angular_momentum
                      << "  cutoff_index=" << projector.cutoff_radius_index
                      << "\n";
        }

        for (const UPFAtomicWavefunction& wavefunction :
             upf.atomic_wavefunctions) {
            std::cout << "  chi  " << wavefunction.index
                      << "  label=" << wavefunction.label
                      << "  l=" << wavefunction.angular_momentum
                      << "  occupation=" << wavefunction.occupation
                      << "\n";
        }

        if (!upf.dij_ry.empty()) {
            const int n = upf.header.number_of_projectors;
            std::cout << "D_ij (Ry):\n" << std::scientific << std::setprecision(10);
            for (int i = 0; i < n; ++i) {
                std::cout << " ";
                for (int j = 0; j < n; ++j) {
                    std::cout << " " << std::setw(17) << upf.dij(i, j);
                }
                std::cout << "\n";
            }
            std::cout << "D_ij (Ha):\n";
            for (int i = 0; i < n; ++i) {
                std::cout << " ";
                for (int j = 0; j < n; ++j) {
                    std::cout << " " << std::setw(17)
                              << RYDBERG_TO_HARTREE * upf.dij(i, j);
                }
                std::cout << "\n";
            }
        }

        if (argc == 3) {
            const double sigma = std::stod(argv[2]);
            const UPFLocalSpecies local =
                prepare_upf_local_species(upf, sigma);

            std::cout << "\nScreened local potential\n"
                      << "------------------------\n"
                      << "Gaussian sigma (Bohr) : " << sigma << "\n"
                      << "Delta V(rmax) (Ha)    : "
                      << std::scientific << std::setprecision(10)
                      << local.correction_hartree.back() << "\n"
                      << "isolated radial transforms (Ha Bohr^3):\n"
                      << "       G        Delta V(G)          full kernel\n";
            for (double g : std::vector<double>{0.0, 0.5, 1.0, 2.0, 4.0}) {
                std::cout << "  " << std::setw(7) << std::fixed
                          << std::setprecision(2) << g
                          << "  " << std::setw(18) << std::scientific
                          << std::setprecision(10)
                          << upf_local_correction_transform(local, g)
                          << "  " << std::setw(18)
                          << upf_local_kernel_G(local, g)
                          << "\n";
            }
            std::cout << "At G=0 the full-kernel column is the standard "
                      << "finite integral of V_loc + Z/r.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
