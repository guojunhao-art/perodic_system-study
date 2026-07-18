#include "upf_reader.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: upf_info FILE.UPF\n";
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
                  << "projectors   : " << upf.header.number_of_projectors << "\n";

        for (const UPFProjector& projector : upf.projectors) {
            std::cout << "  beta " << projector.index
                      << "  label=" << projector.label
                      << "  l=" << projector.angular_momentum
                      << "  cutoff_index=" << projector.cutoff_radius_index
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
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
