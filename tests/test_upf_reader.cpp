#include "upf_reader.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

std::string data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& label) {
    const double error = std::abs(actual - expected);
    std::cout << label << " error = " << error << "\n";
    if (error > 1.0e-14) {
        throw std::runtime_error(label + " mismatch");
    }
}

void test_local_only_file() {
    const UPFData upf = read_nc_upf(data_path("minimal_local_nc.upf"));

    require(upf.header.version == "2.0.1", "wrong UPF version");
    require(upf.header.element == "H", "element was not trimmed");
    require(upf.header.pseudo_type == "NC", "wrong pseudo type");
    require(upf.header.mesh_size == 3, "wrong mesh size");
    require(upf.header.number_of_projectors == 0, "wrong projector count");
    require(upf.projectors.empty(), "local-only file produced projectors");
    require(upf.dij_ry.empty(), "zero-projector dummy DIJ was not ignored");
    require_close(upf.header.z_valence, 1.0, "Fortran-D exponent");
    require_close(upf.local_potential_ry[2], -0.5, "local potential");
}

void test_s_and_p_projectors() {
    const UPFData upf = read_nc_upf(data_path("minimal_sp_nc.upf"));

    require(upf.header.element == "Si", "wrong element");
    require(upf.header.number_of_projectors == 2, "wrong projector count");
    require(upf.projectors.size() == 2, "projectors were not read");
    require(upf.projectors[0].angular_momentum == 0, "wrong s angular momentum");
    require(upf.projectors[1].angular_momentum == 1, "wrong p angular momentum");
    require(upf.projectors[0].r_times_beta.size() == 5, "s projector was not padded");
    require_close(upf.projectors[0].r_times_beta[3], 0.0, "s padding");
    require_close(upf.projectors[1].r_times_beta[2], 0.3, "p projector value");
    require_close(upf.dij(0, 0), 1.5, "D_ss in Ry");
    require_close(upf.dij(0, 1), 0.0, "D_sp in Ry");
    require_close(upf.dij(1, 1), 3.5, "D_pp in Ry");
    require_close(
        RYDBERG_TO_HARTREE * upf.dij(1, 1),
        1.75,
        "D_pp in Hartree"
    );
}

void test_unsupported_physics_is_rejected() {
    bool rejected = false;
    try {
        (void)read_nc_upf(data_path("unsupported_us.upf"));
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("pseudo_type=NC")
            != std::string::npos;
    }
    require(rejected, "ultrasoft fixture was not rejected clearly");
}

} // namespace

int main() {
    try {
        test_local_only_file();
        test_s_and_p_projectors();
        test_unsupported_physics_is_rejected();
    } catch (const std::exception& error) {
        std::cerr << "UPF-reader test failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "All NC-UPF reader tests passed.\n";
    return 0;
}
