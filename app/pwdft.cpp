#include "calculation.hpp"
#include "input.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: pwdft CALCULATION.in\n";
        return 2;
    }

    try {
        const CalculationConfig config = read_calculation_config(argv[1]);
        const AtomicStructure structure = read_poscar(config.structure_path);
        const SinglePointResult result = run_single_point(
            structure,
            config,
            &std::cout
        );
        print_single_point_result(std::cout, structure, result);
        return result.converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PWDFT calculation failed: " << error.what() << "\n";
        return 1;
    }
}
