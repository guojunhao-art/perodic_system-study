#include "calculation.hpp"
#include "input.hpp"
#include "parallel.hpp"
#include "relaxation.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        parallel::Environment environment(argc, argv);
        if (argc != 2) {
            if (parallel::is_root()) {
                std::cerr << "Usage: pwdft CALCULATION.in\n";
            }
            return 2;
        }

        try {
            const CalculationConfig config = read_calculation_config(argv[1]);
            const AtomicStructure structure = read_poscar(
                config.structure_path
            );
            std::ostream* log_stream =
                parallel::is_root() ? &std::cout : nullptr;
            if (config.calculation == CalculationType::Relax) {
                const RelaxationResult result =
                    run_fixed_cell_relaxation(
                        structure, config, log_stream
                    );
                if (parallel::is_root()) {
                    print_single_point_result(
                        std::cout,
                        result.structure,
                        result.electronic
                    );
                }
                return result.converged ? 0 : 1;
            }

            const SinglePointResult result = run_single_point(
                structure, config, log_stream
            );
            if (parallel::is_root()) {
                print_single_point_result(std::cout, structure, result);
            }
            return result.converged ? 0 : 1;
        } catch (const std::exception& error) {
            if (parallel::is_root()) {
                std::cerr << "PWDFT calculation failed: "
                    << error.what() << "\n";
            }
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "PWDFT MPI initialization failed: "
            << error.what() << "\n";
        return 1;
    }
}
