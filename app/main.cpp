#include "bands.hpp"
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
            if (config.calculation == CalculationType::RelaxBands) {
                CalculationConfig relaxation_config = config;
                relaxation_config.calculation = CalculationType::Relax;
                const RelaxationResult relaxation =
                    run_fixed_cell_relaxation(
                        structure, relaxation_config, log_stream
                    );
                if (!relaxation.converged) {
                    if (parallel::is_root()) {
                        print_single_point_result(
                            std::cout,
                            relaxation.structure,
                            relaxation.electronic
                        );
                    }
                    return 1;
                }
                if (log_stream) {
                    *log_stream
                        << "\n FINAL STATIC SCF ON THE RELAXED STRUCTURE\n"
                        << " ===============================================================================\n";
                }
                const SinglePointResult final_scf = run_single_point(
                    relaxation.structure, config, log_stream
                );
                if (parallel::is_root()) {
                    print_single_point_result(
                        std::cout, relaxation.structure, final_scf
                    );
                }
                if (!final_scf.converged) {
                    return 1;
                }
                const BandStructureResult bands = run_band_structure(
                    relaxation.structure, config, final_scf, log_stream
                );
                if (parallel::is_root()) {
                    write_band_structure(
                        config.bands.output_path, bands
                    );
                    std::cout << "  band structure written to "
                        << config.bands.output_path << "\n";
                }
                return bands.converged ? 0 : 1;
            }
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
            if (!result.converged) {
                return 1;
            }
            if (config.calculation == CalculationType::Bands) {
                const BandStructureResult bands = run_band_structure(
                    structure, config, result, log_stream
                );
                if (parallel::is_root()) {
                    write_band_structure(
                        config.bands.output_path, bands
                    );
                    std::cout << "  band structure written to "
                        << config.bands.output_path << "\n";
                }
                return bands.converged ? 0 : 1;
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
