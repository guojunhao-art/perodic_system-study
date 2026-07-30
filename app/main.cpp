#include "bands.hpp"
#include "calculation.hpp"
#include "checkpoint.hpp"
#include "dos.hpp"
#include "input.hpp"
#include "nscf.hpp"
#include "parallel.hpp"
#include "pdos.hpp"
#include "relaxation.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_checkpoint_if_requested(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SinglePointResult& result) {

    if (config.checkpoint_output_path.empty()) {
        return;
    }
    std::string local_error;
    if (parallel::is_root()) {
        try {
            write_scf_checkpoint(
                config.checkpoint_output_path,
                make_scf_checkpoint(structure, config, result)
            );
        } catch (const std::exception& error) {
            local_error = error.what();
        }
    }
    const std::string error =
        parallel::first_error(local_error);
    if (!error.empty()) {
        throw std::runtime_error(
            "SCF checkpoint write failed: " + error
        );
    }
    if (parallel::is_root()) {
        std::cout << "  SCF checkpoint written to "
            << config.checkpoint_output_path << "\n";
    }
}

} // namespace

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
            if (config.calculation == CalculationType::NSCF) {
                const SCFCheckpoint checkpoint =
                    read_scf_checkpoint(
                        config.checkpoint_input_path
                    );
                const NSCFResult nscf =
                    run_fixed_density_nscf(
                        structure, config, checkpoint, log_stream
                    );
                if (!nscf.converged) {
                    return 1;
                }
                if (nscf.band_path) {
                    if (parallel::is_root()) {
                        if (!config.bands.output_path.empty()) {
                            write_band_structure(
                                config.bands.output_path,
                                nscf
                            );
                            std::cout
                                << "  NSCF band structure written to "
                                << config.bands.output_path
                                << "\n";
                        }
                        if (!config.bands.projection_output_path.empty()) {
                            write_fat_band_structure(
                                config.bands.projection_output_path,
                                nscf
                            );
                            std::cout
                                << "  NSCF fat bands written to "
                                << config.bands.projection_output_path
                                << "\n";
                        }
                    }
                    return 0;
                }
                const bool needs_dos_grid =
                    !config.dos.output_path.empty()
                    || !config.pdos.output_path.empty();
                if (!needs_dos_grid) {
                    throw std::runtime_error(
                        "A Brillouin-zone NSCF calculation has no "
                        "DOS or PDOS output request."
                    );
                }
                const DensityOfStatesResult dos =
                    compute_density_of_states(
                        nscf.electronic,
                        nscf.options_used,
                        config.dos
                    );
                if (parallel::is_root() &&
                    !config.dos.output_path.empty()) {
                    write_density_of_states(
                        config.dos.output_path, dos
                    );
                    std::cout
                        << "  NSCF density of states written to "
                        << config.dos.output_path << "\n";
                }
                if (!config.pdos.output_path.empty()) {
                    const ProjectedDensityOfStatesResult pdos =
                        compute_projected_density_of_states(
                            nscf.electronic,
                            nscf.options_used,
                            dos,
                            nscf.projection
                        );
                    if (parallel::is_root()) {
                        write_projected_density_of_states(
                            config.pdos.output_path,
                            pdos
                        );
                        std::cout
                            << "  NSCF projected density of states "
                               "written to "
                            << config.pdos.output_path
                            << "\n";
                    }
                }
                return 0;
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
                if (result.converged) {
                    write_checkpoint_if_requested(
                        result.structure, config, result.electronic
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
            write_checkpoint_if_requested(
                structure, config, result
            );
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
