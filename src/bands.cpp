#include "bands.hpp"

#include "nscf.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace {

int electronic_state_index(
    int spin,
    int kpoint,
    int kpoint_count) {

    return spin * kpoint_count + kpoint;
}

} // namespace

std::vector<BandPathSample> interpolate_band_path(
    const Lattice& lattice,
    const BandStructureOptions& options) {

    if (options.path.size() < 2) {
        throw std::runtime_error(
            "A band structure requires at least two band_point nodes."
        );
    }
    if (options.points_per_segment < 2) {
        throw std::runtime_error(
            "band_points_per_segment must be at least 2."
        );
    }

    std::vector<BandPathSample> samples;
    samples.reserve(
        (options.path.size() - 1)
            * static_cast<std::size_t>(
                options.points_per_segment - 1
            )
        + 1
    );
    double distance = 0.0;
    for (int segment = 0;
         segment + 1 < static_cast<int>(options.path.size());
         ++segment) {
        const BandPathNode& begin = options.path[segment];
        const BandPathNode& end = options.path[segment + 1];
        if (!begin.frac_position.allFinite() ||
            !end.frac_position.allFinite()) {
            throw std::runtime_error(
                "Band-path reciprocal coordinates must be finite."
            );
        }
        const Eigen::Vector3d step_fractional =
            (end.frac_position - begin.frac_position)
            / static_cast<double>(
                options.points_per_segment - 1
            );
        const double step_distance =
            (lattice.B * step_fractional).norm();
        const int first_sample = segment == 0 ? 0 : 1;
        for (int point = first_sample;
             point < options.points_per_segment;
             ++point) {
            if (!samples.empty()) {
                distance += step_distance;
            }
            BandPathSample sample;
            sample.frac_position =
                begin.frac_position
                + static_cast<double>(point)
                    * step_fractional;
            sample.distance_bohr_inverse = distance;
            if (point == 0) {
                sample.label = begin.label;
            }
            if (point ==
                options.points_per_segment - 1) {
                sample.label = end.label;
            }
            samples.push_back(std::move(sample));
        }
    }
    return samples;
}

void write_band_structure(
    const std::string& path,
    const NSCFResult& result) {

    if (path.empty()) {
        throw std::runtime_error(
            "The band output path is empty."
        );
    }
    if (!result.band_path || result.path.empty() ||
        !result.diagonalization.converged ||
        result.diagonalization.nbands <= 0 ||
        (result.diagonalization.nspin != 1 &&
         result.diagonalization.nspin != 2)) {
        throw std::runtime_error(
            "Cannot write an incomplete NSCF band-structure result."
        );
    }
    const int kpoint_count =
        static_cast<int>(result.path.size());
    if (static_cast<int>(
            result.diagonalization.states.size()
        ) != result.diagonalization.nspin * kpoint_count) {
        throw std::runtime_error(
            "NSCF band-state count does not match nspin times path size."
        );
    }

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create band-structure output: " + path
        );
    }
    output << "# PWDFT fixed-density NSCF band structure\n"
        << "# fermi reference comes from the SCF checkpoint\n"
        << "# fermi_energy_ha = " << std::scientific
        << std::setprecision(16)
        << result.diagonalization.fermi_energy_ha << "\n"
        << "# fermi_energy_ev = "
        << result.diagonalization.fermi_energy_ha
            * HARTREE_TO_EV << "\n"
        << "# path nodes: index distance_bohr^-1 label kx ky kz\n";
    for (int ik = 0; ik < kpoint_count; ++ik) {
        if (result.path[ik].label.empty()) {
            continue;
        }
        output << "# node " << ik << " "
            << result.path[ik].distance_bohr_inverse << " "
            << result.path[ik].label << " "
            << result.path[ik].frac_position[0] << " "
            << result.path[ik].frac_position[1] << " "
            << result.path[ik].frac_position[2] << "\n";
    }
    output
        << "# columns: k_index distance_bohr^-1 kx ky kz spin band "
           "eigenvalue_ha eigenvalue_ev eigenvalue_minus_fermi_ev residual\n";

    for (int spin = 0;
         spin < result.diagonalization.nspin;
         ++spin) {
        for (int band = 0;
             band < result.diagonalization.nbands;
             ++band) {
            for (int ik = 0; ik < kpoint_count; ++ik) {
                const NSCFElectronicState& state =
                    result.diagonalization.states[
                        electronic_state_index(
                            spin, ik, kpoint_count
                        )
                    ];
                const double eigenvalue =
                    state.eigenvalues[band];
                output << std::setw(6) << ik << " "
                    << std::scientific
                    << std::setprecision(12)
                    << std::setw(20)
                    << result.path[ik]
                        .distance_bohr_inverse
                    << " " << std::setw(18)
                    << result.path[ik].frac_position[0]
                    << " " << std::setw(18)
                    << result.path[ik].frac_position[1]
                    << " " << std::setw(18)
                    << result.path[ik].frac_position[2]
                    << " " << std::setw(3) << spin
                    << " " << std::setw(5) << band
                    << " " << std::setw(20) << eigenvalue
                    << " " << std::setw(20)
                    << eigenvalue * HARTREE_TO_EV
                    << " " << std::setw(20)
                    << (eigenvalue -
                        result.diagonalization.fermi_energy_ha)
                        * HARTREE_TO_EV
                    << " " << std::setw(14)
                    << state.residual_norms[band] << "\n";
            }
            output << "\n";
        }
    }
}
