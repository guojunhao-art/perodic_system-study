#pragma once

#include "input.hpp"
#include "scf.hpp"

#include <string>
#include <vector>

struct DensityOfStatesSample {
    double energy_ha = 0.0;
    double total_per_ha = 0.0;
    double integrated_total = 0.0;
    std::vector<double> spin_per_ha;
    std::vector<double> integrated_spin;
};

struct DensityOfStatesResult {
    int nspin = 1;
    int nbands = 0;
    int kpoint_count = 0;
    double fermi_energy_ha = 0.0;
    double gaussian_sigma_ha = 0.0;
    double energy_min_ha = 0.0;
    double energy_max_ha = 0.0;
    std::vector<DensityOfStatesSample> samples;
};

/*
 * Broaden the converged SCF eigenvalues on their normalized Brillouin-zone
 * k-point mesh.  A non-spin-polarized state carries a degeneracy of two;
 * each collinear spin channel carries a degeneracy of one.
 */
DensityOfStatesResult compute_density_of_states(
    const KPointSCFResult& scf,
    const SCFOptions& scf_options,
    const DensityOfStatesOptions& dos_options);

void write_density_of_states(
    const std::string& path,
    const DensityOfStatesResult& result);
