#pragma once

#include "calculation.hpp"
#include "input.hpp"
#include "scf.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct PseudopotentialFingerprint {
    std::uint64_t content_hash = 0;
    std::uint64_t size_bytes = 0;
};

struct SCFCheckpoint {
    int format_version = 1;
    AtomicStructure structure;
    double ecut_hartree = 0.0;
    std::array<int, 3> fft_grid{{0, 0, 0}};
    int nspin = 1;
    double nelect = 0.0;
    LDAFunctional lda_functional = LDAFunctional::PerdewZunger;
    double ewald_width_bohr = 0.0;
    double fermi_energy_ha = 0.0;
    std::map<std::string, PseudopotentialFingerprint> pseudopotentials;
    std::vector<std::vector<double>> spin_densities;
};

PseudopotentialFingerprint fingerprint_file(const std::string& path);

SCFCheckpoint make_scf_checkpoint(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SinglePointResult& result);

void write_scf_checkpoint(
    const std::string& path,
    const SCFCheckpoint& checkpoint);

SCFCheckpoint read_scf_checkpoint(const std::string& path);

/*
 * Validate all inputs that define the fixed Kohn--Sham potential.  The
 * k-point mesh and nbands are intentionally excluded because NSCF is
 * expected to change them.
 */
void validate_scf_checkpoint(
    const SCFCheckpoint& checkpoint,
    const AtomicStructure& structure,
    const CalculationConfig& config,
    double resolved_nelect);
