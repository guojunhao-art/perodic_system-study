#include "checkpoint.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

constexpr const char* checkpoint_magic = "PWDFT_SCF_CHECKPOINT";
constexpr int current_checkpoint_version = 2;
constexpr int legacy_lda_checkpoint_version = 1;
constexpr double metadata_tolerance = 1.0e-10;

void require_key(
    std::istream& input,
    const std::string& expected,
    const std::string& path) {

    std::string key;
    if (!(input >> key) || key != expected) {
        throw std::runtime_error(
            path + ": expected checkpoint field '" + expected + "'."
        );
    }
}

bool nearly_equal(double first, double second) {
    const double scale = std::max({
        1.0, std::abs(first), std::abs(second)
    });
    return std::abs(first - second) <= metadata_tolerance * scale;
}

std::size_t checked_grid_size(const std::array<int, 3>& dimensions) {
    std::size_t result = 1;
    for (int dimension : dimensions) {
        if (dimension <= 0) {
            throw std::runtime_error(
                "Checkpoint FFT dimensions must be positive."
            );
        }
        if (result >
            std::numeric_limits<std::size_t>::max()
                / static_cast<std::size_t>(dimension)) {
            throw std::runtime_error(
                "Checkpoint FFT grid size exceeds the addressable range."
            );
        }
        result *= static_cast<std::size_t>(dimension);
    }
    if (result >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Checkpoint FFT grid exceeds the supported integer size."
        );
    }
    return result;
}

void validate_checkpoint_contents(const SCFCheckpoint& checkpoint) {
    if (checkpoint.format_version != current_checkpoint_version &&
        checkpoint.format_version != legacy_lda_checkpoint_version) {
        throw std::runtime_error(
            "Unsupported SCF checkpoint format version "
            + std::to_string(checkpoint.format_version) + "."
        );
    }
    if (checkpoint.structure.atoms.empty() ||
        checkpoint.structure.species_order.empty() ||
        !checkpoint.structure.lattice_bohr.allFinite() ||
        std::abs(checkpoint.structure.lattice_bohr.determinant())
            < 1.0e-14) {
        throw std::runtime_error(
            "The SCF checkpoint contains an invalid structure."
        );
    }
    if (!std::isfinite(checkpoint.ecut_hartree) ||
        checkpoint.ecut_hartree <= 0.0 ||
        !std::isfinite(checkpoint.nelect) ||
        checkpoint.nelect <= 0.0 ||
        !std::isfinite(checkpoint.ewald_width_bohr) ||
        checkpoint.ewald_width_bohr <= 0.0 ||
        !std::isfinite(checkpoint.fermi_energy_ha) ||
        (checkpoint.nspin != 1 && checkpoint.nspin != 2)) {
        throw std::runtime_error(
            "The SCF checkpoint contains invalid electronic metadata."
        );
    }
    if (checkpoint.xc_functional != XCFunctional::ExchangeOnly &&
        checkpoint.xc_functional != XCFunctional::PerdewZunger &&
        checkpoint.xc_functional !=
            XCFunctional::PerdewBurkeErnzerhof) {
        throw std::runtime_error(
            "The SCF checkpoint contains an unknown XC functional."
        );
    }
    if (static_cast<int>(checkpoint.spin_densities.size()) !=
        checkpoint.nspin) {
        throw std::runtime_error(
            "The SCF checkpoint spin-density count does not match nspin."
        );
    }
    const std::size_t grid_size =
        checked_grid_size(checkpoint.fft_grid);
    for (const std::vector<double>& density :
         checkpoint.spin_densities) {
        if (density.size() != grid_size) {
            throw std::runtime_error(
                "An SCF checkpoint density does not match the FFT grid."
            );
        }
        for (double value : density) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "An SCF checkpoint density contains a non-finite value."
                );
            }
        }
    }
    for (const std::string& species :
         checkpoint.structure.species_order) {
        if (checkpoint.pseudopotentials.count(species) != 1) {
            throw std::runtime_error(
                "The SCF checkpoint has no unique pseudopotential "
                "fingerprint for " + species + "."
            );
        }
    }
}

} // namespace

PseudopotentialFingerprint fingerprint_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Cannot open pseudopotential for fingerprinting: " + path
        );
    }

    constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
    constexpr std::uint64_t fnv_prime = 1099511628211ULL;
    PseudopotentialFingerprint result;
    result.content_hash = fnv_offset;
    char buffer[8192];
    while (input) {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            result.content_hash ^=
                static_cast<unsigned char>(buffer[index]);
            result.content_hash *= fnv_prime;
        }
        result.size_bytes += static_cast<std::uint64_t>(count);
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "Failed while reading pseudopotential for fingerprinting: "
            + path
        );
    }
    return result;
}

SCFCheckpoint make_scf_checkpoint(
    const AtomicStructure& structure,
    const CalculationConfig& config,
    const SinglePointResult& result) {

    if (!result.converged) {
        throw std::runtime_error(
            "Cannot checkpoint an unconverged SCF calculation."
        );
    }

    SCFCheckpoint checkpoint;
    checkpoint.format_version = current_checkpoint_version;
    checkpoint.structure = structure;
    checkpoint.ecut_hartree = result.ecut_hartree;
    checkpoint.fft_grid = result.fft_grid;
    checkpoint.nspin = result.options_used.nspin;
    checkpoint.nelect = result.options_used.nelec;
    checkpoint.xc_functional = result.options_used.xc_functional;
    checkpoint.ewald_width_bohr = config.ewald_width_bohr;
    checkpoint.fermi_energy_ha = result.scf.occupations.mu;
    checkpoint.spin_densities = result.scf.spin_densities;

    for (const std::string& species : structure.species_order) {
        const auto mapping =
            config.pseudopotential_paths.find(species);
        if (mapping == config.pseudopotential_paths.end()) {
            throw std::runtime_error(
                "Cannot checkpoint without a pseudopotential mapping for "
                + species + "."
            );
        }
        checkpoint.pseudopotentials.emplace(
            species, fingerprint_file(mapping->second)
        );
    }
    validate_checkpoint_contents(checkpoint);
    return checkpoint;
}

void write_scf_checkpoint(
    const std::string& path,
    const SCFCheckpoint& checkpoint) {

    if (path.empty()) {
        throw std::runtime_error("The SCF checkpoint output path is empty.");
    }
    validate_checkpoint_contents(checkpoint);

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create SCF checkpoint output: " + path
        );
    }
    output << checkpoint_magic << " "
        << current_checkpoint_version << "\n"
        << std::scientific << std::setprecision(17)
        << "ecut_hartree " << checkpoint.ecut_hartree << "\n"
        << "fft_grid " << checkpoint.fft_grid[0] << " "
        << checkpoint.fft_grid[1] << " "
        << checkpoint.fft_grid[2] << "\n"
        << "nspin " << checkpoint.nspin << "\n"
        << "nelect " << checkpoint.nelect << "\n"
        << "xc_functional "
        << static_cast<int>(checkpoint.xc_functional) << "\n"
        << "ewald_width_bohr " << checkpoint.ewald_width_bohr << "\n"
        << "fermi_energy_ha " << checkpoint.fermi_energy_ha << "\n"
        << "lattice_bohr";
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            output << " "
                << checkpoint.structure.lattice_bohr(row, column);
        }
    }
    output << "\n"
        << "species " << checkpoint.structure.species_order.size();
    for (const std::string& species :
         checkpoint.structure.species_order) {
        output << " " << std::quoted(species);
    }
    output << "\n"
        << "atoms " << checkpoint.structure.atoms.size() << "\n";
    for (const StructureAtom& atom : checkpoint.structure.atoms) {
        output << "atom " << std::quoted(atom.element) << " "
            << atom.frac_position[0] << " "
            << atom.frac_position[1] << " "
            << atom.frac_position[2] << "\n";
    }
    output << "pseudopotentials "
        << checkpoint.pseudopotentials.size() << "\n";
    for (const auto& item : checkpoint.pseudopotentials) {
        output << "pseudo " << std::quoted(item.first) << " "
            << item.second.size_bytes << " "
            << std::hex << item.second.content_hash
            << std::dec << "\n";
    }

    const std::size_t grid_size =
        checked_grid_size(checkpoint.fft_grid);
    output << "density_points " << grid_size << "\n";
    for (int spin = 0; spin < checkpoint.nspin; ++spin) {
        output << "spin_density " << spin << "\n";
        const std::vector<double>& density =
            checkpoint.spin_densities[spin];
        for (std::size_t point = 0; point < density.size(); ++point) {
            output << density[point];
            if ((point + 1) % 4 == 0 ||
                point + 1 == density.size()) {
                output << "\n";
            } else {
                output << " ";
            }
        }
    }
    output << "end\n";
    if (!output) {
        throw std::runtime_error(
            "Failed while writing SCF checkpoint: " + path
        );
    }
}

SCFCheckpoint read_scf_checkpoint(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Cannot open SCF checkpoint input: " + path
        );
    }

    SCFCheckpoint checkpoint;
    std::string magic;
    if (!(input >> magic >> checkpoint.format_version) ||
        magic != checkpoint_magic) {
        throw std::runtime_error(
            path + ": not a PWDFT SCF checkpoint."
        );
    }
    if (checkpoint.format_version != current_checkpoint_version &&
        checkpoint.format_version != legacy_lda_checkpoint_version) {
        throw std::runtime_error(
            path + ": unsupported SCF checkpoint format version "
            + std::to_string(checkpoint.format_version) + "."
        );
    }

    require_key(input, "ecut_hartree", path);
    input >> checkpoint.ecut_hartree;
    require_key(input, "fft_grid", path);
    input >> checkpoint.fft_grid[0]
        >> checkpoint.fft_grid[1]
        >> checkpoint.fft_grid[2];
    require_key(input, "nspin", path);
    input >> checkpoint.nspin;
    if (checkpoint.nspin != 1 && checkpoint.nspin != 2) {
        throw std::runtime_error(
            path + ": nspin must be one or two."
        );
    }
    require_key(input, "nelect", path);
    input >> checkpoint.nelect;
    require_key(
        input,
        checkpoint.format_version == legacy_lda_checkpoint_version
            ? "lda_functional"
            : "xc_functional",
        path
    );
    int functional = 0;
    input >> functional;
    checkpoint.xc_functional =
        static_cast<XCFunctional>(functional);
    require_key(input, "ewald_width_bohr", path);
    input >> checkpoint.ewald_width_bohr;
    require_key(input, "fermi_energy_ha", path);
    input >> checkpoint.fermi_energy_ha;
    require_key(input, "lattice_bohr", path);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            input >> checkpoint.structure.lattice_bohr(row, column);
        }
    }

    require_key(input, "species", path);
    std::size_t species_count = 0;
    input >> species_count;
    checkpoint.structure.species_order.resize(species_count);
    for (std::string& species :
         checkpoint.structure.species_order) {
        input >> std::quoted(species);
    }

    require_key(input, "atoms", path);
    std::size_t atom_count = 0;
    input >> atom_count;
    checkpoint.structure.atoms.resize(atom_count);
    for (StructureAtom& atom : checkpoint.structure.atoms) {
        require_key(input, "atom", path);
        input >> std::quoted(atom.element)
            >> atom.frac_position[0]
            >> atom.frac_position[1]
            >> atom.frac_position[2];
    }

    require_key(input, "pseudopotentials", path);
    std::size_t pseudopotential_count = 0;
    input >> pseudopotential_count;
    for (std::size_t index = 0;
         index < pseudopotential_count;
         ++index) {
        require_key(input, "pseudo", path);
        std::string element;
        PseudopotentialFingerprint fingerprint;
        input >> std::quoted(element)
            >> fingerprint.size_bytes
            >> std::hex >> fingerprint.content_hash >> std::dec;
        if (!checkpoint.pseudopotentials.emplace(
                element, fingerprint
            ).second) {
            throw std::runtime_error(
                path + ": duplicate pseudopotential fingerprint for "
                + element + "."
            );
        }
    }

    require_key(input, "density_points", path);
    std::size_t density_points = 0;
    input >> density_points;
    if (density_points != checked_grid_size(checkpoint.fft_grid)) {
        throw std::runtime_error(
            path + ": density_points does not match fft_grid."
        );
    }
    checkpoint.spin_densities.assign(
        checkpoint.nspin,
        std::vector<double>(density_points, 0.0)
    );
    for (int spin = 0; spin < checkpoint.nspin; ++spin) {
        require_key(input, "spin_density", path);
        int input_spin = -1;
        input >> input_spin;
        if (input_spin != spin) {
            throw std::runtime_error(
                path + ": spin-density channels are out of order."
            );
        }
        for (double& value : checkpoint.spin_densities[spin]) {
            input >> value;
        }
    }
    require_key(input, "end", path);
    if (!input) {
        throw std::runtime_error(
            path + ": truncated or invalid SCF checkpoint."
        );
    }

    validate_checkpoint_contents(checkpoint);
    return checkpoint;
}

void validate_scf_checkpoint(
    const SCFCheckpoint& checkpoint,
    const AtomicStructure& structure,
    const CalculationConfig& config,
    double resolved_nelect) {

    validate_checkpoint_contents(checkpoint);
    if (structure.species_order !=
        checkpoint.structure.species_order ||
        structure.atoms.size() !=
            checkpoint.structure.atoms.size()) {
        throw std::runtime_error(
            "SCF checkpoint structure species or atom count differs from "
            "the NSCF structure."
        );
    }
    if ((structure.lattice_bohr -
         checkpoint.structure.lattice_bohr).cwiseAbs().maxCoeff() >
        metadata_tolerance) {
        throw std::runtime_error(
            "SCF checkpoint lattice differs from the NSCF lattice."
        );
    }
    for (int atom = 0;
         atom < static_cast<int>(structure.atoms.size());
         ++atom) {
        if (structure.atoms[atom].element !=
            checkpoint.structure.atoms[atom].element) {
            throw std::runtime_error(
                "SCF checkpoint atom ordering differs from the NSCF "
                "structure."
            );
        }
        const Eigen::Vector3d difference =
            structure.atoms[atom].frac_position
            - checkpoint.structure.atoms[atom].frac_position;
        if ((difference.array() -
             difference.array().round()).abs().maxCoeff() >
            metadata_tolerance) {
            throw std::runtime_error(
                "SCF checkpoint atomic coordinates differ from the NSCF "
                "structure."
            );
        }
    }
    if (config.scf.nspin != checkpoint.nspin) {
        throw std::runtime_error(
            "SCF checkpoint nspin differs from the NSCF input."
        );
    }
    if (!nearly_equal(resolved_nelect, checkpoint.nelect)) {
        throw std::runtime_error(
            "SCF checkpoint electron count differs from the NSCF input."
        );
    }
    if (config.scf.xc_functional != checkpoint.xc_functional) {
        throw std::runtime_error(
            "SCF checkpoint XC functional differs from the NSCF input."
        );
    }
    if (!nearly_equal(
            config.ewald_width_bohr,
            checkpoint.ewald_width_bohr)) {
        throw std::runtime_error(
            "SCF checkpoint Ewald width differs from the NSCF input."
        );
    }
    if (config.ecut_hartree > 0.0 &&
        !nearly_equal(
            config.ecut_hartree, checkpoint.ecut_hartree)) {
        throw std::runtime_error(
            "SCF checkpoint cutoff differs from the explicit NSCF cutoff."
        );
    }
    const bool explicit_fft =
        config.fft_grid[0] > 0 &&
        config.fft_grid[1] > 0 &&
        config.fft_grid[2] > 0;
    if (explicit_fft && config.fft_grid != checkpoint.fft_grid) {
        throw std::runtime_error(
            "SCF checkpoint FFT grid differs from the explicit NSCF grid."
        );
    }

    for (const std::string& species : structure.species_order) {
        const auto mapping =
            config.pseudopotential_paths.find(species);
        if (mapping == config.pseudopotential_paths.end()) {
            throw std::runtime_error(
                "No NSCF pseudopotential mapping was provided for "
                + species + "."
            );
        }
        const PseudopotentialFingerprint current =
            fingerprint_file(mapping->second);
        const auto stored =
            checkpoint.pseudopotentials.find(species);
        if (stored == checkpoint.pseudopotentials.end() ||
            stored->second.size_bytes != current.size_bytes ||
            stored->second.content_hash != current.content_hash) {
            throw std::runtime_error(
                "SCF checkpoint pseudopotential content differs for "
                + species + "."
            );
        }
    }
}
