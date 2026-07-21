#include "input.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c); }
    );
    if (first == value.end()) {
        return {};
    }
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c); }
    ).base();
    return std::string(first, last);
}

std::string lowercase(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );
    return value;
}

std::vector<std::string> tokens(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> result;
    for (std::string token; input >> token;) {
        result.push_back(token);
    }
    return result;
}

std::string read_required_line(
    std::istream& input,
    const std::string& path,
    int& line_number) {

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error(
            path + ": unexpected end of file after line "
            + std::to_string(line_number)
        );
    }
    ++line_number;
    return line;
}

double parse_double(
    const std::string& token,
    const std::string& context) {

    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(context + ": expected a real number.");
    }
    if (consumed != token.size() || !std::isfinite(value)) {
        throw std::runtime_error(context + ": invalid real number '" + token + "'.");
    }
    return value;
}

int parse_integer(
    const std::string& token,
    const std::string& context) {

    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(context + ": expected an integer.");
    }
    if (consumed != token.size()) {
        throw std::runtime_error(context + ": invalid integer '" + token + "'.");
    }
    return value;
}

Eigen::Vector3d parse_vector3(
    const std::string& line,
    const std::string& context) {

    const auto fields = tokens(line);
    if (fields.size() < 3) {
        throw std::runtime_error(context + ": expected three coordinates.");
    }
    return Eigen::Vector3d(
        parse_double(fields[0], context),
        parse_double(fields[1], context),
        parse_double(fields[2], context)
    );
}

bool parse_selective_flag(
    const std::string& token,
    const std::string& context) {

    if (token.empty()) {
        throw std::runtime_error(context + ": empty selective-dynamics flag.");
    }
    const char flag = static_cast<char>(std::tolower(
        static_cast<unsigned char>(token.front())
    ));
    if (flag == 't') {
        return true;
    }
    if (flag == 'f') {
        return false;
    }
    throw std::runtime_error(
        context + ": selective-dynamics flags must be T or F."
    );
}

std::filesystem::path resolve_relative_path(
    const std::filesystem::path& base_directory,
    const std::string& value) {

    std::filesystem::path path(value);
    if (path.is_relative()) {
        path = base_directory / path;
    }
    return path.lexically_normal();
}

std::string remove_config_comment(const std::string& line) {
    const auto hash = line.find('#');
    const auto bang = line.find('!');
    const auto cut = std::min(
        hash == std::string::npos ? line.size() : hash,
        bang == std::string::npos ? line.size() : bang
    );
    return trim(line.substr(0, cut));
}

SCFVerbosity parse_verbosity(const std::string& value) {
    const std::string mode = lowercase(trim(value));
    if (mode == "silent") {
        return SCFVerbosity::Silent;
    }
    if (mode == "compact") {
        return SCFVerbosity::Compact;
    }
    if (mode == "detailed") {
        return SCFVerbosity::Detailed;
    }
    throw std::runtime_error(
        "verbosity must be silent, compact, or detailed."
    );
}

OccupationMode parse_occupation_mode(const std::string& value) {
    std::string mode = lowercase(trim(value));
    std::replace(mode.begin(), mode.end(), '-', '_');
    if (mode == "fermi_dirac" || mode == "smearing") {
        return OccupationMode::FermiDirac;
    }
    if (mode == "zero_t" || mode == "degeneracy_aware_zero_t") {
        return OccupationMode::DegeneracyAwareZeroT;
    }
    if (mode == "fixed") {
        return OccupationMode::Fixed;
    }
    throw std::runtime_error(
        "occupations must be fermi_dirac, zero_t, or fixed."
    );
}

void require_positive(double value, const std::string& name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(name + " must be positive and finite.");
    }
}

} // namespace

KPointSet make_uniform_kpoint_mesh(
    const std::array<int, 3>& mesh,
    bool gamma_centered) {

    long long point_count_wide = 1;
    for (int direction = 0; direction < 3; ++direction) {
        if (mesh[direction] <= 0) {
            throw std::runtime_error(
                "K-point mesh dimensions must be positive."
            );
        }
        point_count_wide *= mesh[direction];
        if (point_count_wide > std::numeric_limits<int>::max()) {
            throw std::runtime_error("K-point mesh is too large.");
        }
    }
    const int point_count = static_cast<int>(point_count_wide);

    KPointSet result;
    result.points.clear();
    result.points.reserve(point_count);
    result.description = gamma_centered
        ? "Gamma-centered"
        : "Monkhorst-Pack";

    auto coordinate = [gamma_centered](int index, int count) {
        if (!gamma_centered) {
            return (2.0 * static_cast<double>(index) + 1.0
                    - static_cast<double>(count))
                / (2.0 * static_cast<double>(count));
        }
        double value = static_cast<double>(index)
            / static_cast<double>(count);
        if (value >= 0.5) {
            value -= 1.0;
        }
        return value;
    };

    const double weight = 1.0 / static_cast<double>(point_count);
    for (int i = 0; i < mesh[0]; ++i) {
        for (int j = 0; j < mesh[1]; ++j) {
            for (int k = 0; k < mesh[2]; ++k) {
                KPoint point;
                point.frac_position = Eigen::Vector3d(
                    coordinate(i, mesh[0]),
                    coordinate(j, mesh[1]),
                    coordinate(k, mesh[2])
                );
                point.weight = weight;
                result.points.push_back(point);
            }
        }
    }
    return result;
}

void normalize_kpoint_weights(KPointSet& kpoints) {
    if (kpoints.points.empty()) {
        throw std::runtime_error("At least one k point is required.");
    }
    double weight_sum = 0.0;
    for (const KPoint& point : kpoints.points) {
        if (!point.frac_position.allFinite()) {
            throw std::runtime_error("K-point coordinates must be finite.");
        }
        if (!std::isfinite(point.weight) || point.weight <= 0.0) {
            throw std::runtime_error("K-point weights must be positive and finite.");
        }
        weight_sum += point.weight;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
        throw std::runtime_error("K-point weight sum is invalid.");
    }
    for (KPoint& point : kpoints.points) {
        point.weight /= weight_sum;
    }
}

AtomicStructure read_poscar(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open POSCAR file: " + path);
    }

    AtomicStructure structure;
    int line_number = 0;
    structure.comment = trim(read_required_line(input, path, line_number));

    const auto scale_fields = tokens(
        read_required_line(input, path, line_number)
    );
    if (scale_fields.size() != 1) {
        throw std::runtime_error(
            path + ": line 2 must contain one POSCAR scale factor; "
            "three-factor scaling is not supported yet."
        );
    }
    const double input_scale = parse_double(
        scale_fields[0], path + ": POSCAR scale factor"
    );
    if (input_scale == 0.0) {
        throw std::runtime_error(path + ": POSCAR scale factor cannot be zero.");
    }

    Eigen::Matrix3d raw_lattice;
    for (int direction = 0; direction < 3; ++direction) {
        raw_lattice.col(direction) = parse_vector3(
            read_required_line(input, path, line_number),
            path + ": lattice vector on line " + std::to_string(line_number)
        );
    }
    const double raw_volume = std::abs(raw_lattice.determinant());
    if (!std::isfinite(raw_volume) || raw_volume < 1.0e-14) {
        throw std::runtime_error(path + ": POSCAR lattice is singular.");
    }

    double scale = input_scale;
    if (input_scale < 0.0) {
        const double target_volume_angstrom3 = -input_scale;
        scale = std::cbrt(target_volume_angstrom3 / raw_volume);
    }
    const Eigen::Matrix3d lattice_angstrom = scale * raw_lattice;
    structure.lattice_bohr = ANGSTROM_TO_BOHR * lattice_angstrom;

    structure.species_order = tokens(
        read_required_line(input, path, line_number)
    );
    if (structure.species_order.empty()) {
        throw std::runtime_error(path + ": POSCAR element-symbol line is empty.");
    }
    for (const std::string& symbol : structure.species_order) {
        if (!symbol.empty() && std::isdigit(
                static_cast<unsigned char>(symbol.front()))) {
            throw std::runtime_error(
                path + ": VASP 4 POSCAR without element symbols is unsupported; "
                "provide a VASP 5-style element-symbol line."
            );
        }
    }
    const std::set<std::string> unique_species(
        structure.species_order.begin(), structure.species_order.end()
    );
    if (unique_species.size() != structure.species_order.size()) {
        throw std::runtime_error(
            path + ": each element may appear only once in the POSCAR species line."
        );
    }

    const auto count_fields = tokens(
        read_required_line(input, path, line_number)
    );
    if (count_fields.size() != structure.species_order.size()) {
        throw std::runtime_error(
            path + ": POSCAR species and count lines have different lengths."
        );
    }
    std::vector<int> counts(count_fields.size(), 0);
    int total_atoms = 0;
    for (int i = 0; i < static_cast<int>(count_fields.size()); ++i) {
        counts[i] = parse_integer(
            count_fields[i], path + ": atom count on line "
                + std::to_string(line_number)
        );
        if (counts[i] < 0) {
            throw std::runtime_error(path + ": POSCAR atom counts cannot be negative.");
        }
        total_atoms += counts[i];
    }
    if (total_atoms <= 0) {
        throw std::runtime_error(path + ": POSCAR contains no atoms.");
    }

    std::string coordinate_line = trim(
        read_required_line(input, path, line_number)
    );
    bool selective_dynamics = false;
    if (!coordinate_line.empty() &&
        std::tolower(static_cast<unsigned char>(coordinate_line.front())) == 's') {
        selective_dynamics = true;
        coordinate_line = trim(read_required_line(input, path, line_number));
    }
    if (coordinate_line.empty()) {
        throw std::runtime_error(path + ": missing POSCAR coordinate mode.");
    }
    const char coordinate_mode = static_cast<char>(std::tolower(
        static_cast<unsigned char>(coordinate_line.front())
    ));
    const bool direct = coordinate_mode == 'd';
    const bool cartesian = coordinate_mode == 'c' || coordinate_mode == 'k';
    if (!direct && !cartesian) {
        throw std::runtime_error(
            path + ": POSCAR coordinate mode must be Direct or Cartesian."
        );
    }

    structure.atoms.reserve(total_atoms);
    for (int ispecies = 0;
         ispecies < static_cast<int>(structure.species_order.size());
         ++ispecies) {
        for (int iatom = 0; iatom < counts[ispecies]; ++iatom) {
            const std::string atom_line =
                read_required_line(input, path, line_number);
            const auto fields = tokens(atom_line);
            if (fields.size() < 3) {
                throw std::runtime_error(
                    path + ": atom line " + std::to_string(line_number)
                    + " needs three coordinates."
                );
            }
            const Eigen::Vector3d coordinates(
                parse_double(fields[0], path + ": atom coordinate"),
                parse_double(fields[1], path + ": atom coordinate"),
                parse_double(fields[2], path + ": atom coordinate")
            );

            StructureAtom atom;
            atom.element = structure.species_order[ispecies];
            atom.frac_position = direct
                ? coordinates
                : lattice_angstrom.inverse() * (scale * coordinates);
            if (selective_dynamics) {
                if (fields.size() < 6) {
                    throw std::runtime_error(
                        path + ": atom line " + std::to_string(line_number)
                        + " needs three selective-dynamics flags."
                    );
                }
                for (int direction = 0; direction < 3; ++direction) {
                    atom.movable[direction] = parse_selective_flag(
                        fields[3 + direction],
                        path + ": atom line " + std::to_string(line_number)
                    );
                }
            }
            structure.atoms.push_back(atom);
        }
    }

    return structure;
}

CalculationConfig read_calculation_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open calculation input: " + path);
    }

    CalculationConfig config;
    const std::filesystem::path input_path =
        std::filesystem::absolute(path).lexically_normal();
    const std::filesystem::path base_directory = input_path.parent_path();
    config.source_path = input_path.string();
    config.structure_path =
        resolve_relative_path(base_directory, "POSCAR").string();
    config.scf.occupation_mode = OccupationMode::DegeneracyAwareZeroT;
    config.scf.smearing_sigma = 0.0;
    bool explicit_kpoints = false;

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = remove_config_comment(line);
        if (line.empty()) {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + " must use key = value syntax."
            );
        }
        const std::string key = lowercase(trim(line.substr(0, equals)));
        const std::string value = trim(line.substr(equals + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                path + ": empty key or value on line "
                + std::to_string(line_number) + "."
            );
        }

        if (key == "calculation") {
            if (lowercase(value) != "scf") {
                throw std::runtime_error(
                    "Only calculation = scf is implemented."
                );
            }
        } else if (key == "structure") {
            config.structure_path =
                resolve_relative_path(base_directory, value).string();
        } else if (key == "pseudo") {
            const auto fields = tokens(value);
            if (fields.size() != 2) {
                throw std::runtime_error(
                    "pseudo requires: pseudo = ELEMENT PATH."
                );
            }
            if (!config.pseudopotential_paths.emplace(
                    fields[0],
                    resolve_relative_path(base_directory, fields[1]).string()
                ).second) {
                throw std::runtime_error(
                    "Duplicate pseudopotential mapping for " + fields[0] + "."
                );
            }
        } else if (key == "ecut_ha") {
            if (lowercase(value) == "auto") {
                config.ecut_hartree = 0.0;
            } else {
                config.ecut_hartree = parse_double(value, "ecut_ha");
                require_positive(config.ecut_hartree, "ecut_ha");
            }
        } else if (key == "fft_grid") {
            const auto fields = tokens(value);
            if (fields.size() != 3) {
                throw std::runtime_error("fft_grid requires three integers.");
            }
            for (int direction = 0; direction < 3; ++direction) {
                config.fft_grid[direction] = parse_integer(
                    fields[direction], "fft_grid"
                );
                if (config.fft_grid[direction] < 4) {
                    throw std::runtime_error(
                        "Each fft_grid dimension must be at least 4."
                    );
                }
            }
        } else if (key == "ewald_width_bohr") {
            config.ewald_width_bohr = parse_double(value, key);
            require_positive(config.ewald_width_bohr, key);
        } else if (key == "nelect") {
            config.nelect_auto = lowercase(value) == "auto";
            if (!config.nelect_auto) {
                config.scf.nelec = parse_double(value, key);
                require_positive(config.scf.nelec, key);
            }
        } else if (key == "nbands") {
            config.nbands_auto = lowercase(value) == "auto";
            if (!config.nbands_auto) {
                config.scf.nbands = parse_integer(value, key);
                if (config.scf.nbands <= 0) {
                    throw std::runtime_error("nbands must be positive.");
                }
            }
        } else if (key == "occupations") {
            config.scf.occupation_mode = parse_occupation_mode(value);
        } else if (key == "fixed_occupations") {
            config.scf.fixed_occupations.clear();
            for (const std::string& field : tokens(value)) {
                config.scf.fixed_occupations.push_back(
                    parse_double(field, key)
                );
            }
        } else if (key == "smearing_ev") {
            const double smearing_ev = parse_double(value, key);
            if (smearing_ev < 0.0) {
                throw std::runtime_error("smearing_ev cannot be negative.");
            }
            config.scf.smearing_sigma = smearing_ev / HARTREE_TO_EV;
        } else if (key == "scf_max_iterations") {
            config.scf.max_iterations = parse_integer(value, key);
        } else if (key == "density_tolerance") {
            config.scf.density_tolerance = parse_double(value, key);
        } else if (key == "energy_tolerance_ha") {
            config.scf.energy_tolerance = parse_double(value, key);
        } else if (key == "eigensolver_max_iterations") {
            config.scf.eigensolver_max_iterations = parse_integer(value, key);
        } else if (key == "eigensolver_max_subspace") {
            config.scf.eigensolver_max_subspace = parse_integer(value, key);
        } else if (key == "eigensolver_tolerance_ha") {
            config.scf.eigensolver_tolerance = parse_double(value, key);
        } else if (key == "eigensolver_denom_floor_ha") {
            config.scf.eigensolver_denom_floor = parse_double(value, key);
        } else if (key == "mixing_alpha") {
            config.scf.mixing_alpha = parse_double(value, key);
        } else if (key == "pulay_max_history") {
            config.scf.pulay_max_history = parse_integer(value, key);
        } else if (key == "pulay_min_history") {
            config.scf.pulay_min_history = parse_integer(value, key);
        } else if (key == "pulay_regularization") {
            config.scf.pulay_regularization = parse_double(value, key);
        } else if (key == "degeneracy_tolerance_ha") {
            config.scf.degeneracy_tolerance = parse_double(value, key);
        } else if (key == "verbosity") {
            config.scf.verbosity = parse_verbosity(value);
        } else if (key == "bands_to_print") {
            config.scf.bands_to_print = parse_integer(value, key);
        } else if (key == "band_print_interval") {
            config.scf.band_print_interval = parse_integer(value, key);
        } else if (key == "kpoints") {
            const auto fields = tokens(lowercase(value));
            if (fields.size() == 1 && fields[0] == "gamma") {
                config.kpoints = KPointSet{};
                explicit_kpoints = false;
            } else if (fields.size() == 1 && fields[0] == "explicit") {
                config.kpoints.points.clear();
                config.kpoints.description = "Explicit";
                explicit_kpoints = true;
            } else if (fields.size() == 4) {
                const bool gamma_centered =
                    fields[0] == "gamma"
                    || fields[0] == "gamma_centered";
                const bool monkhorst_pack =
                    fields[0] == "monkhorst_pack"
                    || fields[0] == "mp";
                if (!gamma_centered && !monkhorst_pack) {
                    throw std::runtime_error(
                        "kpoints mesh must be gamma, gamma_centered, "
                        "monkhorst_pack, or mp."
                    );
                }
                std::array<int, 3> mesh;
                for (int direction = 0; direction < 3; ++direction) {
                    mesh[direction] = parse_integer(
                        fields[direction + 1], "kpoints"
                    );
                }
                config.kpoints = make_uniform_kpoint_mesh(
                    mesh, gamma_centered
                );
                explicit_kpoints = false;
            } else {
                throw std::runtime_error(
                    "kpoints requires gamma, explicit, or a scheme followed "
                    "by three mesh dimensions."
                );
            }
        } else if (key == "kpoint") {
            if (!explicit_kpoints) {
                throw std::runtime_error(
                    "kpoint lines require an earlier kpoints = explicit line."
                );
            }
            const auto fields = tokens(value);
            if (fields.size() != 4) {
                throw std::runtime_error(
                    "kpoint requires three reciprocal fractional coordinates "
                    "and one positive weight."
                );
            }
            KPoint point;
            for (int direction = 0; direction < 3; ++direction) {
                point.frac_position[direction] = parse_double(
                    fields[direction], "kpoint"
                );
            }
            point.weight = parse_double(fields[3], "kpoint weight");
            config.kpoints.points.push_back(point);
        } else if (key == "xc") {
            std::string functional = lowercase(value);
            std::replace(functional.begin(), functional.end(), '-', '_');
            if (functional != "pz_lda" && functional != "lda_pz") {
                throw std::runtime_error(
                    "Only xc = pz_lda is implemented."
                );
            }
        } else {
            throw std::runtime_error(
                path + ": unknown input key '" + key + "' on line "
                + std::to_string(line_number) + "."
            );
        }
    }

    if (config.fft_grid[0] == 0 ||
        config.fft_grid[1] == 0 ||
        config.fft_grid[2] == 0) {
        throw std::runtime_error(
            "fft_grid must be specified explicitly in the calculation input."
        );
    }
    if (config.pseudopotential_paths.empty()) {
        throw std::runtime_error(
            "At least one pseudo = ELEMENT PATH mapping is required."
        );
    }
    if (config.scf.max_iterations <= 0 ||
        config.scf.eigensolver_max_iterations <= 0) {
        throw std::runtime_error(
            "SCF and eigensolver iteration limits must be positive."
        );
    }
    if (config.scf.density_tolerance <= 0.0 ||
        config.scf.energy_tolerance <= 0.0 ||
        config.scf.eigensolver_tolerance <= 0.0 ||
        config.scf.eigensolver_denom_floor <= 0.0 ||
        config.scf.degeneracy_tolerance <= 0.0) {
        throw std::runtime_error("SCF tolerances must be positive.");
    }
    if (config.scf.eigensolver_max_subspace < 0) {
        throw std::runtime_error(
            "eigensolver_max_subspace cannot be negative."
        );
    }
    if (!std::isfinite(config.scf.mixing_alpha) ||
        config.scf.mixing_alpha <= 0.0 ||
        config.scf.mixing_alpha > 1.0) {
        throw std::runtime_error("mixing_alpha must be in (0, 1].");
    }
    if (config.scf.pulay_min_history <= 0 ||
        config.scf.pulay_max_history < config.scf.pulay_min_history) {
        throw std::runtime_error(
            "Pulay history sizes require 0 < min <= max."
        );
    }
    if (!std::isfinite(config.scf.pulay_regularization) ||
        config.scf.pulay_regularization < 0.0) {
        throw std::runtime_error(
            "pulay_regularization must be finite and non-negative."
        );
    }
    if (config.scf.bands_to_print < 0 ||
        config.scf.band_print_interval < 0) {
        throw std::runtime_error(
            "Band-output counts and intervals cannot be negative."
        );
    }
    if (config.scf.occupation_mode == OccupationMode::FermiDirac &&
        config.scf.smearing_sigma <= 0.0) {
        throw std::runtime_error(
            "Fermi-Dirac occupations require smearing_ev > 0."
        );
    }
    if (config.scf.occupation_mode == OccupationMode::Fixed &&
        config.scf.fixed_occupations.empty()) {
        throw std::runtime_error(
            "Fixed occupations require fixed_occupations values."
        );
    }
    if (config.scf.occupation_mode == OccupationMode::Fixed) {
        if (config.nbands_auto) {
            throw std::runtime_error(
                "Fixed occupations require an explicit nbands value."
            );
        }
        if (static_cast<int>(config.scf.fixed_occupations.size())
                != config.scf.nbands) {
            throw std::runtime_error(
                "fixed_occupations must contain exactly nbands values."
            );
        }
        for (double occupation : config.scf.fixed_occupations) {
            if (!std::isfinite(occupation) ||
                occupation < 0.0 || occupation > 2.0) {
                throw std::runtime_error(
                    "Each fixed occupation must be in [0, 2]."
                );
            }
        }
    }
    normalize_kpoint_weights(config.kpoints);
    return config;
}
