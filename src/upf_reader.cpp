#include "upf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

using Attributes = std::unordered_map<std::string, std::string>;

struct OpeningTag {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::string text;
};

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](unsigned char c) { return std::isspace(c); }
    );
    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](unsigned char c) { return std::isspace(c); }
    ).base();

    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string lowercase(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

std::runtime_error parse_error(
    const std::string& path,
    const std::string& message) {

    return std::runtime_error("UPF parse error in '" + path + "': " + message);
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw parse_error(path, "cannot open file");
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw parse_error(path, "failed while reading file");
    }
    return contents.str();
}

OpeningTag find_opening_tag(
    const std::string& document,
    const std::string& name,
    const std::string& path) {

    const std::string needle = "<" + name;
    std::size_t begin = 0;

    while ((begin = document.find(needle, begin)) != std::string::npos) {
        const std::size_t after_name = begin + needle.size();
        if (after_name >= document.size() ||
            std::isspace(static_cast<unsigned char>(document[after_name])) ||
            document[after_name] == '>' || document[after_name] == '/') {
            const std::size_t end = document.find('>', after_name);
            if (end == std::string::npos) {
                throw parse_error(path, "unterminated <" + name + "> tag");
            }
            return {begin, end, document.substr(begin, end - begin + 1)};
        }
        begin = after_name;
    }

    throw parse_error(path, "missing <" + name + "> tag");
}

std::string find_field_contents(
    const std::string& document,
    const std::string& name,
    const std::string& path) {

    const OpeningTag opening = find_opening_tag(document, name, path);
    const std::string closing = "</" + name + ">";
    const std::size_t close_begin = document.find(closing, opening.end + 1);
    if (close_begin == std::string::npos) {
        throw parse_error(path, "missing " + closing + " tag");
    }
    return document.substr(opening.end + 1, close_begin - opening.end - 1);
}

Attributes parse_attributes(const OpeningTag& tag) {
    Attributes attributes;
    const std::regex pattern(R"upf(([A-Za-z0-9_]+)\s*=\s*"([^"]*)")upf");

    for (std::sregex_iterator it(tag.text.begin(), tag.text.end(), pattern), end;
         it != end;
         ++it) {
        attributes[(*it)[1].str()] = (*it)[2].str();
    }
    return attributes;
}

std::string required_attribute(
    const Attributes& attributes,
    const std::string& name,
    const std::string& tag_name,
    const std::string& path) {

    const auto it = attributes.find(name);
    if (it == attributes.end()) {
        throw parse_error(
            path,
            "missing attribute '" + name + "' in <" + tag_name + ">"
        );
    }
    return trim(it->second);
}

std::string required_one_of_attributes(
    const Attributes& attributes,
    const std::vector<std::string>& names,
    const std::string& tag_name,
    const std::string& path) {

    for (const std::string& name : names) {
        const auto it = attributes.find(name);
        if (it != attributes.end()) {
            return trim(it->second);
        }
    }
    std::string alternatives;
    for (const std::string& name : names) {
        if (!alternatives.empty()) {
            alternatives += "' or '";
        }
        alternatives += name;
    }
    throw parse_error(
        path,
        "missing attribute '" + alternatives + "' in <" + tag_name + ">"
    );
}

std::string optional_attribute(
    const Attributes& attributes,
    const std::string& name,
    const std::string& fallback = {}) {

    const auto it = attributes.find(name);
    return it == attributes.end() ? fallback : trim(it->second);
}

double parse_double_token(
    std::string token,
    const std::string& context,
    const std::string& path) {

    std::replace(token.begin(), token.end(), 'D', 'E');
    std::replace(token.begin(), token.end(), 'd', 'e');

    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() ||
        end != token.c_str() + token.size()) {
        throw parse_error(path, "invalid number '" + token + "' in " + context);
    }
    /*
     * Old converted UPFs can contain finite subnormal radial tails such as
     * 1e-315. std::stod reports ERANGE for those on some C libraries even
     * though retaining the subnormal value (or zero after underflow) is
     * harmless. Reject only a genuine non-finite overflow.
     */
    if (!std::isfinite(value)) {
        throw parse_error(path, "out-of-range number '" + token + "' in " + context);
    }
    return value;
}

int parse_int_token(
    const std::string& token,
    const std::string& context,
    const std::string& path) {

    try {
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size()) {
            throw parse_error(
                path,
                "invalid integer '" + token + "' in " + context
            );
        }
        return value;
    } catch (const std::invalid_argument&) {
        throw parse_error(
            path,
            "invalid integer '" + token + "' in " + context
        );
    } catch (const std::out_of_range&) {
        throw parse_error(
            path,
            "out-of-range integer '" + token + "' in " + context
        );
    }
}

int parse_int_attribute(
    const Attributes& attributes,
    const std::string& name,
    const std::string& tag_name,
    const std::string& path) {

    return parse_int_token(
        required_attribute(
            attributes, name, tag_name, path
        ),
        "<" + tag_name + ">",
        path
    );
}

double parse_double_attribute(
    const Attributes& attributes,
    const std::string& name,
    const std::string& tag_name,
    const std::string& path) {

    return parse_double_token(
        required_attribute(attributes, name, tag_name, path),
        "<" + tag_name + ">",
        path
    );
}

bool parse_bool_attribute(
    const Attributes& attributes,
    const std::string& name,
    const std::string& tag_name,
    const std::string& path) {

    const std::string value = lowercase(
        required_attribute(attributes, name, tag_name, path)
    );
    if (value == "true" || value == ".true." || value == ".t." ||
        value == "t" || value == "1") {
        return true;
    }
    if (value == "false" || value == ".false." || value == ".f." ||
        value == "f" || value == "0") {
        return false;
    }
    throw parse_error(
        path,
        "invalid boolean '" + value + "' in <" + tag_name + ">"
    );
}

std::vector<double> parse_numeric_field(
    std::string contents,
    const std::string& field_name,
    const std::string& path) {

    std::replace(contents.begin(), contents.end(), ',', ' ');
    std::istringstream input(contents);
    std::vector<double> values;
    std::string token;
    while (input >> token) {
        values.push_back(parse_double_token(token, "<" + field_name + ">", path));
    }
    return values;
}

void require_size(
    const std::vector<double>& values,
    int expected,
    const std::string& field_name,
    const std::string& path) {

    if (static_cast<int>(values.size()) != expected) {
        throw parse_error(
            path,
            "<" + field_name + "> has " + std::to_string(values.size()) +
            " values; expected " + std::to_string(expected)
        );
    }
}

void validate_supported_header(const UPFHeader& header, const std::string& path) {
    if (header.version.rfind("2.", 0) != 0) {
        throw parse_error(path, "only UPF v2 is supported");
    }
    if (header.pseudo_type != "NC") {
        throw parse_error(path, "only pseudo_type=NC is supported");
    }
    if (header.is_ultrasoft) {
        throw parse_error(path, "ultrasoft pseudopotentials are not supported");
    }
    if (header.is_paw) {
        throw parse_error(path, "PAW datasets are not supported");
    }
    if (header.has_spin_orbit) {
        throw parse_error(path, "spin-orbit projectors are not supported");
    }
    if (header.has_core_correction) {
        throw parse_error(path, "nonlinear core correction is not supported");
    }
    if (header.mesh_size <= 0) {
        throw parse_error(path, "mesh_size must be positive");
    }
    if (header.number_of_projectors < 0) {
        throw parse_error(path, "number_of_proj cannot be negative");
    }
    if (header.number_of_wavefunctions < 0) {
        throw parse_error(path, "number_of_wfc cannot be negative");
    }
}

} // namespace

double UPFData::dij(int i, int j) const {
    const int nprojectors = header.number_of_projectors;
    if (i < 0 || j < 0 || i >= nprojectors || j >= nprojectors) {
        throw std::runtime_error("UPF DIJ index out of range.");
    }
    return dij_ry[i * nprojectors + j];
}

UPFData read_nc_upf(const std::string& path) {
    const std::string document = read_text_file(path);
    UPFData data;
    data.source_path = path;

    const OpeningTag upf_tag = find_opening_tag(document, "UPF", path);
    const Attributes upf_attributes = parse_attributes(upf_tag);
    data.header.version = required_attribute(
        upf_attributes, "version", "UPF", path
    );

    const OpeningTag header_tag = find_opening_tag(document, "PP_HEADER", path);
    const Attributes header_attributes = parse_attributes(header_tag);
    UPFHeader& header = data.header;
    header.element = required_attribute(
        header_attributes, "element", "PP_HEADER", path
    );
    header.pseudo_type = required_attribute(
        header_attributes, "pseudo_type", "PP_HEADER", path
    );
    header.relativistic = required_attribute(
        header_attributes, "relativistic", "PP_HEADER", path
    );
    header.functional = required_attribute(
        header_attributes, "functional", "PP_HEADER", path
    );
    header.z_valence = parse_double_attribute(
        header_attributes, "z_valence", "PP_HEADER", path
    );
    header.wavefunction_cutoff_ry = parse_double_attribute(
        header_attributes, "wfc_cutoff", "PP_HEADER", path
    );
    header.charge_density_cutoff_ry = parse_double_attribute(
        header_attributes, "rho_cutoff", "PP_HEADER", path
    );
    header.l_max = parse_int_attribute(
        header_attributes, "l_max", "PP_HEADER", path
    );
    header.l_local = parse_int_attribute(
        header_attributes, "l_local", "PP_HEADER", path
    );
    header.mesh_size = parse_int_attribute(
        header_attributes, "mesh_size", "PP_HEADER", path
    );
    header.number_of_wavefunctions = parse_int_attribute(
        header_attributes, "number_of_wfc", "PP_HEADER", path
    );
    header.number_of_projectors = parse_int_attribute(
        header_attributes, "number_of_proj", "PP_HEADER", path
    );
    header.is_ultrasoft = parse_bool_attribute(
        header_attributes, "is_ultrasoft", "PP_HEADER", path
    );
    header.is_paw = parse_bool_attribute(
        header_attributes, "is_paw", "PP_HEADER", path
    );
    header.has_spin_orbit = parse_bool_attribute(
        header_attributes, "has_so", "PP_HEADER", path
    );
    header.has_core_correction = parse_bool_attribute(
        header_attributes, "core_correction", "PP_HEADER", path
    );
    validate_supported_header(header, path);

    const OpeningTag mesh_tag = find_opening_tag(document, "PP_MESH", path);
    const Attributes mesh_attributes = parse_attributes(mesh_tag);
    data.mesh.dx = parse_double_attribute(
        mesh_attributes, "dx", "PP_MESH", path
    );
    data.mesh.xmin = parse_double_attribute(
        mesh_attributes, "xmin", "PP_MESH", path
    );
    data.mesh.rmax = parse_double_attribute(
        mesh_attributes, "rmax", "PP_MESH", path
    );
    data.mesh.zmesh = parse_double_attribute(
        mesh_attributes, "zmesh", "PP_MESH", path
    );
    const int mesh_attribute = parse_int_attribute(
        mesh_attributes, "mesh", "PP_MESH", path
    );
    if (mesh_attribute != header.mesh_size) {
        throw parse_error(path, "PP_MESH mesh disagrees with PP_HEADER mesh_size");
    }

    data.mesh.r = parse_numeric_field(
        find_field_contents(document, "PP_R", path), "PP_R", path
    );
    data.mesh.rab = parse_numeric_field(
        find_field_contents(document, "PP_RAB", path), "PP_RAB", path
    );
    data.local_potential_ry = parse_numeric_field(
        find_field_contents(document, "PP_LOCAL", path), "PP_LOCAL", path
    );
    require_size(data.mesh.r, header.mesh_size, "PP_R", path);
    require_size(data.mesh.rab, header.mesh_size, "PP_RAB", path);
    require_size(
        data.local_potential_ry, header.mesh_size, "PP_LOCAL", path
    );

    for (int i = 0; i < header.mesh_size; ++i) {
        if (data.mesh.r[i] < 0.0) {
            throw parse_error(path, "PP_R contains a negative radius");
        }
        if (i > 0 && data.mesh.r[i] <= data.mesh.r[i - 1]) {
            throw parse_error(path, "PP_R must be strictly increasing");
        }
        if (data.mesh.rab[i] <= 0.0) {
            throw parse_error(path, "PP_RAB integration factors must be positive");
        }
    }

    data.projectors.reserve(header.number_of_projectors);
    for (int projector_number = 1;
         projector_number <= header.number_of_projectors;
         ++projector_number) {
        const std::string tag_name = "PP_BETA." + std::to_string(projector_number);
        const OpeningTag beta_tag = find_opening_tag(document, tag_name, path);
        const Attributes beta_attributes = parse_attributes(beta_tag);

        UPFProjector projector;
        projector.index = parse_int_attribute(
            beta_attributes, "index", tag_name, path
        );
        projector.label = required_attribute(
            beta_attributes, "label", tag_name, path
        );
        projector.angular_momentum = parse_int_attribute(
            beta_attributes, "angular_momentum", tag_name, path
        );
        projector.cutoff_radius_index = parse_int_attribute(
            beta_attributes, "cutoff_radius_index", tag_name, path
        );
        projector.cutoff_radius_bohr = parse_double_attribute(
            beta_attributes, "cutoff_radius", tag_name, path
        );
        projector.r_times_beta = parse_numeric_field(
            find_field_contents(document, tag_name, path), tag_name, path
        );

        if (projector.index != projector_number) {
            throw parse_error(path, tag_name + " index is not sequential");
        }
        if (projector.angular_momentum < 0) {
            throw parse_error(path, tag_name + " has negative angular momentum");
        }
        if (projector.cutoff_radius_index < 0 ||
            projector.cutoff_radius_index > header.mesh_size) {
            throw parse_error(path, tag_name + " has an invalid cutoff index");
        }
        if (projector.cutoff_radius_bohr < 0.0) {
            throw parse_error(path, tag_name + " has a negative cutoff radius");
        }
        if (header.l_max >= 0 && projector.angular_momentum > header.l_max) {
            throw parse_error(path, tag_name + " angular momentum exceeds l_max");
        }
        if (projector.r_times_beta.empty() ||
            static_cast<int>(projector.r_times_beta.size()) > header.mesh_size) {
            throw parse_error(path, tag_name + " has an invalid radial-array length");
        }

        projector.r_times_beta.resize(header.mesh_size, 0.0);
        data.projectors.push_back(std::move(projector));
    }

    if (header.number_of_projectors == 0) {
        /*
         * Some converted legacy files contain an uninitialized dummy value
         * inside PP_DIJ even though the header says there are no projectors.
         * The header is authoritative, so that payload is intentionally ignored.
         */
        data.dij_ry.clear();
    } else {
        data.dij_ry = parse_numeric_field(
            find_field_contents(document, "PP_DIJ", path), "PP_DIJ", path
        );
        require_size(
            data.dij_ry,
            header.number_of_projectors * header.number_of_projectors,
            "PP_DIJ",
            path
        );

        const double symmetry_tolerance = 1.0e-12;
        for (int i = 0; i < header.number_of_projectors; ++i) {
            for (int j = i + 1; j < header.number_of_projectors; ++j) {
                if (std::abs(data.dij_ry[i * header.number_of_projectors + j]
                             - data.dij_ry[j * header.number_of_projectors + i])
                    > symmetry_tolerance) {
                    throw parse_error(path, "PP_DIJ must be symmetric for NC data");
                }
            }
        }
    }

    data.atomic_wavefunctions.reserve(
        header.number_of_wavefunctions
    );
    for (int wavefunction_number = 1;
         wavefunction_number <= header.number_of_wavefunctions;
         ++wavefunction_number) {
        const std::string tag_name =
            "PP_CHI." + std::to_string(wavefunction_number);
        const OpeningTag chi_tag =
            find_opening_tag(document, tag_name, path);
        const Attributes chi_attributes =
            parse_attributes(chi_tag);

        UPFAtomicWavefunction wavefunction;
        wavefunction.index = parse_int_attribute(
            chi_attributes, "index", tag_name, path
        );
        wavefunction.label = required_attribute(
            chi_attributes, "label", tag_name, path
        );
        wavefunction.angular_momentum = parse_int_token(
            required_one_of_attributes(
                chi_attributes,
                {"l", "angular_momentum"},
                tag_name,
                path
            ),
            "<" + tag_name + ">",
            path
        );
        wavefunction.occupation = parse_double_attribute(
            chi_attributes, "occupation", tag_name, path
        );
        const std::string principal_quantum_number =
            optional_attribute(chi_attributes, "n");
        if (!principal_quantum_number.empty()) {
            wavefunction.principal_quantum_number =
                parse_int_token(
                    principal_quantum_number,
                    "<" + tag_name + ">",
                    path
                );
        }
        const std::string pseudo_energy =
            optional_attribute(chi_attributes, "pseudo_energy");
        if (!pseudo_energy.empty()) {
            wavefunction.pseudo_energy_ry =
                parse_double_token(
                    pseudo_energy, "<" + tag_name + ">", path
                );
        }
        wavefunction.r_times_radial_wavefunction =
            parse_numeric_field(
                find_field_contents(document, tag_name, path),
                tag_name,
                path
            );

        if (wavefunction.index != wavefunction_number) {
            throw parse_error(
                path, tag_name + " index is not sequential"
            );
        }
        if (wavefunction.angular_momentum < 0 ||
            (header.l_max >= 0 &&
             wavefunction.angular_momentum > header.l_max)) {
            throw parse_error(
                path, tag_name + " has invalid angular momentum"
            );
        }
        if (!std::isfinite(wavefunction.occupation) ||
            wavefunction.occupation < 0.0) {
            throw parse_error(
                path, tag_name + " has invalid occupation"
            );
        }
        if (wavefunction.r_times_radial_wavefunction.empty() ||
            static_cast<int>(
                wavefunction.r_times_radial_wavefunction.size()
            ) > header.mesh_size) {
            throw parse_error(
                path, tag_name + " has an invalid radial-array length"
            );
        }
        wavefunction.r_times_radial_wavefunction.resize(
            header.mesh_size, 0.0
        );
        data.atomic_wavefunctions.push_back(
            std::move(wavefunction)
        );
    }

    return data;
}
