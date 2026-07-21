#pragma once

#include <string>
#include <vector>

/*
 * UPF uses atomic Rydberg units, while the PWDFT code uses Hartree atomic
 * units internally. Lengths are Bohr in both conventions.
 */
constexpr double RYDBERG_TO_HARTREE = 0.5;

struct UPFHeader {
    std::string version;
    std::string element;
    std::string pseudo_type;
    std::string relativistic;
    std::string functional;

    double z_valence = 0.0;
    double wavefunction_cutoff_ry = 0.0;
    double charge_density_cutoff_ry = 0.0;

    int l_max = -1;
    int l_local = -1;
    int mesh_size = 0;
    int number_of_wavefunctions = 0;
    int number_of_projectors = 0;

    bool is_ultrasoft = false;
    bool is_paw = false;
    bool has_spin_orbit = false;
    bool has_core_correction = false;
};

struct UPFRadialMesh {
    double dx = 0.0;
    double xmin = 0.0;
    double rmax = 0.0;
    double zmesh = 0.0;

    /* Bohr. */
    std::vector<double> r;

    /* UPF discrete radial-integration factors. */
    std::vector<double> rab;
};

struct UPFProjector {
    int index = 0;
    std::string label;
    int angular_momentum = -1;
    int cutoff_radius_index = 0;
    double cutoff_radius_bohr = 0.0;

    /*
     * UPF stores r * beta_l(r), including the factor r. The reader pads a
     * projector with zeros when a file stores only points up to its cutoff.
     */
    std::vector<double> r_times_beta;
};

struct UPFData {
    std::string source_path;
    UPFHeader header;
    UPFRadialMesh mesh;

    /* PP_LOCAL and PP_DIJ are stored in their native UPF energy unit, Ry. */
    std::vector<double> local_potential_ry;
    std::vector<UPFProjector> projectors;
    std::vector<double> dij_ry;

    double dij(int i, int j) const;
};

/*
 * Read the deliberately supported first subset:
 *
 *   - UPF v2
 *   - norm-conserving, fully separable pseudopotentials
 *   - no ultrasoft, PAW, nonlinear core correction, or spin-orbit data
 *
 * Unsupported physics is rejected instead of being silently ignored.
 */
UPFData read_nc_upf(const std::string& path);
