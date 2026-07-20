#include "upf_local_potential.hpp"

#include "radial_transform.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {

void validate_species(const UPFLocalSpecies& species) {
    if (species.valence_charge <= 0.0 ||
        !std::isfinite(species.valence_charge)) {
        throw std::runtime_error(
            "UPF local species needs a positive finite valence charge."
        );
    }
    if (species.gaussian_width_bohr <= 0.0 ||
        !std::isfinite(species.gaussian_width_bohr)) {
        throw std::runtime_error(
            "UPF local species needs a positive finite Gaussian width."
        );
    }
    if (species.radial_grid_bohr.size()
            != species.quadrature_weights.size() ||
        species.radial_grid_bohr.size()
            != species.correction_hartree.size()) {
        throw std::runtime_error(
            "UPF local radial arrays must have identical sizes."
        );
    }
    if (!std::isfinite(species.g0_correction_hartree_bohr3)) {
        throw std::runtime_error(
            "UPF local G=0 correction must be finite."
        );
    }
}

void validate_ions(
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    for (const UPFLocalIon& ion : ions) {
        if (ion.species_index < 0 ||
            ion.species_index >= static_cast<int>(species.size())) {
            throw std::runtime_error("UPF local ion has an invalid species index.");
        }
        if (!ion.frac_position.allFinite()) {
            throw std::runtime_error(
                "UPF local ion has a non-finite fractional position."
            );
        }
    }
}

std::complex<double> phase_factor(
    const Eigen::Vector3d& G,
    const Eigen::Vector3d& R) {

    const double phase = -G.dot(R);
    return {std::cos(phase), std::sin(phase)};
}

} // namespace

double gaussian_coulomb_shape(
    double radius_bohr,
    double gaussian_width_bohr) {

    if (!std::isfinite(radius_bohr) || radius_bohr < 0.0) {
        throw std::runtime_error(
            "Gaussian Coulomb radius must be non-negative and finite."
        );
    }
    if (!std::isfinite(gaussian_width_bohr) ||
        gaussian_width_bohr <= 0.0) {
        throw std::runtime_error(
            "Gaussian Coulomb width must be positive and finite."
        );
    }

    if (radius_bohr < 1.0e-10 * gaussian_width_bohr) {
        return std::sqrt(2.0 / M_PI) / gaussian_width_bohr;
    }

    return std::erf(
        radius_bohr / (std::sqrt(2.0) * gaussian_width_bohr)
    ) / radius_bohr;
}

UPFLocalSpecies prepare_upf_local_species(
    const UPFData& upf,
    double gaussian_width_bohr) {

    if (upf.mesh.r.size() != upf.local_potential_ry.size() ||
        upf.mesh.r.size() != upf.mesh.rab.size()) {
        throw std::runtime_error(
            "UPF local potential and radial mesh sizes do not match."
        );
    }
    if (upf.header.z_valence <= 0.0 ||
        !std::isfinite(upf.header.z_valence)) {
        throw std::runtime_error(
            "UPF z_valence must be positive and finite."
        );
    }

    UPFLocalSpecies species;
    species.element = upf.header.element;
    species.valence_charge = upf.header.z_valence;
    species.gaussian_width_bohr = gaussian_width_bohr;
    species.radial_grid_bohr = upf.mesh.r;
    species.quadrature_weights =
        make_upf_simpson_weights(upf.mesh.rab);
    species.correction_hartree.resize(upf.mesh.r.size(), 0.0);
    species.g0_correction_hartree_bohr3 = 0.0;

    for (int i = 0; i < static_cast<int>(upf.mesh.r.size()); ++i) {
        const double r = upf.mesh.r[i];
        const double local_hartree =
            RYDBERG_TO_HARTREE * upf.local_potential_ry[i];
        const double positive_gaussian_coulomb =
            species.valence_charge * gaussian_coulomb_shape(
                upf.mesh.r[i],
                gaussian_width_bohr
            );
        species.correction_hartree[i] =
            local_hartree + positive_gaussian_coulomb;
        species.g0_correction_hartree_bohr3 +=
            4.0 * M_PI * species.quadrature_weights[i]
            * r * (r * local_hartree + species.valence_charge);
    }

    validate_species(species);
    return species;
}

double upf_local_correction_transform(
    const UPFLocalSpecies& species,
    double g_bohr_inverse) {

    validate_species(species);
    return radial_fourier_bessel_transform(
        0,
        g_bohr_inverse,
        species.radial_grid_bohr,
        species.quadrature_weights,
        species.correction_hartree
    );
}

double upf_local_kernel_G(
    const UPFLocalSpecies& species,
    double g_bohr_inverse) {

    if (!std::isfinite(g_bohr_inverse) || g_bohr_inverse < 0.0) {
        throw std::runtime_error(
            "UPF local reciprocal radius must be non-negative and finite."
        );
    }

    if (g_bohr_inverse < 1.0e-14) {
        validate_species(species);
        return species.g0_correction_hartree_bohr3;
    }

    const double correction =
        upf_local_correction_transform(species, g_bohr_inverse);
    const double g2 = g_bohr_inverse * g_bohr_inverse;
    const double sigma = species.gaussian_width_bohr;
    const double gaussian_coulomb =
        -4.0 * M_PI * species.valence_charge
        * std::exp(-0.5 * sigma * sigma * g2)
        / g2;
    return gaussian_coulomb + correction;
}

std::vector<std::complex<double>> build_upf_local_potential_G(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    validate_ions(species, ions);
    const double volume = lattice.volume();
    std::vector<std::complex<double>> potential_G(
        grid.ngrid,
        std::complex<double>(0.0, 0.0)
    );
    std::vector<double> species_kernels(species.size(), 0.0);

    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);
                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);
                const double g = G.norm();

                for (int ispecies = 0;
                     ispecies < static_cast<int>(species.size());
                     ++ispecies) {
                    species_kernels[ispecies] =
                        upf_local_kernel_G(species[ispecies], g);
                }

                std::complex<double> coefficient(0.0, 0.0);
                for (const UPFLocalIon& ion : ions) {
                    const Eigen::Vector3d R =
                        lattice.cart_from_frac(ion.frac_position);
                    coefficient +=
                        species_kernels[ion.species_index]
                        * phase_factor(G, R)
                        / volume;
                }
                potential_G[p] = coefficient;
            }
        }
    }

    return potential_G;
}

std::vector<double> build_upf_local_potential_real(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    const auto potential_G = build_upf_local_potential_G(
        lattice,
        fft.grid,
        species,
        ions
    );
    std::copy(
        potential_G.begin(),
        potential_G.end(),
        fft.reciprocal_grid.begin()
    );
    fftw_execute(fft.backward_plan);

    std::vector<double> potential(fft.grid.ngrid, 0.0);
    for (int p = 0; p < fft.grid.ngrid; ++p) {
        potential[p] = fft.real_grid[p].real();
    }
    return potential;
}
