#include "upf_local_potential.hpp"

#include "radial_transform.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace {

constexpr long long openmp_minimum_work = 32768;

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
    if (species.radial_grid_bohr.empty()) {
        throw std::runtime_error("UPF local radial grid is empty.");
    }
    for (int ir = 0;
         ir < static_cast<int>(species.radial_grid_bohr.size());
         ++ir) {
        const double radius = species.radial_grid_bohr[ir];
        const double weight = species.quadrature_weights[ir];
        const double correction = species.correction_hartree[ir];
        if (!std::isfinite(radius) || radius < 0.0 ||
            (ir > 0 && radius <= species.radial_grid_bohr[ir - 1])) {
            throw std::runtime_error(
                "UPF local radial grid must be finite and increasing."
            );
        }
        if (!std::isfinite(weight) || weight <= 0.0 ||
            !std::isfinite(correction)) {
            throw std::runtime_error(
                "UPF local radial weights and values must be finite."
            );
        }
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

double upf_local_correction_transform_unchecked(
    const UPFLocalSpecies& species,
    double g_bohr_inverse) {

    return radial_fourier_bessel_transform_unchecked(
        0,
        g_bohr_inverse,
        species.radial_grid_bohr,
        species.quadrature_weights,
        species.correction_hartree
    );
}

double upf_local_kernel_G_unchecked(
    const UPFLocalSpecies& species,
    double g_bohr_inverse) {

    if (g_bohr_inverse < 1.0e-14) {
        return species.g0_correction_hartree_bohr3;
    }

    const double correction =
        upf_local_correction_transform_unchecked(
            species,
            g_bohr_inverse
        );
    const double g2 = g_bohr_inverse * g_bohr_inverse;
    const double sigma = species.gaussian_width_bohr;
    const double gaussian_coulomb =
        -4.0 * M_PI * species.valence_charge
        * std::exp(-0.5 * sigma * sigma * g2)
        / g2;
    return gaussian_coulomb + correction;
}

void validate_cache(
    const UPFLocalReciprocalCache& cache,
    int expected_species_count = -1) {

    if (cache.n1 <= 0 || cache.n2 <= 0 || cache.n3 <= 0 ||
        cache.species_count < 0) {
        throw std::runtime_error(
            "UPF local reciprocal cache has invalid dimensions."
        );
    }
    const int ngrid = cache.grid_size();
    if (static_cast<int>(cache.g_cart.size()) != ngrid ||
        static_cast<int>(cache.species_kernels.size())
            != cache.species_count * ngrid) {
        throw std::runtime_error(
            "UPF local reciprocal cache has inconsistent storage."
        );
    }
    if (expected_species_count >= 0 &&
        cache.species_count != expected_species_count) {
        throw std::runtime_error(
            "UPF local reciprocal cache species count mismatch."
        );
    }
}

void validate_cache_ions(
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions) {

    validate_cache(cache);
    for (const UPFLocalIon& ion : ions) {
        if (ion.species_index < 0 ||
            ion.species_index >= cache.species_count) {
            throw std::runtime_error(
                "UPF local ion has an invalid cached species index."
            );
        }
        if (!ion.frac_position.allFinite()) {
            throw std::runtime_error(
                "UPF local ion has a non-finite fractional position."
            );
        }
    }
}

} // namespace

int UPFLocalReciprocalCache::grid_size() const {
    if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
        return 0;
    }
    return n1 * n2 * n3;
}

double UPFLocalReciprocalCache::kernel(
    int species_index,
    int grid_index) const {

    const int ngrid = grid_size();
    if (species_index < 0 || species_index >= species_count ||
        grid_index < 0 || grid_index >= ngrid) {
        throw std::runtime_error(
            "UPF local reciprocal-cache index is out of range."
        );
    }
    return species_kernels[
        static_cast<std::size_t>(species_index)
            * static_cast<std::size_t>(ngrid)
        + static_cast<std::size_t>(grid_index)
    ];
}

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
    return upf_local_correction_transform_unchecked(
        species,
        g_bohr_inverse
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

    validate_species(species);
    return upf_local_kernel_G_unchecked(species, g_bohr_inverse);
}

UPFLocalReciprocalCache build_upf_local_reciprocal_cache(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    int thread_count) {

    if (thread_count <= 0) {
        throw std::runtime_error(
            "UPF local reciprocal-cache thread count must be positive."
        );
    }
    for (const UPFLocalSpecies& one_species : species) {
        validate_species(one_species);
    }

    UPFLocalReciprocalCache cache;
    cache.n1 = grid.n1;
    cache.n2 = grid.n2;
    cache.n3 = grid.n3;
    cache.species_count = static_cast<int>(species.size());
    cache.g_cart.resize(grid.ngrid);
    cache.species_kernels.resize(
        static_cast<std::size_t>(cache.species_count)
            * static_cast<std::size_t>(grid.ngrid),
        0.0
    );

#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && grid.ngrid >= openmp_minimum_work) \
    num_threads(thread_count)
    for (int p = 0; p < grid.ngrid; ++p) {
        const int plane_size = grid.n2 * grid.n3;
        const int i = p / plane_size;
        const int remainder = p - i * plane_size;
        const int j = remainder / grid.n3;
        const int k = remainder - j * grid.n3;
        const Eigen::Vector3i n =
            grid.freq_from_indices(i, j, k);
        cache.g_cart[p] = lattice.gvector_from_freq(n);
    }

    std::map<double, int> exact_radius_classes;
    std::vector<double> unique_radii;
    std::vector<int> grid_to_radius(grid.ngrid, 0);
    for (int p = 0; p < grid.ngrid; ++p) {
        const double g2 = cache.g_cart[p].squaredNorm();
        const auto inserted = exact_radius_classes.emplace(
            g2,
            static_cast<int>(unique_radii.size())
        );
        if (inserted.second) {
            unique_radii.push_back(std::sqrt(g2));
        }
        grid_to_radius[p] = inserted.first->second;
    }

    const int nradii = static_cast<int>(unique_radii.size());
    std::vector<double> unique_kernels(
        static_cast<std::size_t>(cache.species_count)
            * static_cast<std::size_t>(nradii),
        0.0
    );
    const std::int64_t transform_work =
        static_cast<std::int64_t>(cache.species_count)
        * static_cast<std::int64_t>(nradii);
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && transform_work >= openmp_minimum_work) \
    num_threads(thread_count)
    for (std::int64_t index = 0;
         index < transform_work;
         ++index) {
        const int ispecies = static_cast<int>(
            index / static_cast<std::int64_t>(nradii)
        );
        const int iradius = static_cast<int>(
            index % static_cast<std::int64_t>(nradii)
        );
        unique_kernels[static_cast<std::size_t>(index)] =
            upf_local_kernel_G_unchecked(
                species[ispecies],
                unique_radii[iradius]
            );
    }

    const std::int64_t expansion_work =
        static_cast<std::int64_t>(cache.species_count)
        * static_cast<std::int64_t>(grid.ngrid);
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && expansion_work >= openmp_minimum_work) \
    num_threads(thread_count)
    for (std::int64_t index = 0;
         index < expansion_work;
         ++index) {
        const int ispecies = static_cast<int>(
            index / static_cast<std::int64_t>(grid.ngrid)
        );
        const int p = static_cast<int>(
            index % static_cast<std::int64_t>(grid.ngrid)
        );
        cache.species_kernels[static_cast<std::size_t>(index)] =
            unique_kernels[
                static_cast<std::size_t>(ispecies)
                    * static_cast<std::size_t>(nradii)
                + static_cast<std::size_t>(grid_to_radius[p])
            ];
    }

    validate_cache(cache, static_cast<int>(species.size()));
    return cache;
}

std::vector<std::complex<double>> build_upf_local_potential_G(
    const Lattice& lattice,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions,
    int thread_count) {

    if (thread_count <= 0) {
        throw std::runtime_error(
            "UPF local-potential thread count must be positive."
        );
    }
    validate_cache_ions(cache, ions);
    const double volume = lattice.volume();
    const int ngrid = cache.grid_size();
    std::vector<std::complex<double>> potential_G(
        ngrid,
        std::complex<double>(0.0, 0.0)
    );
    std::vector<Eigen::Vector3d> cart_positions(ions.size());
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        cart_positions[iion] =
            lattice.cart_from_frac(ions[iion].frac_position);
    }

#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && \
       static_cast<long long>(ngrid) * ions.size() >= \
           openmp_minimum_work) \
    num_threads(thread_count)
    for (int p = 0; p < ngrid; ++p) {
        const Eigen::Vector3d& G = cache.g_cart[p];
        std::complex<double> coefficient(0.0, 0.0);
        for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
            const std::size_t kernel_index =
                static_cast<std::size_t>(ions[iion].species_index)
                    * static_cast<std::size_t>(ngrid)
                + static_cast<std::size_t>(p);
            coefficient +=
                cache.species_kernels[kernel_index]
                * phase_factor(G, cart_positions[iion]);
        }
        potential_G[p] = coefficient / volume;
    }
    return potential_G;
}

std::vector<std::complex<double>> build_upf_local_potential_G(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    validate_ions(species, ions);
    const UPFLocalReciprocalCache cache =
        build_upf_local_reciprocal_cache(
            lattice,
            grid,
            species
        );
    return build_upf_local_potential_G(
        lattice,
        cache,
        ions
    );
}

std::vector<double> build_upf_local_potential_real(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions) {

    validate_cache_ions(cache, ions);
    if (cache.n1 != fft.grid.n1 ||
        cache.n2 != fft.grid.n2 ||
        cache.n3 != fft.grid.n3) {
        throw std::runtime_error(
            "UPF local reciprocal cache does not match the FFT grid."
        );
    }
    const auto potential_G = build_upf_local_potential_G(
        lattice,
        cache,
        ions,
        fft.thread_count
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

std::vector<double> build_upf_local_potential_real(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    validate_ions(species, ions);
    const UPFLocalReciprocalCache cache =
        build_upf_local_reciprocal_cache(
            lattice,
            fft.grid,
            species,
            fft.thread_count
        );
    return build_upf_local_potential_real(
        lattice,
        fft,
        cache,
        ions
    );
}
