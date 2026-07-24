#include "forces.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr long long openmp_minimum_work = 32768;

std::complex<double> phase_factor(
    const Eigen::Vector3d& G,
    const Eigen::Vector3d& R) {

    const double phase = -G.dot(R);
    return {std::cos(phase), std::sin(phase)};
}

std::complex<double> single_ion_local_potential_G(
    const Lattice& lattice,
    const Ion& ion,
    const Eigen::Vector3d& G) {

    const double volume = lattice.volume();
    const double G2 = G.squaredNorm();
    const Eigen::Vector3d R =
        lattice.cart_from_frac(ion.frac_position);
    const std::complex<double> phase = phase_factor(G, R);

    std::complex<double> V_G(0.0, 0.0);

    if (G2 >= 1.0e-14) {
        const double rho_prefactor =
            ion.Z * std::exp(-0.5 * ion.sigma * ion.sigma * G2)
            / volume;
        V_G += -(4.0 * M_PI / G2) * rho_prefactor * phase;
    }

    if (std::abs(ion.short_amp) >= 1.0e-14) {
        if (ion.short_rc <= 0.0) {
            throw std::runtime_error("Ion short_rc must be positive.");
        }

        const double rc = ion.short_rc;
        const double prefactor =
            ion.short_amp * std::pow(2.0 * M_PI, 1.5)
            * rc * rc * rc / volume;
        V_G += prefactor * std::exp(-0.5 * rc * rc * G2) * phase;
    }

    return V_G;
}

std::vector<Eigen::Vector3d> zero_forces(int nions) {
    return std::vector<Eigen::Vector3d>(
        nions,
        Eigen::Vector3d::Zero()
    );
}

} // namespace

std::vector<Eigen::Vector3d> compute_local_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions,
    const std::vector<std::complex<double>>& n_G) {

    if (static_cast<int>(n_G.size()) != grid.ngrid) {
        throw std::runtime_error("n_G size mismatch in local ionic force.");
    }

    const double volume = lattice.volume();
    auto forces = zero_forces(static_cast<int>(ions.size()));
    const std::complex<double> imag_unit(0.0, 1.0);

    /*
     * E_loc,I = Omega sum_G n_G^* V_I(G),
     * d V_I(G) / d R_Ia = -i G_a V_I(G), hence
     * F_Ia = Omega Re sum_G i G_a n_G^* V_I(G).
     */
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);
                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);

                for (int iion = 0;
                     iion < static_cast<int>(ions.size());
                     ++iion) {
                    const std::complex<double> V_I_G =
                        single_ion_local_potential_G(
                            lattice,
                            ions[iion],
                            G
                        );

                    const std::complex<double> common =
                        volume * imag_unit * std::conj(n_G[p]) * V_I_G;

                    for (int a = 0; a < 3; ++a) {
                        forces[iion][a] += G[a] * common.real();
                    }
                }
            }
        }
    }

    return forces;
}

std::vector<Eigen::Vector3d> compute_upf_local_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<UPFLocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    const std::vector<std::complex<double>>& n_G) {

    const UPFLocalReciprocalCache cache =
        build_upf_local_reciprocal_cache(
            lattice,
            grid,
            species
        );
    return compute_upf_local_ionic_forces(
        lattice,
        cache,
        ions,
        n_G
    );
}

std::vector<Eigen::Vector3d> compute_upf_local_ionic_forces(
    const Lattice& lattice,
    const UPFLocalReciprocalCache& cache,
    const std::vector<UPFLocalIon>& ions,
    const std::vector<std::complex<double>>& n_G,
    int thread_count) {

    if (thread_count <= 0) {
        throw std::runtime_error(
            "UPF local-force thread count must be positive."
        );
    }
    const int ngrid = cache.grid_size();
    if (static_cast<int>(n_G.size()) != ngrid) {
        throw std::runtime_error("n_G size mismatch in UPF local ionic force.");
    }
    if (static_cast<int>(cache.g_cart.size()) != ngrid ||
        static_cast<int>(cache.species_kernels.size())
            != cache.species_count * ngrid) {
        throw std::runtime_error(
            "UPF local reciprocal cache has inconsistent storage."
        );
    }
    for (const UPFLocalIon& ion : ions) {
        if (ion.species_index < 0 ||
            ion.species_index >= cache.species_count) {
            throw std::runtime_error(
                "UPF local ion has an invalid species index."
            );
        }
    }

    const double volume = lattice.volume();
    auto forces = zero_forces(static_cast<int>(ions.size()));
    const std::complex<double> imag_unit(0.0, 1.0);
    std::vector<Eigen::Vector3d> cart_positions(ions.size());
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        cart_positions[iion] =
            lattice.cart_from_frac(ions[iion].frac_position);
    }

    const long long work =
        static_cast<long long>(ngrid)
        * static_cast<long long>(ions.size());
#pragma omp parallel \
    if(thread_count > 1 && work >= openmp_minimum_work) \
    num_threads(thread_count)
    {
        auto local_forces =
            zero_forces(static_cast<int>(ions.size()));
#pragma omp for schedule(static)
        for (int p = 0; p < ngrid; ++p) {
            const Eigen::Vector3d& G = cache.g_cart[p];
            const std::complex<double> density_factor =
                volume * imag_unit * std::conj(n_G[p]);
            for (int iion = 0;
                 iion < static_cast<int>(ions.size());
                 ++iion) {
                const UPFLocalIon& ion = ions[iion];
                const double kernel =
                    cache.species_kernels[
                        static_cast<std::size_t>(ion.species_index)
                            * static_cast<std::size_t>(ngrid)
                        + static_cast<std::size_t>(p)
                    ];
                const std::complex<double> V_I_G =
                    kernel
                    * phase_factor(G, cart_positions[iion])
                    / volume;
                const double common =
                    (density_factor * V_I_G).real();
                local_forces[iion] += common * G;
            }
        }
#pragma omp critical
        {
            for (int iion = 0;
                 iion < static_cast<int>(ions.size());
                 ++iion) {
                forces[iion] += local_forces[iion];
            }
        }
    }

    return forces;
}

std::vector<Eigen::Vector3d> compute_smooth_ion_ion_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions) {

    const double volume = lattice.volume();
    auto forces = zero_forces(static_cast<int>(ions.size()));
    const auto rho_total_G =
        build_gaussian_ion_density_G(lattice, grid, ions);
    const std::complex<double> imag_unit(0.0, 1.0);

    /*
     * E_II = 2 pi Omega sum_{G != 0} |rho_I(G)|^2 / G^2.
     * Differentiating rho_Iion(G) gives -i G rho_Iion(G):
     *
     * F_Ia = 4 pi Omega Re sum_{G != 0}
     *         [i G_a rho_total(G)^* rho_I(G)] / G^2.
     */
    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);
                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);
                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);
                const double G2 = G.squaredNorm();

                if (G2 < 1.0e-14) {
                    continue;
                }

                for (int iion = 0;
                     iion < static_cast<int>(ions.size());
                     ++iion) {
                    const Ion& ion = ions[iion];
                    const Eigen::Vector3d R =
                        lattice.cart_from_frac(ion.frac_position);
                    const std::complex<double> rho_I_G =
                        ion.Z
                        * std::exp(-0.5 * ion.sigma * ion.sigma * G2)
                        * phase_factor(G, R)
                        / volume;
                    const std::complex<double> common =
                        (4.0 * M_PI * volume / G2)
                        * imag_unit
                        * std::conj(rho_total_G[p])
                        * rho_I_G;

                    for (int a = 0; a < 3; ++a) {
                        forces[iion][a] += G[a] * common.real();
                    }
                }
            }
        }
    }

    return forces;
}

std::vector<Eigen::Vector3d> compute_nonlocal_ionic_forces(
    const PlaneWaveBasis3D& basis,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    int nions,
    int thread_count) {

    if (C.rows() != basis.size()) {
        throw std::runtime_error("C row size mismatch in nonlocal force.");
    }
    if (static_cast<int>(occupations.size()) != C.cols()) {
        throw std::runtime_error(
            "Occupation size mismatch in nonlocal force."
        );
    }
    if (nions < 0) {
        throw std::runtime_error(
            "Ion count cannot be negative in nonlocal force."
        );
    }
    if (thread_count <= 0) {
        throw std::runtime_error(
            "Nonlocal-force thread count must be positive."
        );
    }

    auto forces = zero_forces(nions);
    const int nprojectors = static_cast<int>(projectors.size());
    if (nprojectors == 0 || C.cols() == 0) {
        return forces;
    }

    Eigen::MatrixXcd beta(basis.size(), nprojectors);
    Eigen::VectorXd strengths(nprojectors);
    std::vector<int> ion_indices(nprojectors, -1);
    for (int ip = 0; ip < nprojectors; ++ip) {
        const NonlocalProjector& proj = projectors[ip];
        if (proj.beta_G.size() != basis.size()) {
            throw std::runtime_error("Projector size mismatch in nonlocal force.");
        }
        if (proj.ion_index < 0 || proj.ion_index >= nions) {
            throw std::runtime_error("Invalid projector ion index.");
        }
    }
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && \
       static_cast<long long>(basis.size()) * nprojectors >= \
           openmp_minimum_work) \
    num_threads(thread_count)
    for (int ip = 0; ip < nprojectors; ++ip) {
        const NonlocalProjector& proj = projectors[ip];
        beta.col(ip) = proj.beta_G;
        strengths[ip] = proj.D;
        ion_indices[ip] = proj.ion_index;
    }

    Eigen::MatrixXcd overlaps(nprojectors, C.cols());
    overlaps.noalias() = beta.adjoint() * C;
    Eigen::MatrixXcd weighted_orbitals(C.rows(), C.cols());
    Eigen::MatrixXcd derivative_overlaps(nprojectors, C.cols());
    const std::complex<double> imag_unit(0.0, 1.0);

    for (int a = 0; a < 3; ++a) {
#pragma omp parallel for collapse(2) schedule(static) \
    if(thread_count > 1 && \
       static_cast<long long>(C.rows()) * C.cols() >= \
           openmp_minimum_work) \
    num_threads(thread_count)
        for (int ib = 0; ib < C.cols(); ++ib) {
            for (int ig = 0; ig < C.rows(); ++ig) {
                weighted_orbitals(ig, ib) =
                    imag_unit
                    * basis.gvectors[ig].q_cart[a]
                    * C(ig, ib);
            }
        }
        derivative_overlaps.noalias() =
            beta.adjoint() * weighted_orbitals;

        for (int ip = 0; ip < nprojectors; ++ip) {
            for (int ib = 0; ib < C.cols(); ++ib) {
                const double occ = occupations[ib];
                if (std::abs(occ) < 1.0e-14) {
                    continue;
                }
                /*
                 * E_NL = sum_n f_n D |<beta|psi_nk>|^2.
                 * The force is minus its explicit R derivative.
                 */
                forces[ion_indices[ip]][a] +=
                    -2.0 * occ * strengths[ip]
                    * std::real(
                        std::conj(overlaps(ip, ib))
                        * derivative_overlaps(ip, ib)
                    );
            }
        }
    }

    return forces;
}

IonicForceComponents compute_ionic_forces(
    const Lattice& lattice,
    const FFTGrid& grid,
    const PlaneWaveBasis3D& basis,
    const std::vector<Ion>& ions,
    const std::vector<std::complex<double>>& n_G,
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations) {

    IonicForceComponents out;
    out.local = compute_local_ionic_forces(lattice, grid, ions, n_G);
    out.ion_ion = compute_smooth_ion_ion_forces(lattice, grid, ions);
    out.nonlocal = compute_nonlocal_ionic_forces(
        basis,
        projectors,
        C,
        occupations,
        static_cast<int>(ions.size())
    );
    out.total = zero_forces(static_cast<int>(ions.size()));

    for (int i = 0; i < static_cast<int>(ions.size()); ++i) {
        out.total[i] = out.local[i] + out.ion_ion[i] + out.nonlocal[i];
    }

    return out;
}
