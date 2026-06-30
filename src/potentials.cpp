#include "potentials.hpp"

#include <complex>
#include <cmath>
#include <stdexcept>

std::vector<std::complex<double>> build_gaussian_ion_density_G(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions) {

    const double volume = lattice.volume();

    std::vector<std::complex<double>> rhoI_G(
        grid.ngrid,
        std::complex<double>(0.0, 0.0)
    );

    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);

                const Eigen::Vector3i n =
                    grid.freq_from_indices(i, j, k);

                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);

                const double G2 = G.squaredNorm();

                std::complex<double> structure_factor(0.0, 0.0);

                for (const Ion& ion : ions) {
                    const Eigen::Vector3d R =
                        lattice.cart_from_frac(ion.frac_position);

                    const double phase = -G.dot(R);

                    const std::complex<double> exp_phase(
                        std::cos(phase),
                        std::sin(phase)
                    );

                    const double smooth =
                        std::exp(-0.5 * ion.sigma * ion.sigma * G2);

                    structure_factor +=
                        ion.Z * smooth * exp_phase;
                }

                /*
                 * rho_I(G) is charge density Fourier coefficient:
                 *
                 *   rho_I(G) = 1/Omega * sum_I Z_I ...
                 */
                rhoI_G[p] = structure_factor / volume;
            }
        }
    }

    return rhoI_G;
}

std::vector<double> build_gaussian_local_pseudopotential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions) {

    const int ngrid = fft.grid.ngrid;

    std::vector<std::complex<double>> rhoI_G =
        build_gaussian_ion_density_G(
            lattice,
            fft.grid,
            ions
        );

    std::fill(
        fft.reciprocal_grid.begin(),
        fft.reciprocal_grid.end(),
        std::complex<double>(0.0, 0.0)
    );

    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                const int p = fft.grid.index(i, j, k);

                const Eigen::Vector3i n =
                    fft.grid.freq_from_indices(i, j, k);

                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);

                const double G2 = G.squaredNorm();

                if (G2 < 1.0e-14) {
                    fft.reciprocal_grid[p] =
                        std::complex<double>(0.0, 0.0);
                } else {
                    /*
                     * Electron-ion potential:
                     *
                     *   Vloc(G) = -4 pi rho_I(G) / G^2
                     */
                    fft.reciprocal_grid[p] =
                        -(4.0 * M_PI / G2) * rhoI_G[p];
                }
            }
        }
    }

    fftw_execute(fft.backward_plan);

    std::vector<double> Vloc(ngrid, 0.0);

    for (int p = 0; p < ngrid; ++p) {
        Vloc[p] = fft.real_grid[p].real();
    }

    return Vloc;
}

double compute_smooth_ion_ion_energy(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<Ion>& ions) {

    const double volume = lattice.volume();

    std::vector<std::complex<double>> rhoI_G =
        build_gaussian_ion_density_G(
            lattice,
            grid,
            ions
        );

    double e = 0.0;

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

                e += std::norm(rhoI_G[p]) / G2;
            }
        }
    }

    return 2.0 * M_PI * volume * e;
}

double compute_total_electrostatic_energy_direct(
    const Lattice& lattice,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& rhoI_G,
    const std::vector<std::complex<double>>& n_G) {

    if (rhoI_G.size() != n_G.size()) {
        throw std::runtime_error(
            "rhoI_G and n_G size mismatch in electrostatic energy."
        );
    }

    if (static_cast<int>(rhoI_G.size()) != grid.ngrid) {
        throw std::runtime_error(
            "Grid size mismatch in electrostatic energy."
        );
    }

    const double volume = lattice.volume();

    double e = 0.0;

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

                /*
                 * Total charge density:
                 *
                 *   rho_tot(G) = rho_I(G) - n(G)
                 *
                 * where n(G) is the positive electron number density.
                 */
                const std::complex<double> rho_tot_G =
                    rhoI_G[p] - n_G[p];

                e += std::norm(rho_tot_G) / G2;
            }
        }
    }

    return 2.0 * M_PI * volume * e;
}

std::vector<double> build_short_range_local_potential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions) {

    const int ngrid = fft.grid.ngrid;
    const double volume = lattice.volume();

    std::fill(
        fft.reciprocal_grid.begin(),
        fft.reciprocal_grid.end(),
        std::complex<double>(0.0, 0.0)
    );

    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                const int p = fft.grid.index(i, j, k);

                const Eigen::Vector3i n =
                    fft.grid.freq_from_indices(i, j, k);

                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);

                const double G2 = G.squaredNorm();

                std::complex<double> Vg(0.0, 0.0);

                for (const Ion& ion : ions) {
                    if (std::abs(ion.short_amp) < 1.0e-14) {
                        continue;
                    }

                    if (ion.short_rc <= 0.0) {
                        throw std::runtime_error("Ion short_rc must be positive.");
                    }

                    const Eigen::Vector3d R =
                        lattice.cart_from_frac(ion.frac_position);

                    const double phase = -G.dot(R);

                    const std::complex<double> exp_phase(
                        std::cos(phase),
                        std::sin(phase)
                    );

                    const double rc = ion.short_rc;

                    const double prefactor =
                        ion.short_amp
                        * std::pow(2.0 * M_PI, 1.5)
                        * rc * rc * rc
                        / volume;

                    const double smooth =
                        std::exp(-0.5 * rc * rc * G2);

                    Vg += prefactor * smooth * exp_phase;
                }

                fft.reciprocal_grid[p] = Vg;
            }
        }
    }

    fftw_execute(fft.backward_plan);

    std::vector<double> Vshort(ngrid, 0.0);

    for (int p = 0; p < ngrid; ++p) {
        Vshort[p] = fft.real_grid[p].real();
    }

    return Vshort;
}

LocalPotentialComponents build_local_pseudopotential_components(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<Ion>& ions) {

    LocalPotentialComponents out;

    out.Vcoul =
        build_gaussian_local_pseudopotential(
            lattice,
            fft,
            ions
        );

    out.Vshort =
        build_short_range_local_potential(
            lattice,
            fft,
            ions
        );

    if (out.Vcoul.size() != out.Vshort.size()) {
        throw std::runtime_error("Vloc component size mismatch.");
    }

    out.Vtotal.assign(out.Vcoul.size(), 0.0);

    for (int p = 0; p < static_cast<int>(out.Vtotal.size()); ++p) {
        out.Vtotal[p] = out.Vcoul[p] + out.Vshort[p];
    }

    out.Eion_smooth =
        compute_smooth_ion_ion_energy(
            lattice,
            fft.grid,
            ions
        );

    return out;
}

std::vector<NonlocalProjector> build_s_gaussian_projectors(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<Ion>& ions) {

    std::vector<NonlocalProjector> projectors;

    for (const Ion& ion : ions) {
        if (std::abs(ion.beta_D) < 1.0e-14) {
            continue;
        }

        if (ion.beta_rc <= 0.0) {
            throw std::runtime_error("Ion beta_rc must be positive.");
        }

        NonlocalProjector proj;
        proj.beta_G = Eigen::VectorXcd::Zero(basis.size());
        proj.D = ion.beta_D;

        const Eigen::Vector3d R =
            lattice.cart_from_frac(ion.frac_position);

        for (int ig = 0; ig < basis.size(); ++ig) {
            const Eigen::Vector3d G =
                basis.gvectors[ig].G_cart;

            const double G2 = G.squaredNorm();

            const double phase =
                -G.dot(R);

            const std::complex<double> exp_phase(
                std::cos(phase),
                std::sin(phase)
            );

            const double smooth =
                std::exp(
                    -0.5 * ion.beta_rc * ion.beta_rc * G2
                );

            proj.beta_G[ig] =
                smooth * exp_phase;
        }

        const double nrm =
            proj.beta_G.norm();

        if (nrm < 1.0e-14) {
            throw std::runtime_error("Projector norm too small.");
        }

        proj.beta_G /= nrm;

        projectors.push_back(proj);
    }

    return projectors;
}

Eigen::VectorXcd apply_nonlocal_projectors(
    const std::vector<NonlocalProjector>& projectors,
    const Eigen::VectorXcd& c) {

    Eigen::VectorXcd out =
        Eigen::VectorXcd::Zero(c.size());

    for (const NonlocalProjector& proj : projectors) {
        if (proj.beta_G.size() != c.size()) {
            throw std::runtime_error("Projector size mismatch.");
        }

        const std::complex<double> coeff =
            proj.beta_G.dot(c);

        out += proj.D * proj.beta_G * coeff;
    }

    return out;
}