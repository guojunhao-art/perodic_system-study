#include "scf_modules.hpp"
#include "hamiltonian.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

std::vector<double> build_density_from_orbitals(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    double volume) {

    if (C.rows() != basis.size()) {
        throw std::runtime_error("C row size does not match basis size.");
    }

    if (static_cast<int>(occupations.size()) != C.cols()) {
        throw std::runtime_error("Occupation size does not match number of orbitals.");
    }

    std::vector<double> rho(fft.grid.ngrid, 0.0);

    for (int ib = 0; ib < C.cols(); ++ib) {
        const double occ = occupations[ib];

        if (occ == 0.0) {
            continue;
        }

        std::vector<std::complex<double>> coeffs(C.rows());

        for (int ig = 0; ig < C.rows(); ++ig) {
            coeffs[ig] = C(ig, ib);
        }

        scatter_coeffs_to_fft_grid(
            basis,
            fft.grid,
            coeffs,
            fft.reciprocal_grid
        );

        fftw_execute(fft.backward_plan);

        for (int p = 0; p < fft.grid.ngrid; ++p) {
            rho[p] += occ * std::norm(fft.real_grid[p]) / volume;
        }
    }

    return rho;
}

double integrate_real_grid(
    const std::vector<double>& f,
    double dV) {

    double sum = 0.0;

    for (double x : f) {
        sum += x;
    }

    return dV * sum;
}

double electron_number_from_density(
    const std::vector<double>& rho,
    double dV) {

    return integrate_real_grid(rho, dV);
}

std::vector<double> build_hartree_potential(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& rho) {

    if (static_cast<int>(rho.size()) != fft.grid.ngrid) {
        throw std::runtime_error("rho size does not match FFT grid.");
    }

    /*
     * rho(r) -> rho(G)
     */
    for (int p = 0; p < fft.grid.ngrid; ++p) {
        fft.real_grid[p] = std::complex<double>(rho[p], 0.0);
    }

    fftw_execute(fft.forward_plan);

    std::vector<std::complex<double>> rho_G(fft.grid.ngrid);

    for (int p = 0; p < fft.grid.ngrid; ++p) {
        rho_G[p] =
            fft.forward_raw[p] / static_cast<double>(fft.grid.ngrid);
    }

    /*
     * VH(G) = 4 pi rho(G) / G^2, G != 0
     * VH(0) = 0
     */
    std::fill(
        fft.reciprocal_grid.begin(),
        fft.reciprocal_grid.end(),
        std::complex<double>(0.0, 0.0)
    );

    const double tiny = 1.0e-14;

    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                const int p = fft.grid.index(i, j, k);

                const Eigen::Vector3i n =
                    fft.grid.freq_from_indices(i, j, k);

                const Eigen::Vector3d G =
                    lattice.gvector_from_freq(n);

                const double G2 = G.squaredNorm();

                if (G2 < tiny) {
                    fft.reciprocal_grid[p] = std::complex<double>(0.0, 0.0);
                } else {
                    fft.reciprocal_grid[p] =
                        (4.0 * M_PI / G2) * rho_G[p];
                }
            }
        }
    }

    /*
     * VH(G) -> VH(r)
     */
    fftw_execute(fft.backward_plan);

    std::vector<double> VH(fft.grid.ngrid, 0.0);

    for (int p = 0; p < fft.grid.ngrid; ++p) {
        VH[p] = fft.real_grid[p].real();
    }

    return VH;
}

LDAExchangeResult build_lda_exchange(
    const std::vector<double>& rho,
    double dV) {

    LDAExchangeResult out;

    out.Vx.assign(rho.size(), 0.0);
    out.Ex = 0.0;

    const double Cx =
        std::pow(3.0 / M_PI, 1.0 / 3.0);

    const double eps_rho = 1.0e-14;

    for (int p = 0; p < static_cast<int>(rho.size()); ++p) {
        const double n = std::max(rho[p], 0.0);

        if (n < eps_rho) {
            out.Vx[p] = 0.0;
            continue;
        }

        const double n13 = std::cbrt(n);

        out.Vx[p] = -Cx * n13;

        const double ex_density =
            -0.75 * Cx * n * n13;

        out.Ex += dV * ex_density;
    }

    return out;
}

std::vector<double> build_toy_ionic_potential(
    const FFTGrid& grid,
    double V0) {

    std::vector<double> Vion(grid.ngrid, 0.0);

    for (int i = 0; i < grid.n1; ++i) {
        for (int j = 0; j < grid.n2; ++j) {
            for (int k = 0; k < grid.n3; ++k) {
                const int p = grid.index(i, j, k);

                const Eigen::Vector3d s =
                    grid.frac_coord(i, j, k);

                Vion[p] =
                    V0 * (
                        std::cos(2.0 * M_PI * s[0]) +
                        std::cos(2.0 * M_PI * s[1]) +
                        std::cos(2.0 * M_PI * s[2])
                    );
            }
        }
    }

    return Vion;
}

std::vector<double> combine_effective_potential(
    const std::vector<double>& Vion,
    const std::vector<double>& VH,
    const std::vector<double>& Vx) {

    if (Vion.size() != VH.size() || Vion.size() != Vx.size()) {
        throw std::runtime_error("Potential size mismatch.");
    }

    std::vector<double> Veff(Vion.size(), 0.0);

    for (int p = 0; p < static_cast<int>(Veff.size()); ++p) {
        Veff[p] = Vion[p] + VH[p] + Vx[p];
    }

    return Veff;
}
double sum_occupations(
    const std::vector<double>& occ) {

    double s = 0.0;

    for (double f : occ) {
        s += f;
    }

    return s;
}

OccupationResult degeneracy_aware_zeroT_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    double occ_max,
    double deg_tol) {

    const int nbands = eps.size();

    if (nelec < -1.0e-12) {
        throw std::runtime_error("nelec cannot be negative.");
    }

    if (nelec > occ_max * nbands + 1.0e-12) {
        throw std::runtime_error(
            "Not enough bands to hold all electrons."
        );
    }

    OccupationResult out;
    out.occ.assign(nbands, 0.0);

    double remaining = nelec;

    int i = 0;

    while (i < nbands && remaining > 1.0e-12) {
        /*
         * Find a degenerate shell:
         *
         *   eps[i], eps[i+1], ..., eps[j-1]
         *
         * where |eps[k] - eps[i]| < deg_tol.
         */
        int j = i + 1;

        while (j < nbands &&
               std::abs(eps[j] - eps[i]) < deg_tol) {
            ++j;
        }

        const int shell_size = j - i;
        const double shell_capacity =
            occ_max * static_cast<double>(shell_size);

        if (remaining >= shell_capacity - 1.0e-12) {
            /*
             * Fill the whole shell.
             */
            for (int k = i; k < j; ++k) {
                out.occ[k] = occ_max;
            }

            remaining -= shell_capacity;
        } else {
            /*
             * Partially filled degenerate shell.
             * Distribute remaining electrons evenly.
             */
            const double f =
                remaining / static_cast<double>(shell_size);

            for (int k = i; k < j; ++k) {
                out.occ[k] = f;
            }

            remaining = 0.0;
        }

        i = j;
    }

    out.nelec_sum = sum_occupations(out.occ);

    /*
     * Fermi level diagnostic.
     * For zero-T, if partially filled shell exists, mu is set
     * to the energy of that shell. Otherwise set between HOMO/LUMO.
     */
    int last_occ = -1;
    int first_empty = -1;

    for (int ib = 0; ib < nbands; ++ib) {
        if (out.occ[ib] > 1.0e-12) {
            last_occ = ib;
        }

        if (first_empty < 0 && out.occ[ib] < occ_max - 1.0e-12) {
            first_empty = ib;
        }
    }

    if (first_empty >= 0 &&
        out.occ[first_empty] > 1.0e-12 &&
        out.occ[first_empty] < occ_max - 1.0e-12) {
        out.mu = eps[first_empty];
    } else if (last_occ >= 0 && last_occ + 1 < nbands) {
        out.mu = 0.5 * (eps[last_occ] + eps[last_occ + 1]);
    } else if (last_occ >= 0) {
        out.mu = eps[last_occ];
    } else {
        out.mu = eps[0];
    }

    return out;
}

double fermi_dirac_occ_single(
    double eps,
    double mu,
    double sigma,
    double occ_max) {

    const double x = (eps - mu) / sigma;

    if (x > 80.0) {
        return 0.0;
    }

    if (x < -80.0) {
        return occ_max;
    }

    return occ_max / (std::exp(x) + 1.0);
}

OccupationResult fermi_dirac_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    double sigma,
    double occ_max) {

    const int nbands = eps.size();

    if (sigma <= 0.0) {
        throw std::runtime_error("Fermi-Dirac sigma must be positive.");
    }

    if (nelec < -1.0e-12) {
        throw std::runtime_error("nelec cannot be negative.");
    }

    if (nelec > occ_max * nbands + 1.0e-12) {
        throw std::runtime_error(
            "Not enough bands for Fermi-Dirac occupations."
        );
    }

    const double eps_min = eps.minCoeff();
    const double eps_max = eps.maxCoeff();

    /*
     * Wide enough bracket for chemical potential.
     */
    double mu_low  = eps_min - std::max(1.0, 100.0 * sigma);
    double mu_high = eps_max + std::max(1.0, 100.0 * sigma);

    auto electron_count_at_mu = [&](double mu) {
        double nsum = 0.0;

        for (int i = 0; i < nbands; ++i) {
            nsum += fermi_dirac_occ_single(
                eps[i],
                mu,
                sigma,
                occ_max
            );
        }

        return nsum;
    };

    /*
     * Since electron count increases with mu:
     *
     *   if nsum > nelec, mu is too high
     *   if nsum < nelec, mu is too low
     */
    for (int it = 0; it < 200; ++it) {
        const double mu = 0.5 * (mu_low + mu_high);
        const double nsum = electron_count_at_mu(mu);

        if (nsum > nelec) {
            mu_high = mu;
        } else {
            mu_low = mu;
        }
    }

    const double mu = 0.5 * (mu_low + mu_high);

    OccupationResult out;
    out.occ.assign(nbands, 0.0);
    out.mu = mu;
    out.nelec_sum = 0.0;
    out.entropy = 0.0;

    for (int i = 0; i < nbands; ++i) {
        const double f =
            fermi_dirac_occ_single(
                eps[i],
                mu,
                sigma,
                occ_max
            );

        out.occ[i] = f;
        out.nelec_sum += f;

        /*
         * Entropy diagnostic:
         *
         *   S = -occ_max * sum [p log p + (1-p) log(1-p)]
         *
         * where p = f / occ_max.
         *
         * Units: k_B = 1.
         */
        const double p = f / occ_max;

        if (p > 1.0e-14 && p < 1.0 - 1.0e-14) {
            out.entropy +=
                -occ_max * (
                    p * std::log(p)
                    + (1.0 - p) * std::log(1.0 - p)
                );
        }
    }

    return out;
}

OccupationResult compute_occupations(
    const Eigen::VectorXd& eps,
    double nelec,
    OccupationMode mode,
    const std::vector<double>& fixed_occ,
    double sigma,
    double deg_tol) {

    if (mode == OccupationMode::Fixed) {
        if (static_cast<int>(fixed_occ.size()) != eps.size()) {
            throw std::runtime_error(
                "Fixed occupation size does not match number of bands."
            );
        }

        OccupationResult out;
        out.occ = fixed_occ;
        out.nelec_sum = sum_occupations(out.occ);

        /*
         * Set mu as a diagnostic only.
         */
        int last_occ = -1;
        for (int i = 0; i < eps.size(); ++i) {
            if (out.occ[i] > 1.0e-12) {
                last_occ = i;
            }
        }

        if (last_occ >= 0) {
            out.mu = eps[last_occ];
        } else {
            out.mu = eps[0];
        }

        return out;
    }

    if (mode == OccupationMode::DegeneracyAwareZeroT) {
        return degeneracy_aware_zeroT_occupations(
            eps,
            nelec,
            2.0,
            deg_tol
        );
    }

    if (mode == OccupationMode::FermiDirac) {
        return fermi_dirac_occupations(
            eps,
            nelec,
            sigma,
            2.0
        );
    }

    throw std::runtime_error("Unknown occupation mode.");
}
double compute_kinetic_energy(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations) {

    if (static_cast<int>(occupations.size()) != C.cols()) {
        throw std::runtime_error("Occupation size mismatch in kinetic energy.");
    }

    double T = 0.0;

    for (int ib = 0; ib < C.cols(); ++ib) {
        const double occ = occupations[ib];

        for (int ig = 0; ig < basis.size(); ++ig) {
            T += occ
               * std::norm(C(ig, ib))
               * basis.gvectors[ig].kinetic;
        }
    }

    return T;
}

EnergyTerms compute_total_energy(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& C,
    const std::vector<double>& occupations,
    const std::vector<double>& rho,
    const std::vector<double>& Vion,
    const std::vector<double>& VH,
    double Ex,
    double dV,
    double electronic_entropy,
    double sigma) {

    EnergyTerms e;

    e.kinetic =
        compute_kinetic_energy(basis, C, occupations);

    for (int p = 0; p < static_cast<int>(rho.size()); ++p) {
        e.external += dV * rho[p] * Vion[p];
        e.hartree  += 0.5 * dV * rho[p] * VH[p];
    }

    e.exchange = Ex;

    e.total =
        e.kinetic
        + e.external
        + e.hartree
        + e.exchange;

    e.electronic_entropy = electronic_entropy;
    e.entropy_correction = sigma * electronic_entropy;

    e.free_energy =
        e.total - e.entropy_correction;

    e.sigma0_estimate =
        0.5 * (e.total + e.free_energy);

    return e;
    return e;
}

double mix_density(
    std::vector<double>& rho,
    const std::vector<double>& rho_out,
    double alpha,
    double dV,
    double target_nelec) {

    if (rho.size() != rho_out.size()) {
        throw std::runtime_error("Density size mismatch in mixing.");
    }

    double diff2 = 0.0;

    for (int p = 0; p < static_cast<int>(rho.size()); ++p) {
        const double old = rho[p];
        const double mixed =
            (1.0 - alpha) * rho[p]
            + alpha * rho_out[p];

        const double diff = rho_out[p] - old;

        diff2 += diff * diff;

        rho[p] = mixed;
    }

    /*
     * Renormalize density to exact electron number.
     */
    const double nelec_now =
        electron_number_from_density(rho, dV);

    const double scale =
        target_nelec / nelec_now;

    for (double& x : rho) {
        x *= scale;
    }

    return std::sqrt(dV * diff2);
}

std::vector<std::complex<double>> build_density_G(
    FFTWorkspace& fft,
    const std::vector<double>& rho) {

    if (static_cast<int>(rho.size()) != fft.grid.ngrid) {
        throw std::runtime_error("rho size mismatch in build_density_G.");
    }

    const int ngrid = fft.grid.ngrid;

    for (int p = 0; p < ngrid; ++p) {
        fft.real_grid[p] = std::complex<double>(rho[p], 0.0);
    }

    fftw_execute(fft.forward_plan);

    std::vector<std::complex<double>> rho_G(
        ngrid,
        std::complex<double>(0.0, 0.0)
    );

    for (int p = 0; p < ngrid; ++p) {
        /*
         * FFTW forward is unnormalized.
         *
         * rho_G[p] is the Fourier series coefficient:
         *
         *   rho(G) = 1/Omega int rho(r) exp(-iG.r) dr
         *
         * On the uniform grid, this corresponds to forward_raw / Ngrid.
         */
        rho_G[p] =
            fft.forward_raw[p] / static_cast<double>(ngrid);
    }

    return rho_G;
}

double compute_external_energy(
    const std::vector<double>& rho,
    const std::vector<double>& V,
    double dV) {

    if (rho.size() != V.size()) {
        throw std::runtime_error("compute_external_energy size mismatch.");
    }

    double e = 0.0;

    for (int p = 0; p < static_cast<int>(rho.size()); ++p) {
        e += dV * rho[p] * V[p];
    }

    return e;
}