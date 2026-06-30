#include "hamiltonian.hpp"
#include "potentials.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

void scatter_coeffs_to_fft_grid(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& coeffs,
    std::vector<std::complex<double>>& reciprocal_grid) {

    if (static_cast<int>(coeffs.size()) != basis.size()) {
        throw std::runtime_error("Coefficient size does not match basis size.");
    }

    std::fill(reciprocal_grid.begin(),
              reciprocal_grid.end(),
              std::complex<double>(0.0, 0.0));

    for (int ig = 0; ig < basis.size(); ++ig) {
        const int p = grid.index_from_freq(basis.gvectors[ig].n);
        reciprocal_grid[p] = coeffs[ig];
    }
}

void gather_coeffs_from_fft_grid(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& reciprocal_grid,
    std::vector<std::complex<double>>& coeffs) {

    coeffs.assign(basis.size(), {0.0, 0.0});

    for (int ig = 0; ig < basis.size(); ++ig) {
        const int p = grid.index_from_freq(basis.gvectors[ig].n);
        coeffs[ig] = reciprocal_grid[p];
    }
}

std::vector<std::complex<double>> apply_local_potential_fft(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const std::vector<std::complex<double>>& coeffs) {

    if (static_cast<int>(V_r.size()) != fft.grid.ngrid) {
        throw std::runtime_error("V_r size does not match FFT grid.");
    }

    // 1. c(G) -> FFT reciprocal grid
    scatter_coeffs_to_fft_grid(
        basis,
        fft.grid,
        coeffs,
        fft.reciprocal_grid
    );

    // 2. inverse FFT: c(G) -> u_tilde(r)
    fftw_execute(fft.backward_plan);

    // 3. multiply in real space: V(r) * u_tilde(r)
    for (int p = 0; p < fft.grid.ngrid; ++p) {
        fft.real_grid[p] *= V_r[p];
    }

    // 4. forward FFT: V(r)u(r) -> raw reciprocal coefficients
    fftw_execute(fft.forward_plan);

    // 5. divide by Ngrid to get mathematical Fourier coefficients
    for (int p = 0; p < fft.grid.ngrid; ++p) {
        fft.reciprocal_grid[p] =
            fft.forward_raw[p] / static_cast<double>(fft.grid.ngrid);
    }

    // 6. gather back to plane-wave basis
    std::vector<std::complex<double>> Vc;
    gather_coeffs_from_fft_grid(
        basis,
        fft.grid,
        fft.reciprocal_grid,
        Vc
    );

    return Vc;
}

std::vector<std::complex<double>> apply_hamiltonian_fft(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const std::vector<std::complex<double>>& coeffs) {

    std::vector<std::complex<double>> result =
        apply_local_potential_fft(basis, fft, V_r, coeffs);

    // Add kinetic part.
    for (int ig = 0; ig < basis.size(); ++ig) {
        result[ig] += basis.gvectors[ig].kinetic * coeffs[ig];
    }

    return result;
}

std::vector<std::complex<double>> forward_fft_real_potential(
    FFTWorkspace& fft,
    const std::vector<double>& V_r) {

    if (static_cast<int>(V_r.size()) != fft.grid.ngrid) {
        throw std::runtime_error("V_r size does not match FFT grid.");
    }

    for (int p = 0; p < fft.grid.ngrid; ++p) {
        fft.real_grid[p] = std::complex<double>(V_r[p], 0.0);
    }

    fftw_execute(fft.forward_plan);

    std::vector<std::complex<double>> V_G(fft.grid.ngrid);

    for (int p = 0; p < fft.grid.ngrid; ++p) {
        V_G[p] =
            fft.forward_raw[p] / static_cast<double>(fft.grid.ngrid);
    }

    return V_G;
}

bool fft_grid_can_represent_freq(
    const Eigen::Vector3i& n,
    const FFTGrid& grid) {

    auto ok_one_dim = [](int ni, int N) {
        /*
         * Conservative safe range for even N:
         *
         *   -N/2 < ni < N/2
         *
         * This avoids the Nyquist boundary, which can be ambiguous
         * in some real-grid / Hermitian-symmetry contexts.
         */
        return (-N / 2 < ni) && (ni < N / 2);
    };

    return ok_one_dim(n[0], grid.n1)
        && ok_one_dim(n[1], grid.n2)
        && ok_one_dim(n[2], grid.n3);
}

Eigen::MatrixXcd build_explicit_hamiltonian(
    const PlaneWaveBasis3D& basis,
    const FFTGrid& grid,
    const std::vector<std::complex<double>>& V_G) {

    if (static_cast<int>(V_G.size()) != grid.ngrid) {
        throw std::runtime_error("V_G size does not match FFT grid.");
    }

    const int nbasis = basis.size();

    Eigen::MatrixXcd H =
        Eigen::MatrixXcd::Zero(nbasis, nbasis);

    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j) {

            std::complex<double> hij(0.0, 0.0);

            if (i == j) {
                hij += basis.gvectors[i].kinetic;
            }

            const Eigen::Vector3i dn =
                basis.gvectors[i].n - basis.gvectors[j].n;

            if (!fft_grid_can_represent_freq(dn, grid)) {
                std::cerr << "Cannot represent dn = "
                          << dn.transpose()
                          << " on FFT grid.\n";
                throw std::runtime_error(
                    "FFT grid too small for Hamiltonian construction."
                );
            }

            const int p = grid.index_from_freq(dn);

            hij += V_G[p];

            H(i, j) = hij;
        }
    }

    return H;
}

double hermiticity_error(const Eigen::MatrixXcd& H) {
    return (H - H.adjoint()).norm();
}

std::vector<std::complex<double>> eigen_to_std_vector(
    const Eigen::VectorXcd& x) {

    std::vector<std::complex<double>> v(x.size());

    for (int i = 0; i < x.size(); ++i) {
        v[i] = x[i];
    }

    return v;
}

Eigen::VectorXcd std_to_eigen_vector(
    const std::vector<std::complex<double>>& v) {

    Eigen::VectorXcd x(v.size());

    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        x[i] = v[i];
    }

    return x;
}

Eigen::VectorXcd apply_hamiltonian_eigen(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::VectorXcd& c,
    const std::vector<NonlocalProjector>* projectors) {

    const auto c_std = eigen_to_std_vector(c);

    const auto Hc_std =
        apply_hamiltonian_fft(basis, fft, V_r, c_std);

    Eigen::VectorXcd Hc = std_to_eigen_vector(Hc_std);
    if (projectors != nullptr) {
        Hc += apply_nonlocal_projectors(
            *projectors,
            c
        );
    }

    return Hc;
    
}

Eigen::MatrixXcd apply_hamiltonian_to_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& V,
    const std::vector<NonlocalProjector>* projectors) {

    const int nrows = V.rows();
    const int ncols = V.cols();

    Eigen::MatrixXcd W(nrows, ncols);

    for (int j = 0; j < ncols; ++j) {
        W.col(j) = apply_hamiltonian_eigen(
            basis,
            fft,
            V_r,
            V.col(j),
            projectors
        );
    }

    return W;
}