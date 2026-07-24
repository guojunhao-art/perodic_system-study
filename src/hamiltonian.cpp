#include "hamiltonian.hpp"
#include "potentials.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using PerformanceClock = std::chrono::steady_clock;
constexpr long long openmp_minimum_work = 32768;

double elapsed_seconds(PerformanceClock::time_point start) {
    return std::chrono::duration<double>(
        PerformanceClock::now() - start
    ).count();
}

} // namespace

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
    auto stage_start = PerformanceClock::now();
    const int nbasis = basis.size();
    const int ngrid = fft.grid.ngrid;
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int p = 0; p < ngrid; ++p) {
        fft.reciprocal_grid[p] = std::complex<double>(0.0, 0.0);
    }
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && nbasis >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int ig = 0; ig < nbasis; ++ig) {
        const int p = fft.grid.index_from_freq(basis.gvectors[ig].n);
        fft.reciprocal_grid[p] = coeffs[ig];
    }
    fft.performance.hamiltonian_scatter_seconds +=
        elapsed_seconds(stage_start);

    // 2. inverse FFT: c(G) -> u_tilde(r)
    stage_start = PerformanceClock::now();
    fftw_execute(fft.backward_plan);
    fft.performance.hamiltonian_backward_fft_seconds +=
        elapsed_seconds(stage_start);

    // 3. multiply in real space: V(r) * u_tilde(r)
    stage_start = PerformanceClock::now();
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && ngrid >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int p = 0; p < ngrid; ++p) {
        fft.real_grid[p] *= V_r[p];
    }
    fft.performance.hamiltonian_local_multiply_seconds +=
        elapsed_seconds(stage_start);

    // 4. forward FFT: V(r)u(r) -> raw reciprocal coefficients
    stage_start = PerformanceClock::now();
    fftw_execute(fft.forward_plan);
    fft.performance.hamiltonian_forward_fft_seconds +=
        elapsed_seconds(stage_start);

    // 5. Normalize and gather back to the plane-wave basis.
    stage_start = PerformanceClock::now();
    const double inverse_grid_size = 1.0 / static_cast<double>(ngrid);
    std::vector<std::complex<double>> Vc(nbasis);
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && nbasis >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int ig = 0; ig < nbasis; ++ig) {
        const int p = fft.grid.index_from_freq(basis.gvectors[ig].n);
        Vc[ig] = fft.forward_raw[p] * inverse_grid_size;
    }
    fft.performance.hamiltonian_gather_kinetic_seconds +=
        elapsed_seconds(stage_start);

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
    const auto stage_start = PerformanceClock::now();
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && basis.size() >= openmp_minimum_work) \
    num_threads(fft.thread_count)
    for (int ig = 0; ig < basis.size(); ++ig) {
        result[ig] += basis.gvectors[ig].kinetic * coeffs[ig];
    }
    fft.performance.hamiltonian_gather_kinetic_seconds +=
        elapsed_seconds(stage_start);
    fft.performance.hamiltonian_vectors += 1;
    fft.performance.hamiltonian_block_calls += 1;

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
        const auto stage_start = PerformanceClock::now();
        Hc += apply_nonlocal_projectors(
            *projectors,
            c
        );
        fft.performance.hamiltonian_nonlocal_seconds +=
            elapsed_seconds(stage_start);
    }

    return Hc;
    
}

Eigen::MatrixXcd apply_hamiltonian_to_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& V,
    const std::vector<NonlocalProjector>* projectors) {

    if (V.rows() != basis.size()) {
        throw std::runtime_error(
            "Hamiltonian block row count does not match basis size."
        );
    }
    if (static_cast<int>(V_r.size()) != fft.grid.ngrid) {
        throw std::runtime_error("V_r size does not match FFT grid.");
    }

    const int nrows = V.rows();
    const int ncols = V.cols();
    Eigen::MatrixXcd W(nrows, ncols);
    if (ncols == 0) {
        return W;
    }

    /*
     * Bound temporary memory for calculations with many bands.  At the
     * current small-system scale Davidson blocks normally fit in one chunk.
     */
    constexpr int maximum_fft_batch = 16;
    const double inverse_grid_size =
        1.0 / static_cast<double>(fft.grid.ngrid);
    std::vector<int> grid_indices(basis.size(), 0);
    for (int ig = 0; ig < basis.size(); ++ig) {
        grid_indices[ig] =
            fft.grid.index_from_freq(basis.gvectors[ig].n);
    }

    for (int first = 0; first < ncols; first += maximum_fft_batch) {
        const int count = std::min(maximum_fft_batch, ncols - first);
        const auto& plans = fft.ensure_batch_plans(count);
        const std::size_t active_size =
            static_cast<std::size_t>(count)
            * static_cast<std::size_t>(fft.grid.ngrid);

        auto stage_start = PerformanceClock::now();
#pragma omp parallel for schedule(static) \
    if(fft.thread_count > 1 && \
       static_cast<long long>(active_size) >= openmp_minimum_work) \
    num_threads(fft.thread_count)
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(active_size);
             ++index) {
            fft.batch_reciprocal_grid[
                static_cast<std::size_t>(index)
            ] = std::complex<double>(0.0, 0.0);
        }

#pragma omp parallel for collapse(2) schedule(static) \
    if(fft.thread_count > 1 && \
       static_cast<long long>(count) * basis.size() >= \
           openmp_minimum_work) \
    num_threads(fft.thread_count)
        for (int local_col = 0; local_col < count; ++local_col) {
            for (int ig = 0; ig < basis.size(); ++ig) {
                const std::size_t offset =
                    static_cast<std::size_t>(local_col)
                    * static_cast<std::size_t>(fft.grid.ngrid);
                const int p = grid_indices[ig];
                fft.batch_reciprocal_grid[offset + p] =
                    V(ig, first + local_col);
            }
        }
        fft.performance.hamiltonian_scatter_seconds +=
            elapsed_seconds(stage_start);

        stage_start = PerformanceClock::now();
        fftw_execute(plans.backward);
        fft.performance.hamiltonian_backward_fft_seconds +=
            elapsed_seconds(stage_start);

        stage_start = PerformanceClock::now();
#pragma omp parallel for collapse(2) schedule(static) \
    if(fft.thread_count > 1 && \
       static_cast<long long>(count) * fft.grid.ngrid >= \
           openmp_minimum_work) \
    num_threads(fft.thread_count)
        for (int local_col = 0; local_col < count; ++local_col) {
            for (int p = 0; p < fft.grid.ngrid; ++p) {
                const std::size_t offset =
                    static_cast<std::size_t>(local_col)
                    * static_cast<std::size_t>(fft.grid.ngrid);
                fft.batch_real_grid[offset + p] *= V_r[p];
            }
        }
        fft.performance.hamiltonian_local_multiply_seconds +=
            elapsed_seconds(stage_start);

        stage_start = PerformanceClock::now();
        fftw_execute(plans.forward);
        fft.performance.hamiltonian_forward_fft_seconds +=
            elapsed_seconds(stage_start);

        stage_start = PerformanceClock::now();
#pragma omp parallel for collapse(2) schedule(static) \
    if(fft.thread_count > 1 && \
       static_cast<long long>(count) * basis.size() >= \
           openmp_minimum_work) \
    num_threads(fft.thread_count)
        for (int local_col = 0; local_col < count; ++local_col) {
            for (int ig = 0; ig < basis.size(); ++ig) {
                const int column = first + local_col;
                const std::size_t offset =
                    static_cast<std::size_t>(local_col)
                    * static_cast<std::size_t>(fft.grid.ngrid);
                const int p = grid_indices[ig];
                W(ig, column) =
                    fft.batch_reciprocal_grid[offset + p]
                    * inverse_grid_size
                    + basis.gvectors[ig].kinetic * V(ig, column);
            }
        }
        fft.performance.hamiltonian_gather_kinetic_seconds +=
            elapsed_seconds(stage_start);
    }

    if (projectors != nullptr) {
        const auto stage_start = PerformanceClock::now();
        W += apply_nonlocal_projectors(*projectors, V);
        fft.performance.hamiltonian_nonlocal_seconds +=
            elapsed_seconds(stage_start);
    }
    fft.performance.hamiltonian_vectors += ncols;
    fft.performance.hamiltonian_block_calls += 1;

    return W;
}
