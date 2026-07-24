#pragma once

#include <Eigen/Dense>
#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>

struct Lattice {
    Eigen::Matrix3d A;
    Eigen::Matrix3d B;
    Eigen::Matrix3d A_inv;

    Lattice(const Eigen::Vector3d& a1,
            const Eigen::Vector3d& a2,
            const Eigen::Vector3d& a3) {
        A.col(0) = a1;
        A.col(1) = a2;
        A.col(2) = a3;

        const double detA = A.determinant();

        if (std::abs(detA) < 1.0e-14) {
            throw std::runtime_error("Lattice vectors are linearly dependent.");
        }

        A_inv = A.inverse();
        B = 2.0 * M_PI * A_inv.transpose();
    }

    double volume() const {
        return std::abs(A.determinant());
    }

    Eigen::Vector3d gvector_from_freq(const Eigen::Vector3i& n) const {
        return B * n.cast<double>();
    }

    Eigen::Vector3d cart_from_frac(const Eigen::Vector3d& s) const {
        return s[0] * A.col(0) + s[1] * A.col(1) + s[2] * A.col(2);
    }
};

struct FFTGrid {
    int n1;
    int n2;
    int n3;
    int ngrid;

    FFTGrid(int n1_in, int n2_in, int n3_in)
        : n1(n1_in), n2(n2_in), n3(n3_in) {
        if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
            throw std::runtime_error("FFT grid dimensions must be positive.");
        }
        ngrid = n1 * n2 * n3;
    }

    int index(int i, int j, int k) const {
        return (i * n2 + j) * n3 + k;
    }

    Eigen::Vector3d frac_coord(int i, int j, int k) const {
        return Eigen::Vector3d(
            static_cast<double>(i) / static_cast<double>(n1),
            static_cast<double>(j) / static_cast<double>(n2),
            static_cast<double>(k) / static_cast<double>(n3)
        );
    }

    static int freq_index(int idx, int N) {
        return (idx <= N / 2) ? idx : idx - N;
    }

    Eigen::Vector3i freq_from_indices(int i, int j, int k) const {
        return Eigen::Vector3i(
            freq_index(i, n1),
            freq_index(j, n2),
            freq_index(k, n3)
        );
    }

    static int fft_index_from_freq(int n, int N) {
        if (n >= 0) {
            if (n > N / 2) {
                throw std::runtime_error("Positive frequency exceeds FFT grid.");
            }
            return n;
        } else {
            if (-n > N / 2) {
                throw std::runtime_error("Negative frequency reaches/exceeds Nyquist boundary.");
            }
            return N + n;
        }
    }

    int index_from_freq(const Eigen::Vector3i& n) const {
        const int i = fft_index_from_freq(n[0], n1);
        const int j = fft_index_from_freq(n[1], n2);
        const int k = fft_index_from_freq(n[2], n3);
        return index(i, j, k);
    }
};

struct GVector {
    Eigen::Vector3i n;
    Eigen::Vector3d G_cart;
    Eigen::Vector3d q_cart;
    double kinetic;
};

class PlaneWaveBasis3D {
public:
    Eigen::Vector3d k_cart = Eigen::Vector3d::Zero();
    std::vector<GVector> gvectors;

    void generate(const Lattice& lattice,
                  const Eigen::Vector3d& k_cart,
                  double ecut) {
        if (ecut <= 0.0) {
            throw std::runtime_error("ecut must be positive.");
        }

        gvectors.clear();
        this->k_cart = k_cart;

        const double pmax = std::sqrt(2.0 * ecut);
        const double qmax = pmax + k_cart.norm();

        const Eigen::Matrix3d M = lattice.B.transpose() * lattice.B;
        const Eigen::Matrix3d M_inv = M.inverse();

        Eigen::Vector3i nmax;

        for (int i = 0; i < 3; ++i) {
            nmax[i] =
                static_cast<int>(std::ceil(qmax * std::sqrt(M_inv(i, i)))) + 1;
        }

        const double tol = 1.0e-12;

        for (int n1 = -nmax[0]; n1 <= nmax[0]; ++n1) {
            for (int n2 = -nmax[1]; n2 <= nmax[1]; ++n2) {
                for (int n3 = -nmax[2]; n3 <= nmax[2]; ++n3) {
                    Eigen::Vector3i n(n1, n2, n3);
                    Eigen::Vector3d G = lattice.gvector_from_freq(n);
                    Eigen::Vector3d q = k_cart + G;

                    const double kinetic = 0.5 * q.squaredNorm();

                    if (kinetic <= ecut + tol) {
                        gvectors.push_back({n, G, q, kinetic});
                    }
                }
            }
        }

        std::sort(gvectors.begin(), gvectors.end(),
                  [](const GVector& a, const GVector& b) {
                      if (std::abs(a.kinetic - b.kinetic) > 1.0e-14) {
                          return a.kinetic < b.kinetic;
                      }
                      if (a.n[0] != b.n[0]) return a.n[0] < b.n[0];
                      if (a.n[1] != b.n[1]) return a.n[1] < b.n[1];
                      return a.n[2] < b.n[2];
                  });
    }

    int size() const {
        return static_cast<int>(gvectors.size());
    }

    int find_index_by_n(const Eigen::Vector3i& target) const {
        for (int i = 0; i < size(); ++i) {
            if (gvectors[i].n == target) {
                return i;
            }
        }
        return -1;
    }
};

struct FFTPerformanceCounters {
    long long hamiltonian_vectors = 0;
    long long hamiltonian_block_calls = 0;
    double hamiltonian_scatter_seconds = 0.0;
    double hamiltonian_backward_fft_seconds = 0.0;
    double hamiltonian_local_multiply_seconds = 0.0;
    double hamiltonian_forward_fft_seconds = 0.0;
    double hamiltonian_gather_kinetic_seconds = 0.0;
    double hamiltonian_nonlocal_seconds = 0.0;

    long long density_orbitals = 0;
    double density_scatter_seconds = 0.0;
    double density_backward_fft_seconds = 0.0;
    double density_accumulation_seconds = 0.0;

    FFTPerformanceCounters delta_from(
        const FFTPerformanceCounters& earlier) const;
};

class FFTWorkspace {
public:
    struct BatchPlans {
        fftw_plan backward = nullptr;
        fftw_plan forward = nullptr;
    };

    FFTGrid grid;

    std::vector<std::complex<double>> reciprocal_grid;
    std::vector<std::complex<double>> real_grid;
    std::vector<std::complex<double>> forward_raw;

    fftw_plan backward_plan = nullptr;
    fftw_plan forward_plan = nullptr;

    /*
     * Batched transforms share one pair of buffers.  Plans are cached by
     * transform count because fftw_plan_many_dft fixes `howmany` when the
     * plan is created.  If the buffers need to grow, all cached plans are
     * destroyed before reallocation so they never retain stale pointers.
     */
    std::vector<std::complex<double>> batch_reciprocal_grid;
    std::vector<std::complex<double>> batch_real_grid;
    int batch_capacity = 0;
    std::map<int, BatchPlans> batch_plan_cache;

    int thread_count = 1;
    FFTPerformanceCounters performance;

    explicit FFTWorkspace(
        const FFTGrid& grid_in,
        int requested_thread_count = 1);

    ~FFTWorkspace();

    FFTWorkspace(const FFTWorkspace&) = delete;
    FFTWorkspace& operator=(const FFTWorkspace&) = delete;

    const BatchPlans& ensure_batch_plans(int transform_count);

private:
    void destroy_batch_plans();
};

struct NonlocalProjector {
    Eigen::VectorXcd beta_G;
    double D = 0.0;

    /*
     * Index of the ion carrying this projector.  The translational
     * derivative d beta(G+k) / d R_I = -i (G+k) beta(G+k) contributes
     * only to this ion's force.
     */
    int ion_index = -1;
};
