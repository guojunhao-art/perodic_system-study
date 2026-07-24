#include "core.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::mutex& fftw_planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

int automatic_thread_count() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    const unsigned int detected = std::thread::hardware_concurrency();
    return detected == 0 ? 1 : static_cast<int>(detected);
#endif
}

int resolve_thread_count(int requested_thread_count) {
    if (requested_thread_count < 0) {
        throw std::runtime_error(
            "FFT thread count must be auto or a positive integer."
        );
    }

    const int resolved = requested_thread_count == 0
        ? automatic_thread_count()
        : requested_thread_count;

#ifndef PWDFT_USE_FFTW_THREADS
    if (requested_thread_count == 0) {
        return 1;
    }
    if (resolved > 1) {
        throw std::runtime_error(
            "This build does not include threaded FFTW support."
        );
    }
#endif
    return resolved;
}

void configure_fftw_planner_threads(int thread_count) {
#ifdef PWDFT_USE_FFTW_THREADS
    static std::once_flag initialization_flag;
    static bool initialization_succeeded = false;
    std::call_once(initialization_flag, []() {
        initialization_succeeded = fftw_init_threads() != 0;
    });
    if (!initialization_succeeded) {
        throw std::runtime_error("FFTW thread initialization failed.");
    }
    fftw_plan_with_nthreads(thread_count);
#else
    (void)thread_count;
#endif
}

} // namespace

FFTPerformanceCounters FFTPerformanceCounters::delta_from(
    const FFTPerformanceCounters& earlier) const {

    FFTPerformanceCounters delta;
    delta.hamiltonian_vectors =
        hamiltonian_vectors - earlier.hamiltonian_vectors;
    delta.hamiltonian_block_calls =
        hamiltonian_block_calls - earlier.hamiltonian_block_calls;
    delta.hamiltonian_scatter_seconds =
        hamiltonian_scatter_seconds - earlier.hamiltonian_scatter_seconds;
    delta.hamiltonian_backward_fft_seconds =
        hamiltonian_backward_fft_seconds
        - earlier.hamiltonian_backward_fft_seconds;
    delta.hamiltonian_local_multiply_seconds =
        hamiltonian_local_multiply_seconds
        - earlier.hamiltonian_local_multiply_seconds;
    delta.hamiltonian_forward_fft_seconds =
        hamiltonian_forward_fft_seconds
        - earlier.hamiltonian_forward_fft_seconds;
    delta.hamiltonian_gather_kinetic_seconds =
        hamiltonian_gather_kinetic_seconds
        - earlier.hamiltonian_gather_kinetic_seconds;
    delta.hamiltonian_nonlocal_seconds =
        hamiltonian_nonlocal_seconds - earlier.hamiltonian_nonlocal_seconds;
    delta.density_orbitals =
        density_orbitals - earlier.density_orbitals;
    delta.density_scatter_seconds =
        density_scatter_seconds - earlier.density_scatter_seconds;
    delta.density_backward_fft_seconds =
        density_backward_fft_seconds - earlier.density_backward_fft_seconds;
    delta.density_accumulation_seconds =
        density_accumulation_seconds - earlier.density_accumulation_seconds;
    return delta;
}

FFTWorkspace::FFTWorkspace(
    const FFTGrid& grid_in,
    int requested_thread_count)
    : grid(grid_in),
      reciprocal_grid(grid.ngrid, {0.0, 0.0}),
      real_grid(grid.ngrid, {0.0, 0.0}),
      forward_raw(grid.ngrid, {0.0, 0.0}),
      thread_count(resolve_thread_count(requested_thread_count)) {

    Eigen::setNbThreads(thread_count);
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    configure_fftw_planner_threads(thread_count);

    backward_plan = fftw_plan_dft_3d(
        grid.n1, grid.n2, grid.n3,
        reinterpret_cast<fftw_complex*>(reciprocal_grid.data()),
        reinterpret_cast<fftw_complex*>(real_grid.data()),
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    forward_plan = fftw_plan_dft_3d(
        grid.n1, grid.n2, grid.n3,
        reinterpret_cast<fftw_complex*>(real_grid.data()),
        reinterpret_cast<fftw_complex*>(forward_raw.data()),
        FFTW_FORWARD,
        FFTW_ESTIMATE
    );

    if (!backward_plan || !forward_plan) {
        if (backward_plan) {
            fftw_destroy_plan(backward_plan);
            backward_plan = nullptr;
        }
        if (forward_plan) {
            fftw_destroy_plan(forward_plan);
            forward_plan = nullptr;
        }
        throw std::runtime_error("Failed to create FFTW plans.");
    }
}

FFTWorkspace::~FFTWorkspace() {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    destroy_batch_plans();
    if (backward_plan) {
        fftw_destroy_plan(backward_plan);
    }
    if (forward_plan) {
        fftw_destroy_plan(forward_plan);
    }
}

const FFTWorkspace::BatchPlans& FFTWorkspace::ensure_batch_plans(
    int transform_count) {

    if (transform_count <= 0) {
        throw std::runtime_error(
            "Batched FFT transform count must be positive."
        );
    }

    if (transform_count > batch_capacity) {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        destroy_batch_plans();
        batch_capacity = transform_count;
        const std::size_t buffer_size =
            static_cast<std::size_t>(batch_capacity)
            * static_cast<std::size_t>(grid.ngrid);
        batch_reciprocal_grid.assign(
            buffer_size,
            std::complex<double>(0.0, 0.0)
        );
        batch_real_grid.assign(
            buffer_size,
            std::complex<double>(0.0, 0.0)
        );
    }

    const auto found = batch_plan_cache.find(transform_count);
    if (found != batch_plan_cache.end()) {
        return found->second;
    }

    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    configure_fftw_planner_threads(thread_count);

    int dimensions[3] = {grid.n1, grid.n2, grid.n3};
    BatchPlans plans;
    plans.backward = fftw_plan_many_dft(
        3,
        dimensions,
        transform_count,
        reinterpret_cast<fftw_complex*>(batch_reciprocal_grid.data()),
        nullptr,
        1,
        grid.ngrid,
        reinterpret_cast<fftw_complex*>(batch_real_grid.data()),
        nullptr,
        1,
        grid.ngrid,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );
    plans.forward = fftw_plan_many_dft(
        3,
        dimensions,
        transform_count,
        reinterpret_cast<fftw_complex*>(batch_real_grid.data()),
        nullptr,
        1,
        grid.ngrid,
        reinterpret_cast<fftw_complex*>(batch_reciprocal_grid.data()),
        nullptr,
        1,
        grid.ngrid,
        FFTW_FORWARD,
        FFTW_ESTIMATE
    );

    if (!plans.backward || !plans.forward) {
        if (plans.backward) {
            fftw_destroy_plan(plans.backward);
        }
        if (plans.forward) {
            fftw_destroy_plan(plans.forward);
        }
        throw std::runtime_error("Failed to create batched FFTW plans.");
    }

    const auto inserted = batch_plan_cache.emplace(
        transform_count,
        plans
    );
    return inserted.first->second;
}

void FFTWorkspace::destroy_batch_plans() {
    for (auto& entry : batch_plan_cache) {
        if (entry.second.backward) {
            fftw_destroy_plan(entry.second.backward);
        }
        if (entry.second.forward) {
            fftw_destroy_plan(entry.second.forward);
        }
    }
    batch_plan_cache.clear();
}
