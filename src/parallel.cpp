#include "parallel.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

#ifdef PWDFT_USE_MPI
#include <mpi.h>
#endif

namespace parallel {
namespace {

#ifdef PWDFT_USE_MPI
bool mpi_runtime_active() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return false;
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    return finalized == 0;
}

int checked_count(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI buffer is too large for an MPI count.");
    }
    return static_cast<int>(count);
}
#endif

} // namespace

Environment::Environment(int& argc, char**& argv) {
#ifdef PWDFT_USE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = MPI_THREAD_SINGLE;
        const int error = MPI_Init_thread(
            &argc, &argv, MPI_THREAD_FUNNELED, &provided
        );
        if (error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed.");
        }
        owns_runtime_ = true;
        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            owns_runtime_ = false;
            throw std::runtime_error(
                "MPI does not provide the required FUNNELED thread level."
            );
        }
    }
#else
    (void)argc;
    (void)argv;
#endif
}

Environment::~Environment() {
#ifdef PWDFT_USE_MPI
    if (owns_runtime_ && mpi_runtime_active()) {
        MPI_Finalize();
    }
#endif
}

bool active() {
#ifdef PWDFT_USE_MPI
    return mpi_runtime_active();
#else
    return false;
#endif
}

int rank() {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        int value = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &value);
        return value;
    }
#endif
    return 0;
}

int size() {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        int value = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &value);
        return value;
    }
#endif
    return 1;
}

bool is_root() {
    return rank() == 0;
}

KPointDistribution::KPointDistribution(int kpoint_count)
    : kpoint_count_(kpoint_count), rank_(parallel::rank()),
      size_(parallel::size()) {

    if (kpoint_count_ < 0) {
        throw std::runtime_error("K-point count cannot be negative.");
    }
    for (int ik = rank_; ik < kpoint_count_; ik += size_) {
        local_kpoints_.push_back(ik);
    }
}

bool KPointDistribution::owns(int kpoint_index) const {
    if (kpoint_index < 0 || kpoint_index >= kpoint_count_) {
        return false;
    }
    return owner(kpoint_index) == rank_;
}

int KPointDistribution::owner(int kpoint_index) const {
    if (kpoint_index < 0 || kpoint_index >= kpoint_count_) {
        throw std::runtime_error("K-point index is out of range.");
    }
    return kpoint_index % size_;
}

const std::vector<int>& KPointDistribution::local_kpoints() const {
    return local_kpoints_;
}

int KPointDistribution::rank() const {
    return rank_;
}

int KPointDistribution::size() const {
    return size_;
}

int sum(int value) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        int result = 0;
        MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
#endif
    return value;
}

long long sum(long long value) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        long long result = 0;
        MPI_Allreduce(
            &value, &result, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD
        );
        return result;
    }
#endif
    return value;
}

double sum(double value) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        double result = 0.0;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
#endif
    return value;
}

double maximum(double value) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        double result = 0.0;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return result;
    }
#endif
    return value;
}

void sum_in_place(std::vector<double>& values) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active() && !values.empty()) {
        MPI_Allreduce(
            MPI_IN_PLACE,
            values.data(),
            checked_count(values.size()),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );
    }
#else
    (void)values;
#endif
}

std::string first_error(const std::string& local_message) {
#ifdef PWDFT_USE_MPI
    if (mpi_runtime_active()) {
        const int process_count = size();
        const int candidate = local_message.empty() ? process_count : rank();
        int first_failing_rank = process_count;
        MPI_Allreduce(
            &candidate,
            &first_failing_rank,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD
        );
        if (first_failing_rank == process_count) {
            return {};
        }

        int length = rank() == first_failing_rank
            ? checked_count(local_message.size())
            : 0;
        MPI_Bcast(
            &length, 1, MPI_INT, first_failing_rank, MPI_COMM_WORLD
        );
        std::string message(static_cast<std::size_t>(length), '\0');
        if (rank() == first_failing_rank) {
            message = local_message;
        }
        if (length > 0) {
            MPI_Bcast(
                message.data(),
                length,
                MPI_CHAR,
                first_failing_rank,
                MPI_COMM_WORLD
            );
        }
        return message;
    }
#endif
    return local_message;
}

} // namespace parallel
