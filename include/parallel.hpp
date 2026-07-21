#pragma once

#include <string>
#include <vector>

/*
 * Thin process-level parallel layer.
 *
 * A non-MPI build, or an MPI build in which Environment has not been
 * constructed, behaves exactly like a one-rank calculation. Keeping MPI
 * calls behind this interface prevents communicator details from leaking
 * through the SCF implementation.
 */
namespace parallel {

class Environment {
public:
    Environment(int& argc, char**& argv);
    ~Environment();

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

private:
    bool owns_runtime_ = false;
};

bool active();
int rank();
int size();
bool is_root();

class KPointDistribution {
public:
    explicit KPointDistribution(int kpoint_count);

    bool owns(int kpoint_index) const;
    int owner(int kpoint_index) const;
    const std::vector<int>& local_kpoints() const;

    int rank() const;
    int size() const;

private:
    int kpoint_count_ = 0;
    int rank_ = 0;
    int size_ = 1;
    std::vector<int> local_kpoints_;
};

int sum(int value);
long long sum(long long value);
double sum(double value);
double maximum(double value);
void sum_in_place(std::vector<double>& values);

/*
 * Return an empty string if every rank succeeded. Otherwise return the
 * message from the lowest-numbered failing rank on every rank.
 */
std::string first_error(const std::string& local_message);

} // namespace parallel
