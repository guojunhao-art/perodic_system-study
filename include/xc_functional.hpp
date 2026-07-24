#pragma once

#include <memory>
#include <string>
#include <vector>

enum class LDAFunctional {
    ExchangeOnly,
    PerdewZunger
};

struct XCResult {
    std::vector<double> Vxc;
    double exchange_energy = 0.0;
    double correlation_energy = 0.0;

    double total_energy() const {
        return exchange_energy + correlation_energy;
    }
};

struct SpinXCResult {
    std::vector<double> Vxc_up;
    std::vector<double> Vxc_down;
    double exchange_energy = 0.0;
    double correlation_energy = 0.0;

    double total_energy() const {
        return exchange_energy + correlation_energy;
    }
};

/*
 * Unpolarized or collinear spin-polarized LDA wrapper around LibXC.
 *
 * LibXC functional objects are initialized once and reused for every SCF
 * iteration. The implementation is hidden so public headers do not expose
 * LibXC's C data structures.
 */
class LibXCLDAFunctional {
public:
    explicit LibXCLDAFunctional(
        LDAFunctional functional,
        int nspin = 1);
    ~LibXCLDAFunctional();

    LibXCLDAFunctional(const LibXCLDAFunctional&) = delete;
    LibXCLDAFunctional& operator=(const LibXCLDAFunctional&) = delete;
    LibXCLDAFunctional(LibXCLDAFunctional&&) noexcept;
    LibXCLDAFunctional& operator=(LibXCLDAFunctional&&) noexcept;

    XCResult evaluate(
        const std::vector<double>& density,
        double dV) const;

    SpinXCResult evaluate_spin(
        const std::vector<double>& density_up,
        const std::vector<double>& density_down,
        double dV) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string lda_functional_name(LDAFunctional functional);
std::string libxc_runtime_version();
