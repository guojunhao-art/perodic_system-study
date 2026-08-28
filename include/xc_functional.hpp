#pragma once

#include "core.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

enum class XCFunctional {
    ExchangeOnly,
    PerdewZunger,
    PerdewBurkeErnzerhof
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
 * Periodic spectral derivatives used by GGA functionals.  FFTW's forward
 * transform is normalized to Fourier-series coefficients before multiplying
 * by iG.  Nyquist components are removed direction by direction so the
 * discrete derivative remains real and skew-adjoint on even grids.
 *
 * The returned components are Cartesian derivatives in Bohr^-4 for a density
 * supplied in Bohr^-3.
 */
std::array<std::vector<double>, 3> spectral_gradient(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& field);

std::vector<double> spectral_divergence(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::array<std::vector<double>, 3>& vector_field);

/*
 * Unpolarized LDA/GGA or collinear spin-polarized LDA wrapper around LibXC.
 *
 * LibXC functional objects are initialized once and reused for every SCF
 * iteration. The implementation is hidden so public headers do not expose
 * LibXC's C data structures.  The first PBE implementation is deliberately
 * restricted to nspin = 1; the existing PZ-LDA path continues to support
 * nspin = 1 and 2.
 */
class LibXCFunctional {
public:
    explicit LibXCFunctional(
        XCFunctional functional,
        int nspin = 1);
    ~LibXCFunctional();

    LibXCFunctional(const LibXCFunctional&) = delete;
    LibXCFunctional& operator=(const LibXCFunctional&) = delete;
    LibXCFunctional(LibXCFunctional&&) noexcept;
    LibXCFunctional& operator=(LibXCFunctional&&) noexcept;

    XCResult evaluate(
        const Lattice& lattice,
        FFTWorkspace& fft,
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

std::string xc_functional_name(XCFunctional functional);
bool pseudopotential_functional_matches_xc(
    const std::string& label,
    XCFunctional functional);
std::string libxc_runtime_version();
