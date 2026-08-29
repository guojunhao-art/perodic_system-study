#include "xc_functional.hpp"

#include <xc.h>
#include <xc_funcs.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {

constexpr double negative_density_tolerance = 1.0e-12;
constexpr double inverse_fft_imaginary_tolerance = 1.0e-10;
constexpr int sigma_up_up = 0;
constexpr int sigma_up_down = 1;
constexpr int sigma_down_down = 2;

void initialize_functional(
    xc_func_type& functional,
    int identifier,
    int polarization) {

    if (xc_func_init(&functional, identifier, polarization) != 0) {
        throw std::runtime_error(
            "LibXC failed to initialize functional id "
            + std::to_string(identifier) + "."
        );
    }
}

void validate_real_field(
    const std::vector<double>& field,
    const FFTWorkspace& fft,
    const std::string& label) {

    if (static_cast<int>(field.size()) != fft.grid.ngrid) {
        throw std::runtime_error(label + " does not match the FFT grid.");
    }
    for (double value : field) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(label + " contains a non-finite value.");
        }
    }
}

Eigen::Vector3i derivative_frequency(
    const FFTGrid& grid,
    int i,
    int j,
    int k) {

    Eigen::Vector3i frequency = grid.freq_from_indices(i, j, k);
    const std::array<int, 3> indices{{i, j, k}};
    const std::array<int, 3> dimensions{{grid.n1, grid.n2, grid.n3}};
    for (int direction = 0; direction < 3; ++direction) {
        if (dimensions[direction] % 2 == 0 &&
            indices[direction] == dimensions[direction] / 2) {
            /*
             * The even-grid Nyquist mode is its own negative. Giving it a
             * nonzero odd derivative would break Hermitian symmetry. Zero
             * this fractional-direction derivative before converting with
             * B = 2 pi A^{-T}; this also handles skew cells correctly.
             */
            frequency[direction] = 0;
        }
    }
    return frequency;
}

void require_nearly_real(
    double maximum_real,
    double maximum_imaginary,
    const std::string& label) {

    const double tolerance = inverse_fft_imaginary_tolerance
        * std::max(1.0, maximum_real);
    if (!std::isfinite(maximum_real) ||
        !std::isfinite(maximum_imaginary) ||
        maximum_imaginary > tolerance) {
        throw std::runtime_error(
            label + " inverse FFT has a non-negligible imaginary part."
        );
    }
}

std::vector<double> sanitize_density(
    const std::vector<double>& density) {

    std::vector<double> sanitized(density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        const double value = density[p];
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "XC density contains a non-finite value."
            );
        }
        if (value < -negative_density_tolerance) {
            throw std::runtime_error(
                "XC density contains a physically invalid negative value."
            );
        }
        sanitized[p] = std::max(value, 0.0);
    }
    return sanitized;
}

void evaluate_one_lda_functional(
    const xc_func_type& functional,
    const std::vector<double>& density,
    double dV,
    std::vector<double>& potential,
    double& energy) {

    if (density.empty()) {
        energy = 0.0;
        return;
    }
    std::vector<double> energy_per_particle(density.size(), 0.0);
    std::vector<double> local_potential(density.size(), 0.0);
    xc_lda_exc_vxc(
        &functional,
        static_cast<int>(density.size()),
        density.data(),
        energy_per_particle.data(),
        local_potential.data()
    );
    energy = 0.0;
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        energy += dV * density[p] * energy_per_particle[p];
        potential[p] += local_potential[p];
    }
}

void evaluate_one_gga_functional(
    const xc_func_type& functional,
    const std::vector<double>& density,
    const std::vector<double>& sigma,
    double dV,
    std::vector<double>& vrho,
    std::vector<double>& vsigma,
    double& energy) {

    if (density.empty()) {
        energy = 0.0;
        return;
    }
    std::vector<double> energy_per_particle(density.size(), 0.0);
    std::vector<double> local_vrho(density.size(), 0.0);
    std::vector<double> local_vsigma(density.size(), 0.0);
    xc_gga_exc_vxc(
        &functional,
        static_cast<int>(density.size()),
        density.data(),
        sigma.data(),
        energy_per_particle.data(),
        local_vrho.data(),
        local_vsigma.data()
    );
    energy = 0.0;
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        energy += dV * density[p] * energy_per_particle[p];
        vrho[p] += local_vrho[p];
        vsigma[p] += local_vsigma[p];
    }
}

void evaluate_one_spin_lda_functional(
    const xc_func_type& functional,
    const std::vector<double>& interleaved_density,
    const std::vector<double>& total_density,
    double dV,
    std::vector<double>& potential_up,
    std::vector<double>& potential_down,
    double& energy) {

    const int point_count = static_cast<int>(total_density.size());
    if (point_count == 0) {
        energy = 0.0;
        return;
    }
    std::vector<double> energy_per_particle(point_count, 0.0);
    std::vector<double> local_potential(2 * point_count, 0.0);
    xc_lda_exc_vxc(
        &functional,
        point_count,
        interleaved_density.data(),
        energy_per_particle.data(),
        local_potential.data()
    );
    energy = 0.0;
    for (int p = 0; p < point_count; ++p) {
        energy += dV * total_density[p] * energy_per_particle[p];
        potential_up[p] += local_potential[2 * p];
        potential_down[p] += local_potential[2 * p + 1];
    }
}

void evaluate_one_spin_gga_functional(
    const xc_func_type& functional,
    const std::vector<double>& interleaved_density,
    const std::vector<double>& sigma,
    const std::vector<double>& total_density,
    double dV,
    std::vector<double>& potential_up,
    std::vector<double>& potential_down,
    std::array<std::vector<double>, 3>& vsigma,
    double& energy) {

    const int point_count = static_cast<int>(total_density.size());
    if (point_count == 0) {
        energy = 0.0;
        return;
    }
    std::vector<double> energy_per_particle(point_count, 0.0);
    std::vector<double> local_vrho(2 * point_count, 0.0);
    std::vector<double> local_vsigma(3 * point_count, 0.0);
    xc_gga_exc_vxc(
        &functional,
        point_count,
        interleaved_density.data(),
        sigma.data(),
        energy_per_particle.data(),
        local_vrho.data(),
        local_vsigma.data()
    );
    energy = 0.0;
    for (int p = 0; p < point_count; ++p) {
        energy += dV * total_density[p] * energy_per_particle[p];
        potential_up[p] += local_vrho[2 * p];
        potential_down[p] += local_vrho[2 * p + 1];
        for (int component = 0; component < 3; ++component) {
            vsigma[component][p] +=
                local_vsigma[3 * p + component];
        }
    }
}

} // namespace

std::array<std::vector<double>, 3> spectral_gradient(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& field) {

    validate_real_field(field, fft, "Spectral-gradient field");
    const int point_count = fft.grid.ngrid;
    for (int p = 0; p < point_count; ++p) {
        fft.real_grid[p] = std::complex<double>(field[p], 0.0);
    }
    fftw_execute(fft.forward_plan);

    const FFTWorkspace::BatchPlans& plans = fft.ensure_batch_plans(3);
    const std::complex<double> imaginary_unit(0.0, 1.0);
    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                const int p = fft.grid.index(i, j, k);
                const Eigen::Vector3d G = lattice.gvector_from_freq(
                    derivative_frequency(fft.grid, i, j, k)
                );
                const std::complex<double> coefficient =
                    fft.forward_raw[p]
                    / static_cast<double>(point_count);
                for (int direction = 0; direction < 3; ++direction) {
                    fft.batch_reciprocal_grid[
                        direction * point_count + p
                    ] = imaginary_unit * G[direction] * coefficient;
                }
            }
        }
    }
    fftw_execute(plans.backward);

    std::array<std::vector<double>, 3> gradient;
    double maximum_real = 0.0;
    double maximum_imaginary = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
        gradient[direction].resize(point_count);
        for (int p = 0; p < point_count; ++p) {
            const std::complex<double> value =
                fft.batch_real_grid[direction * point_count + p];
            gradient[direction][p] = value.real();
            maximum_real = std::max(maximum_real, std::abs(value.real()));
            maximum_imaginary = std::max(
                maximum_imaginary, std::abs(value.imag())
            );
        }
    }
    require_nearly_real(
        maximum_real, maximum_imaginary, "Spectral gradient"
    );
    return gradient;
}

std::vector<double> spectral_divergence(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::array<std::vector<double>, 3>& vector_field) {

    for (int direction = 0; direction < 3; ++direction) {
        validate_real_field(
            vector_field[direction], fft, "Spectral-divergence field"
        );
    }
    const int point_count = fft.grid.ngrid;
    const FFTWorkspace::BatchPlans& plans = fft.ensure_batch_plans(3);
    for (int direction = 0; direction < 3; ++direction) {
        for (int p = 0; p < point_count; ++p) {
            fft.batch_real_grid[direction * point_count + p] =
                std::complex<double>(vector_field[direction][p], 0.0);
        }
    }
    fftw_execute(plans.forward);

    const std::complex<double> imaginary_unit(0.0, 1.0);
    for (int i = 0; i < fft.grid.n1; ++i) {
        for (int j = 0; j < fft.grid.n2; ++j) {
            for (int k = 0; k < fft.grid.n3; ++k) {
                const int p = fft.grid.index(i, j, k);
                const Eigen::Vector3d G = lattice.gvector_from_freq(
                    derivative_frequency(fft.grid, i, j, k)
                );
                std::complex<double> coefficient(0.0, 0.0);
                for (int direction = 0; direction < 3; ++direction) {
                    coefficient += imaginary_unit * G[direction]
                        * fft.batch_reciprocal_grid[
                            direction * point_count + p
                        ] / static_cast<double>(point_count);
                }
                fft.reciprocal_grid[p] = coefficient;
            }
        }
    }
    fftw_execute(fft.backward_plan);

    std::vector<double> divergence(point_count, 0.0);
    double maximum_real = 0.0;
    double maximum_imaginary = 0.0;
    for (int p = 0; p < point_count; ++p) {
        divergence[p] = fft.real_grid[p].real();
        maximum_real = std::max(
            maximum_real, std::abs(fft.real_grid[p].real())
        );
        maximum_imaginary = std::max(
            maximum_imaginary, std::abs(fft.real_grid[p].imag())
        );
    }
    require_nearly_real(
        maximum_real, maximum_imaginary, "Spectral divergence"
    );
    return divergence;
}

struct LibXCFunctional::Impl {
    xc_func_type exchange{};
    xc_func_type correlation{};
    bool exchange_initialized = false;
    bool correlation_initialized = false;
    bool is_gga = false;
    int nspin = 1;

    ~Impl() {
        if (correlation_initialized) {
            xc_func_end(&correlation);
        }
        if (exchange_initialized) {
            xc_func_end(&exchange);
        }
    }
};

LibXCFunctional::LibXCFunctional(
    XCFunctional functional,
    int nspin)
    : impl_(std::make_unique<Impl>()) {

    if (nspin != 1 && nspin != 2) {
        throw std::runtime_error("LibXC supports nspin = 1 or 2.");
    }
    impl_->nspin = nspin;
    const int polarization =
        nspin == 1 ? XC_UNPOLARIZED : XC_POLARIZED;

    if (functional == XCFunctional::PerdewBurkeErnzerhof) {
        impl_->is_gga = true;
        initialize_functional(
            impl_->exchange, XC_GGA_X_PBE, polarization
        );
        impl_->exchange_initialized = true;
        initialize_functional(
            impl_->correlation, XC_GGA_C_PBE, polarization
        );
        impl_->correlation_initialized = true;
        return;
    }

    initialize_functional(impl_->exchange, XC_LDA_X, polarization);
    impl_->exchange_initialized = true;
    if (functional == XCFunctional::PerdewZunger) {
        initialize_functional(
            impl_->correlation, XC_LDA_C_PZ, polarization
        );
        impl_->correlation_initialized = true;
    } else if (functional != XCFunctional::ExchangeOnly) {
        throw std::runtime_error("Unknown XC functional selection.");
    }
}

LibXCFunctional::~LibXCFunctional() = default;

LibXCFunctional::LibXCFunctional(
    LibXCFunctional&&) noexcept = default;

LibXCFunctional& LibXCFunctional::operator=(
    LibXCFunctional&&) noexcept = default;

XCResult LibXCFunctional::evaluate(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& density,
    double dV) const {

    if (impl_->nspin != 1) {
        throw std::runtime_error(
            "Use evaluate_spin() for a spin-polarized LibXC functional."
        );
    }
    if (!std::isfinite(dV) || dV <= 0.0) {
        throw std::runtime_error(
            "XC grid-cell volume must be positive and finite."
        );
    }
    validate_real_field(density, fft, "XC density");
    const std::vector<double> sanitized_density =
        sanitize_density(density);

    XCResult result;
    result.Vxc.assign(density.size(), 0.0);
    if (!impl_->is_gga) {
        evaluate_one_lda_functional(
            impl_->exchange,
            sanitized_density,
            dV,
            result.Vxc,
            result.exchange_energy
        );
        if (impl_->correlation_initialized) {
            evaluate_one_lda_functional(
                impl_->correlation,
                sanitized_density,
                dV,
                result.Vxc,
                result.correlation_energy
            );
        }
        return result;
    }

    const std::array<std::vector<double>, 3> gradient =
        spectral_gradient(lattice, fft, sanitized_density);
    std::vector<double> sigma(density.size(), 0.0);
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        sigma[p] = gradient[0][p] * gradient[0][p]
            + gradient[1][p] * gradient[1][p]
            + gradient[2][p] * gradient[2][p];
    }

    std::vector<double> vsigma(density.size(), 0.0);
    evaluate_one_gga_functional(
        impl_->exchange,
        sanitized_density,
        sigma,
        dV,
        result.Vxc,
        vsigma,
        result.exchange_energy
    );
    evaluate_one_gga_functional(
        impl_->correlation,
        sanitized_density,
        sigma,
        dV,
        result.Vxc,
        vsigma,
        result.correlation_energy
    );

    std::array<std::vector<double>, 3> flux;
    for (int direction = 0; direction < 3; ++direction) {
        flux[direction].resize(density.size());
        for (int p = 0; p < static_cast<int>(density.size()); ++p) {
            flux[direction][p] =
                2.0 * vsigma[p] * gradient[direction][p];
        }
    }
    const std::vector<double> divergence =
        spectral_divergence(lattice, fft, flux);
    for (int p = 0; p < static_cast<int>(density.size()); ++p) {
        result.Vxc[p] -= divergence[p];
        if (!std::isfinite(result.Vxc[p])) {
            throw std::runtime_error(
                "PBE XC potential contains a non-finite value."
            );
        }
    }
    if (!std::isfinite(result.exchange_energy) ||
        !std::isfinite(result.correlation_energy)) {
        throw std::runtime_error(
            "PBE XC energy contains a non-finite value."
        );
    }
    return result;
}

SpinXCResult LibXCFunctional::evaluate_spin(
    const Lattice& lattice,
    FFTWorkspace& fft,
    const std::vector<double>& density_up,
    const std::vector<double>& density_down,
    double dV) const {

    if (impl_->nspin != 2) {
        throw std::runtime_error(
            "Use evaluate() for an unpolarized LibXC functional."
        );
    }
    if (!std::isfinite(dV) || dV <= 0.0) {
        throw std::runtime_error(
            "XC grid-cell volume must be positive and finite."
        );
    }
    if (density_up.size() != density_down.size()) {
        throw std::runtime_error(
            "Spin-density grids must have the same size."
        );
    }
    validate_real_field(density_up, fft, "Spin-up XC density");
    validate_real_field(density_down, fft, "Spin-down XC density");

    const std::vector<double> sanitized_up =
        sanitize_density(density_up);
    const std::vector<double> sanitized_down =
        sanitize_density(density_down);
    const int point_count = static_cast<int>(density_up.size());
    /* LibXC uses a point-major polarized layout: up, down. */
    std::vector<double> interleaved_density(2 * point_count, 0.0);
    std::vector<double> total_density(point_count, 0.0);
    for (int p = 0; p < point_count; ++p) {
        interleaved_density[2 * p] = sanitized_up[p];
        interleaved_density[2 * p + 1] = sanitized_down[p];
        total_density[p] = sanitized_up[p] + sanitized_down[p];
    }

    SpinXCResult result;
    result.Vxc_up.assign(point_count, 0.0);
    result.Vxc_down.assign(point_count, 0.0);
    if (impl_->is_gga) {
        const std::array<std::vector<double>, 3> gradient_up =
            spectral_gradient(lattice, fft, sanitized_up);
        const std::array<std::vector<double>, 3> gradient_down =
            spectral_gradient(lattice, fft, sanitized_down);
        /*
         * The three point-major LibXC GGA channels are
         * (up-up, up-down, down-down).
         */
        std::vector<double> sigma(3 * point_count, 0.0);
        for (int p = 0; p < point_count; ++p) {
            for (int direction = 0; direction < 3; ++direction) {
                sigma[3 * p + sigma_up_up] +=
                    gradient_up[direction][p]
                    * gradient_up[direction][p];
                sigma[3 * p + sigma_up_down] +=
                    gradient_up[direction][p]
                    * gradient_down[direction][p];
                sigma[3 * p + sigma_down_down] +=
                    gradient_down[direction][p]
                    * gradient_down[direction][p];
            }
        }

        std::array<std::vector<double>, 3> vsigma;
        for (std::vector<double>& component : vsigma) {
            component.assign(point_count, 0.0);
        }
        evaluate_one_spin_gga_functional(
            impl_->exchange,
            interleaved_density,
            sigma,
            total_density,
            dV,
            result.Vxc_up,
            result.Vxc_down,
            vsigma,
            result.exchange_energy
        );
        evaluate_one_spin_gga_functional(
            impl_->correlation,
            interleaved_density,
            sigma,
            total_density,
            dV,
            result.Vxc_up,
            result.Vxc_down,
            vsigma,
            result.correlation_energy
        );

        std::array<std::vector<double>, 3> flux_up;
        std::array<std::vector<double>, 3> flux_down;
        for (int direction = 0; direction < 3; ++direction) {
            flux_up[direction].resize(point_count);
            flux_down[direction].resize(point_count);
            for (int p = 0; p < point_count; ++p) {
                flux_up[direction][p] =
                    2.0 * vsigma[sigma_up_up][p]
                        * gradient_up[direction][p]
                    + vsigma[sigma_up_down][p]
                        * gradient_down[direction][p];
                flux_down[direction][p] =
                    vsigma[sigma_up_down][p]
                        * gradient_up[direction][p]
                    + 2.0 * vsigma[sigma_down_down][p]
                        * gradient_down[direction][p];
            }
        }
        const std::vector<double> divergence_up =
            spectral_divergence(lattice, fft, flux_up);
        const std::vector<double> divergence_down =
            spectral_divergence(lattice, fft, flux_down);
        for (int p = 0; p < point_count; ++p) {
            result.Vxc_up[p] -= divergence_up[p];
            result.Vxc_down[p] -= divergence_down[p];
            if (!std::isfinite(result.Vxc_up[p]) ||
                !std::isfinite(result.Vxc_down[p])) {
                throw std::runtime_error(
                    "Spin-polarized PBE XC potential contains a "
                    "non-finite value."
                );
            }
        }
        if (!std::isfinite(result.exchange_energy) ||
            !std::isfinite(result.correlation_energy)) {
            throw std::runtime_error(
                "Spin-polarized PBE XC energy contains a non-finite value."
            );
        }
        return result;
    }

    evaluate_one_spin_lda_functional(
        impl_->exchange,
        interleaved_density,
        total_density,
        dV,
        result.Vxc_up,
        result.Vxc_down,
        result.exchange_energy
    );
    if (impl_->correlation_initialized) {
        evaluate_one_spin_lda_functional(
            impl_->correlation,
            interleaved_density,
            total_density,
            dV,
            result.Vxc_up,
            result.Vxc_down,
            result.correlation_energy
        );
    }
    return result;
}

std::string xc_functional_name(XCFunctional functional) {
    if (functional == XCFunctional::ExchangeOnly) {
        return "LDA_X";
    }
    if (functional == XCFunctional::PerdewZunger) {
        return "LDA_X + LDA_C_PZ";
    }
    if (functional == XCFunctional::PerdewBurkeErnzerhof) {
        return "GGA_X_PBE + GGA_C_PBE";
    }
    throw std::runtime_error("Unknown XC functional selection.");
}

bool pseudopotential_functional_matches_xc(
    const std::string& label,
    XCFunctional functional) {

    std::string uppercase = label;
    std::transform(
        uppercase.begin(), uppercase.end(), uppercase.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        }
    );
    if (functional == XCFunctional::PerdewBurkeErnzerhof) {
        const bool named_pbe =
            uppercase.find("PBE") != std::string::npos &&
            uppercase.find("PBESOL") == std::string::npos;
        return named_pbe ||
            (uppercase.find("PBX") != std::string::npos &&
             uppercase.find("PBC") != std::string::npos);
    }
    if (functional == XCFunctional::PerdewZunger ||
        functional == XCFunctional::ExchangeOnly) {
        return uppercase.find("PZ") != std::string::npos;
    }
    return false;
}

std::string libxc_runtime_version() {
    return xc_version_string();
}
