#include "pdos.hpp"

#include "radial_transform.hpp"
#include "upf_nonlocal.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

constexpr long long openmp_minimum_work = 32768;
constexpr double inverse_sqrt_two_pi =
    0.398942280401432677939946059934381868;

struct ReciprocalRadiusClasses {
    std::vector<double> radii;
    std::vector<int> basis_to_radius;
};

std::complex<double> minus_i_to_power(int l) {
    switch (l % 4) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, -1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, 1.0};
    }
}

std::complex<double> translation_phase(
    const Eigen::Vector3d& q,
    const Eigen::Vector3d& position) {

    const double phase = -q.dot(position);
    return {std::cos(phase), std::sin(phase)};
}

ReciprocalRadiusClasses make_radius_classes(
    const PlaneWaveBasis3D& basis) {

    ReciprocalRadiusClasses classes;
    classes.basis_to_radius.resize(basis.size(), 0);
    std::map<double, int> exact_classes;
    for (int ig = 0; ig < basis.size(); ++ig) {
        const double q2 =
            basis.gvectors[ig].q_cart.squaredNorm();
        const auto inserted = exact_classes.emplace(
            q2, static_cast<int>(classes.radii.size())
        );
        if (inserted.second) {
            classes.radii.push_back(std::sqrt(q2));
        }
        classes.basis_to_radius[ig] = inserted.first->second;
    }
    return classes;
}

void validate_projection_species(
    const AtomicProjectionSpecies& species) {

    if (species.radial_grid_bohr.empty() ||
        species.radial_grid_bohr.size() !=
            species.quadrature_weights.size()) {
        throw std::runtime_error(
            "Atomic-projection radial arrays have inconsistent sizes."
        );
    }
    if (species.wavefunctions.empty()) {
        throw std::runtime_error(
            "UPF for element " + species.element
            + " has no PP_PSWFC pseudo-atomic wavefunctions."
        );
    }
    for (const UPFAtomicWavefunction& wavefunction :
         species.wavefunctions) {
        if (wavefunction.angular_momentum < 0 ||
            wavefunction.r_times_radial_wavefunction.size() !=
                species.radial_grid_bohr.size()) {
            throw std::runtime_error(
                "Atomic-projection wavefunction metadata or radial size "
                "is invalid for element " + species.element + "."
            );
        }
    }
}

double gaussian_density(double difference, double sigma) {
    const double reduced = difference / sigma;
    return inverse_sqrt_two_pi / sigma
        * std::exp(-0.5 * reduced * reduced);
}

double gaussian_cumulative(double difference, double sigma) {
    return 0.5 * std::erfc(
        -difference / (std::sqrt(2.0) * sigma)
    );
}

int electronic_state_index(
    int spin,
    int kpoint,
    int kpoint_count) {

    return spin * kpoint_count + kpoint;
}

} // namespace

AtomicProjectionSpecies prepare_atomic_projection_species(
    const UPFData& upf) {

    if (upf.header.number_of_wavefunctions !=
        static_cast<int>(upf.atomic_wavefunctions.size())) {
        throw std::runtime_error(
            "UPF header and PP_PSWFC wavefunction counts do not match."
        );
    }
    AtomicProjectionSpecies species;
    species.element = upf.header.element;
    species.radial_grid_bohr = upf.mesh.r;
    species.quadrature_weights =
        make_upf_simpson_weights(upf.mesh.rab);
    species.wavefunctions = upf.atomic_wavefunctions;
    validate_projection_species(species);
    return species;
}

std::vector<AtomicProjectionOrbital> make_atomic_projection_orbitals(
    const std::vector<AtomicProjectionSpecies>& species,
    const std::vector<UPFLocalIon>& ions) {

    std::vector<AtomicProjectionOrbital> orbitals;
    for (int iatom = 0;
         iatom < static_cast<int>(ions.size());
         ++iatom) {
        const UPFLocalIon& ion = ions[iatom];
        if (ion.species_index < 0 ||
            ion.species_index >= static_cast<int>(species.size())) {
            throw std::runtime_error(
                "Atomic-projection ion has an invalid species index."
            );
        }
        const AtomicProjectionSpecies& one_species =
            species[ion.species_index];
        validate_projection_species(one_species);
        for (const UPFAtomicWavefunction& wavefunction :
             one_species.wavefunctions) {
            for (int harmonic = 0;
                 harmonic < 2 * wavefunction.angular_momentum + 1;
                 ++harmonic) {
                orbitals.push_back({
                    iatom,
                    one_species.element,
                    wavefunction.index,
                    wavefunction.label,
                    wavefunction.angular_momentum,
                    harmonic
                });
            }
        }
    }
    return orbitals;
}

AtomicProjectionBasis build_lowdin_atomic_projection_basis(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<AtomicProjectionSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    double relative_eigenvalue_cutoff,
    int thread_count) {

    if (basis.size() <= 0) {
        throw std::runtime_error(
            "Cannot build an atomic projection in an empty plane-wave basis."
        );
    }
    if (!std::isfinite(relative_eigenvalue_cutoff) ||
        relative_eigenvalue_cutoff <= 0.0 ||
        relative_eigenvalue_cutoff >= 1.0) {
        throw std::runtime_error(
            "The Löwdin relative eigenvalue cutoff must be in (0, 1)."
        );
    }
    if (thread_count <= 0) {
        throw std::runtime_error(
            "Atomic-projection thread count must be positive."
        );
    }
    for (const AtomicProjectionSpecies& one_species : species) {
        validate_projection_species(one_species);
    }
    for (const UPFLocalIon& ion : ions) {
        if (ion.species_index < 0 ||
            ion.species_index >= static_cast<int>(species.size())) {
            throw std::runtime_error(
                "Atomic-projection ion has an invalid species index."
            );
        }
    }

    AtomicProjectionBasis result;
    result.orbitals =
        make_atomic_projection_orbitals(species, ions);
    const int orbital_count =
        static_cast<int>(result.orbitals.size());
    if (orbital_count == 0) {
        throw std::runtime_error(
            "No PP_PSWFC atomic projection orbitals were constructed."
        );
    }
    if (orbital_count > basis.size()) {
        throw std::runtime_error(
            "The atomic projection has more orbitals than the plane-wave "
            "basis. Increase ecut_ha."
        );
    }

    const ReciprocalRadiusClasses radius_classes =
        make_radius_classes(basis);
    std::vector<std::vector<std::vector<double>>> radial_transforms(
        species.size()
    );
    for (int ispecies = 0;
         ispecies < static_cast<int>(species.size());
         ++ispecies) {
        const AtomicProjectionSpecies& one_species =
            species[ispecies];
        radial_transforms[ispecies].resize(
            one_species.wavefunctions.size()
        );
        for (int iwfc = 0;
             iwfc < static_cast<int>(
                 one_species.wavefunctions.size()
             );
             ++iwfc) {
            const UPFAtomicWavefunction& wavefunction =
                one_species.wavefunctions[iwfc];
            std::vector<double>& transform =
                radial_transforms[ispecies][iwfc];
            transform.resize(radius_classes.radii.size());
            const std::int64_t work =
                static_cast<std::int64_t>(
                    radius_classes.radii.size()
                );
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && work >= openmp_minimum_work) \
    num_threads(thread_count)
            for (std::int64_t iradius = 0;
                 iradius < work;
                 ++iradius) {
                transform[iradius] =
                    radial_fourier_bessel_transform_from_r_times_function_unchecked(
                        wavefunction.angular_momentum,
                        radius_classes.radii[iradius],
                        one_species.radial_grid_bohr,
                        one_species.quadrature_weights,
                        wavefunction.r_times_radial_wavefunction
                    );
            }
        }
    }

    Eigen::MatrixXcd raw =
        Eigen::MatrixXcd::Zero(basis.size(), orbital_count);
    const double inverse_sqrt_volume =
        1.0 / std::sqrt(lattice.volume());
    int column = 0;
    for (int iatom = 0;
         iatom < static_cast<int>(ions.size());
         ++iatom) {
        const UPFLocalIon& ion = ions[iatom];
        const AtomicProjectionSpecies& one_species =
            species[ion.species_index];
        const Eigen::Vector3d position =
            lattice.cart_from_frac(ion.frac_position);
        for (int iwfc = 0;
             iwfc < static_cast<int>(
                 one_species.wavefunctions.size()
             );
             ++iwfc) {
            const UPFAtomicWavefunction& wavefunction =
                one_species.wavefunctions[iwfc];
            const int harmonic_count =
                2 * wavefunction.angular_momentum + 1;
            const std::complex<double> prefactor =
                inverse_sqrt_volume
                * minus_i_to_power(
                    wavefunction.angular_momentum
                );
            for (int ig = 0; ig < basis.size(); ++ig) {
                const Eigen::Vector3d& q =
                    basis.gvectors[ig].q_cart;
                const std::vector<double> harmonics =
                    qe_real_spherical_harmonics(
                        wavefunction.angular_momentum, q
                    );
                const std::complex<double> phase =
                    translation_phase(q, position);
                const double radial =
                    radial_transforms[ion.species_index][iwfc]
                        [radius_classes.basis_to_radius[ig]];
                for (int harmonic = 0;
                     harmonic < harmonic_count;
                     ++harmonic) {
                    raw(ig, column + harmonic) =
                        prefactor * phase * radial
                        * harmonics[harmonic];
                }
            }
            column += harmonic_count;
        }
    }

    const Eigen::MatrixXcd overlap =
        raw.adjoint() * raw;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(
        overlap
    );
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Failed to diagonalize the atomic-orbital overlap matrix."
        );
    }
    result.minimum_overlap_eigenvalue =
        eigensolver.eigenvalues().minCoeff();
    result.maximum_overlap_eigenvalue =
        eigensolver.eigenvalues().maxCoeff();
    if (!std::isfinite(result.maximum_overlap_eigenvalue) ||
        result.maximum_overlap_eigenvalue <= 0.0 ||
        !std::isfinite(result.minimum_overlap_eigenvalue) ||
        result.minimum_overlap_eigenvalue <=
            relative_eigenvalue_cutoff
                * result.maximum_overlap_eigenvalue) {
        throw std::runtime_error(
            "The PP_PSWFC overlap matrix is singular or ill-conditioned "
            "in the current plane-wave basis. Increase ecut_ha or reduce "
            "pdos_lowdin_cutoff only after inspecting the projection."
        );
    }

    Eigen::VectorXd inverse_square_roots =
        eigensolver.eigenvalues();
    for (int index = 0;
         index < inverse_square_roots.size();
         ++index) {
        inverse_square_roots[index] =
            1.0 / std::sqrt(inverse_square_roots[index]);
    }
    const Eigen::MatrixXcd inverse_square_root =
        eigensolver.eigenvectors()
        * inverse_square_roots.asDiagonal()
        * eigensolver.eigenvectors().adjoint();
    result.coefficients = raw * inverse_square_root;
    const Eigen::MatrixXcd identity_error =
        result.coefficients.adjoint()
        * result.coefficients
        - Eigen::MatrixXcd::Identity(
            orbital_count, orbital_count
        );
    result.orthonormality_error =
        identity_error.cwiseAbs().maxCoeff();
    if (!std::isfinite(result.orthonormality_error) ||
        result.orthonormality_error > 1.0e-8) {
        throw std::runtime_error(
            "Löwdin atomic orbitals failed the orthonormality check."
        );
    }
    return result;
}

Eigen::MatrixXd compute_lowdin_projection_weights(
    const AtomicProjectionBasis& projection_basis,
    const Eigen::MatrixXcd& eigenvectors) {

    if (projection_basis.coefficients.rows() !=
            eigenvectors.rows() ||
        projection_basis.coefficients.cols() == 0 ||
        eigenvectors.cols() == 0) {
        throw std::runtime_error(
            "Atomic-projection and eigenvector dimensions do not match."
        );
    }
    const Eigen::MatrixXcd amplitudes =
        projection_basis.coefficients.adjoint()
        * eigenvectors;
    Eigen::MatrixXd weights =
        amplitudes.cwiseAbs2();
    for (int band = 0; band < weights.cols(); ++band) {
        const double total = weights.col(band).sum();
        if (!std::isfinite(total) || total < -1.0e-12 ||
            total > 1.0 + 1.0e-8) {
            throw std::runtime_error(
                "A Löwdin atomic projection weight is outside [0, 1]."
            );
        }
    }
    return weights;
}

ProjectedDensityOfStatesResult compute_projected_density_of_states(
    const KPointSCFResult& electronic,
    const SCFOptions& scf_options,
    const DensityOfStatesResult& total_dos,
    const AtomicProjectionResult& projection) {

    if (!electronic.converged || total_dos.samples.size() < 2 ||
        projection.orbitals.empty() ||
        projection.states.size() != electronic.kpoints.size()) {
        throw std::runtime_error(
            "PDOS requires converged NSCF eigenvalues and complete "
            "atomic-projection weights."
        );
    }
    const int nspin = scf_options.nspin;
    const int kpoint_count =
        static_cast<int>(electronic.kpoints.size()) / nspin;
    const int orbital_count =
        static_cast<int>(projection.orbitals.size());

    ProjectedDensityOfStatesResult result;
    result.nspin = nspin;
    result.nbands = scf_options.nbands;
    result.kpoint_count = kpoint_count;
    result.fermi_energy_ha = total_dos.fermi_energy_ha;
    result.gaussian_sigma_ha = total_dos.gaussian_sigma_ha;
    result.projection = projection;
    result.energies_ha.reserve(total_dos.samples.size());
    for (const DensityOfStatesSample& sample :
         total_dos.samples) {
        result.energies_ha.push_back(sample.energy_ha);
    }
    result.channels.resize(orbital_count);
    for (int orbital = 0; orbital < orbital_count; ++orbital) {
        ProjectedDensityOfStatesChannel& channel =
            result.channels[orbital];
        channel.orbital = projection.orbitals[orbital];
        channel.spin_per_ha.assign(
            nspin,
            std::vector<double>(result.energies_ha.size(), 0.0)
        );
        channel.integrated_spin.assign(
            nspin,
            std::vector<double>(result.energies_ha.size(), 0.0)
        );
    }

    double represented_occupied_electrons = 0.0;
    double total_occupied_electrons = 0.0;
    const double spin_degeneracy =
        nspin == 1 ? 2.0 : 1.0;
    for (int spin = 0; spin < nspin; ++spin) {
        for (int kpoint = 0;
             kpoint < kpoint_count;
             ++kpoint) {
            const int state_index =
                electronic_state_index(
                    spin, kpoint, kpoint_count
                );
            const KPointElectronicState& state =
                electronic.kpoints[state_index];
            const AtomicProjectionState& projected =
                projection.states[state_index];
            if (projected.spin_channel != spin ||
                projected.kpoint_index != kpoint ||
                projected.weights.rows() != orbital_count ||
                projected.weights.cols() != result.nbands) {
                throw std::runtime_error(
                    "PDOS projection-state metadata or dimensions do "
                    "not match the NSCF states."
                );
            }
            for (int band = 0; band < result.nbands; ++band) {
                const double state_weight =
                    spin_degeneracy * state.weight;
                const double eigenvalue =
                    state.eigenvalues[band];
                const double occupation =
                    state.occupations.empty()
                    ? 0.0
                    : state.occupations[band];
                const double projection_sum =
                    projected.weights.col(band).sum();
                total_occupied_electrons +=
                    state.weight * occupation;
                represented_occupied_electrons +=
                    state.weight * occupation
                    * projection_sum;
                for (int orbital = 0;
                     orbital < orbital_count;
                     ++orbital) {
                    const double projected_weight =
                        state_weight
                        * projected.weights(orbital, band);
                    if (projected_weight == 0.0) {
                        continue;
                    }
                    ProjectedDensityOfStatesChannel& channel =
                        result.channels[orbital];
                    for (int sample = 0;
                         sample < static_cast<int>(
                             result.energies_ha.size()
                         );
                         ++sample) {
                        const double difference =
                            result.energies_ha[sample]
                            - eigenvalue;
                        channel.spin_per_ha[spin][sample] +=
                            projected_weight
                            * gaussian_density(
                                difference,
                                result.gaussian_sigma_ha
                            );
                        channel.integrated_spin[spin][sample] +=
                            projected_weight
                            * gaussian_cumulative(
                                difference,
                                result.gaussian_sigma_ha
                            );
                    }
                }
            }
        }
    }
    if (total_occupied_electrons > 0.0) {
        result.occupied_spilling = std::clamp(
            1.0 - represented_occupied_electrons
                / total_occupied_electrons,
            0.0,
            1.0
        );
    }
    return result;
}

void write_projected_density_of_states(
    const std::string& path,
    const ProjectedDensityOfStatesResult& result) {

    if (path.empty() || result.channels.empty() ||
        result.energies_ha.size() < 2 ||
        (result.nspin != 1 && result.nspin != 2)) {
        throw std::runtime_error(
            "Cannot write an incomplete projected density of states."
        );
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create projected-density-of-states output: "
            + path
        );
    }
    output << "# PWDFT Löwdin projected density of states\n"
        << std::scientific << std::setprecision(16)
        << "# fermi_energy_ha = "
        << result.fermi_energy_ha << "\n"
        << "# gaussian_sigma_ha = "
        << result.gaussian_sigma_ha << "\n"
        << "# gaussian_sigma_ev = "
        << result.gaussian_sigma_ha * HARTREE_TO_EV << "\n"
        << "# nspin = " << result.nspin << "\n"
        << "# nbands = " << result.nbands << "\n"
        << "# irreducible_kpoints = "
        << result.kpoint_count << "\n"
        << "# atomic_orbitals = "
        << result.channels.size() << "\n"
        << "# occupied_spilling = "
        << result.occupied_spilling << "\n"
        << "# overlap_eigenvalue_min = "
        << result.projection.minimum_overlap_eigenvalue << "\n"
        << "# overlap_eigenvalue_max = "
        << result.projection.maximum_overlap_eigenvalue << "\n"
        << "# lowdin_orthonormality_error_max = "
        << result.projection.maximum_orthonormality_error << "\n"
        << "# projection_weight_excess_max = "
        << result.projection.maximum_projection_weight_excess
        << "\n";
    for (int orbital = 0;
         orbital < static_cast<int>(result.channels.size());
         ++orbital) {
        const AtomicProjectionOrbital& item =
            result.channels[orbital].orbital;
        output << "# orbital " << orbital
            << " atom=" << item.atom_index
            << " element=" << std::quoted(item.element)
            << " wfc=" << item.wavefunction_index
            << " label=" << std::quoted(item.label)
            << " l=" << item.angular_momentum
            << " real_harmonic=" << item.real_harmonic_index
            << "\n";
    }
    output
        << "# columns: energy_ha energy_ev energy_minus_fermi_ev "
           "orbital atom element wfc label l real_harmonic spin "
           "pdos_states_per_ev integrated_projected_states\n";

    for (int orbital = 0;
         orbital < static_cast<int>(result.channels.size());
         ++orbital) {
        const ProjectedDensityOfStatesChannel& channel =
            result.channels[orbital];
        for (int spin = 0; spin < result.nspin; ++spin) {
            for (int sample = 0;
                 sample < static_cast<int>(
                     result.energies_ha.size()
                 );
                 ++sample) {
                const double energy =
                    result.energies_ha[sample];
                output << std::scientific
                    << std::setprecision(12)
                    << std::setw(20) << energy << " "
                    << std::setw(20)
                    << energy * HARTREE_TO_EV << " "
                    << std::setw(20)
                    << (energy - result.fermi_energy_ha)
                        * HARTREE_TO_EV << " "
                    << std::setw(6) << orbital << " "
                    << std::setw(6)
                    << channel.orbital.atom_index << " "
                    << std::quoted(channel.orbital.element)
                    << " " << std::setw(5)
                    << channel.orbital.wavefunction_index
                    << " " << std::quoted(
                        channel.orbital.label
                    )
                    << " " << std::setw(3)
                    << channel.orbital.angular_momentum
                    << " " << std::setw(3)
                    << channel.orbital.real_harmonic_index
                    << " " << std::setw(3) << spin
                    << " " << std::setw(20)
                    << channel.spin_per_ha[spin][sample]
                        / HARTREE_TO_EV
                    << " " << std::setw(20)
                    << channel.integrated_spin[spin][sample]
                    << "\n";
            }
            output << "\n";
        }
    }
}
