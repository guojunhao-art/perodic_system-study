#include "upf_nonlocal.hpp"

#include "radial_transform.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

constexpr double zero_tolerance = 1.0e-14;
constexpr long long openmp_minimum_work = 32768;

struct SpeciesProjectorTemplate {
    Eigen::VectorXcd beta_G;
    double D = 0.0;
};

struct ReciprocalRadiusClasses {
    std::vector<double> radii;
    std::vector<int> basis_to_radius;
};

std::complex<double> translation_phase(
    const Eigen::Vector3d& G,
    const Eigen::Vector3d& R) {

    const double phase = -G.dot(R);
    return {std::cos(phase), std::sin(phase)};
}

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

void validate_species(const UPFNonlocalSpecies& species) {
    const int nprojectors =
        static_cast<int>(species.radial_projectors.size());
    if (species.radial_grid_bohr.size()
            != species.quadrature_weights.size()) {
        throw std::runtime_error(
            "UPF nonlocal radial grid and quadrature sizes do not match."
        );
    }
    if (species.radial_grid_bohr.empty()) {
        throw std::runtime_error("UPF nonlocal radial grid is empty.");
    }
    for (int ir = 0;
         ir < static_cast<int>(species.radial_grid_bohr.size());
         ++ir) {
        const double radius = species.radial_grid_bohr[ir];
        const double weight = species.quadrature_weights[ir];
        if (!std::isfinite(radius) || radius < 0.0 ||
            (ir > 0 && radius <= species.radial_grid_bohr[ir - 1])) {
            throw std::runtime_error(
                "UPF nonlocal radial grid must be finite and increasing."
            );
        }
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw std::runtime_error(
                "UPF nonlocal quadrature weights must be positive and finite."
            );
        }
    }
    if (species.dij_hartree.rows() != nprojectors ||
        species.dij_hartree.cols() != nprojectors) {
        throw std::runtime_error(
            "UPF nonlocal DIJ dimensions do not match the projector count."
        );
    }
    if (!species.dij_hartree.allFinite()) {
        throw std::runtime_error("UPF nonlocal DIJ contains non-finite values.");
    }
    if (nprojectors > 0) {
        const double dij_scale = std::max(
            1.0,
            species.dij_hartree.cwiseAbs().maxCoeff()
        );
        const double tolerance = 1.0e-12 * dij_scale;
        if ((species.dij_hartree - species.dij_hartree.transpose())
                .cwiseAbs().maxCoeff() > tolerance) {
            throw std::runtime_error("UPF nonlocal DIJ must be symmetric.");
        }
        for (int i = 0; i < nprojectors; ++i) {
            for (int j = i + 1; j < nprojectors; ++j) {
                const int li =
                    species.radial_projectors[i].angular_momentum;
                const int lj =
                    species.radial_projectors[j].angular_momentum;
                if (li != lj &&
                    std::abs(species.dij_hartree(i, j)) > tolerance) {
                    throw std::runtime_error(
                        "NC-UPF PP_DIJ couples projectors with different l; "
                        "this scalar-relativistic implementation cannot expand it."
                    );
                }
            }
        }
    }
    for (const UPFProjector& projector : species.radial_projectors) {
        if (projector.angular_momentum < 0) {
            throw std::runtime_error(
                "UPF nonlocal projector has negative angular momentum."
            );
        }
        if (projector.r_times_beta.size()
                != species.radial_grid_bohr.size()) {
            throw std::runtime_error(
                "UPF nonlocal projector and radial-grid sizes do not match."
            );
        }
        if (!std::all_of(
                projector.r_times_beta.begin(),
                projector.r_times_beta.end(),
                [](double value) { return std::isfinite(value); }
            )) {
            throw std::runtime_error(
                "UPF nonlocal projector contains non-finite values."
            );
        }
    }
}

std::map<int, std::vector<int>> group_projectors_by_l(
    const UPFNonlocalSpecies& species) {

    std::map<int, std::vector<int>> groups;
    for (int ip = 0;
         ip < static_cast<int>(species.radial_projectors.size());
         ++ip) {
        groups[species.radial_projectors[ip].angular_momentum].push_back(ip);
    }
    return groups;
}

ReciprocalRadiusClasses make_radius_classes(
    const PlaneWaveBasis3D& basis) {

    ReciprocalRadiusClasses classes;
    classes.basis_to_radius.resize(basis.size(), 0);
    std::map<double, int> exact_classes;
    for (int ig = 0; ig < basis.size(); ++ig) {
        const double q2 = basis.gvectors[ig].q_cart.squaredNorm();
        const auto inserted = exact_classes.emplace(
            q2,
            static_cast<int>(classes.radii.size())
        );
        if (inserted.second) {
            classes.radii.push_back(std::sqrt(q2));
        }
        classes.basis_to_radius[ig] = inserted.first->second;
    }
    return classes;
}

std::vector<SpeciesProjectorTemplate> build_species_templates(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const ReciprocalRadiusClasses& radius_classes,
    const UPFNonlocalSpecies& species,
    int thread_count) {

    const int nbasis = basis.size();
    const int nprojectors =
        static_cast<int>(species.radial_projectors.size());
    if (nprojectors == 0) {
        return {};
    }

    /*
     * The radial transform and angular factor depend on species and q, but
     * not on the ion position.  Build them once and apply exp(-i q.R_I)
     * only after the DIJ-diagonalized templates are complete.
     */
    const int nradii = static_cast<int>(radius_classes.radii.size());
    Eigen::MatrixXd radial_transforms(nradii, nprojectors);
    const std::int64_t radial_work =
        static_cast<std::int64_t>(nradii)
        * static_cast<std::int64_t>(nprojectors);
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && radial_work >= openmp_minimum_work) \
    num_threads(thread_count)
    for (std::int64_t index = 0; index < radial_work; ++index) {
        const int ip = static_cast<int>(
            index / static_cast<std::int64_t>(nradii)
        );
        const int iradius = static_cast<int>(
            index % static_cast<std::int64_t>(nradii)
        );
        const UPFProjector& projector =
            species.radial_projectors[ip];
        radial_transforms(iradius, ip) =
            radial_fourier_bessel_transform_from_r_times_function_unchecked(
                projector.angular_momentum,
                radius_classes.radii[iradius],
                species.radial_grid_bohr,
                species.quadrature_weights,
                projector.r_times_beta
            );
    }

    std::vector<SpeciesProjectorTemplate> templates;
    const double inverse_sqrt_volume =
        1.0 / std::sqrt(lattice.volume());
    const auto groups = group_projectors_by_l(species);
    for (const auto& group : groups) {
        const int l = group.first;
        const std::vector<int>& indices = group.second;
        const int nradial = static_cast<int>(indices.size());
        Eigen::MatrixXd dij_block(nradial, nradial);
        for (int i = 0; i < nradial; ++i) {
            for (int j = 0; j < nradial; ++j) {
                dij_block(i, j) =
                    species.dij_hartree(indices[i], indices[j]);
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
            dij_block
        );
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error(
                "Failed to diagonalize an NC-UPF PP_DIJ l block."
            );
        }

        const int nm = 2 * l + 1;
        Eigen::MatrixXd harmonics(nbasis, nm);
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && \
       static_cast<long long>(nbasis) * nm >= openmp_minimum_work) \
    num_threads(thread_count)
        for (int ig = 0; ig < nbasis; ++ig) {
            const std::vector<double> values =
                qe_real_spherical_harmonics(
                    l,
                    basis.gvectors[ig].q_cart
                );
            for (int m = 0; m < nm; ++m) {
                harmonics(ig, m) = values[m];
            }
        }

        const std::complex<double> prefactor =
            inverse_sqrt_volume * minus_i_to_power(l);
        const Eigen::MatrixXcd rotation =
            eigensolver.eigenvectors()
                .cast<std::complex<double>>();
        for (int m = 0; m < nm; ++m) {
            Eigen::MatrixXcd raw(nbasis, nradial);
            const std::int64_t raw_work =
                static_cast<std::int64_t>(nbasis)
                * static_cast<std::int64_t>(nradial);
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && raw_work >= openmp_minimum_work) \
    num_threads(thread_count)
            for (std::int64_t index = 0;
                 index < raw_work;
                 ++index) {
                const int iradial = static_cast<int>(
                    index / static_cast<std::int64_t>(nbasis)
                );
                const int ig = static_cast<int>(
                    index % static_cast<std::int64_t>(nbasis)
                );
                raw(ig, iradial) =
                    prefactor
                    * harmonics(ig, m)
                    * radial_transforms(
                        radius_classes.basis_to_radius[ig],
                        indices[iradial]
                    );
            }

            Eigen::MatrixXcd rotated(nbasis, nradial);
            rotated.noalias() = raw * rotation;
            for (int channel = 0; channel < nradial; ++channel) {
                const double eigenvalue =
                    eigensolver.eigenvalues()[channel];
                if (std::abs(eigenvalue) < zero_tolerance) {
                    continue;
                }
                SpeciesProjectorTemplate projector;
                projector.beta_G = rotated.col(channel);
                projector.D = eigenvalue;
                templates.push_back(std::move(projector));
            }
        }
    }
    return templates;
}

} // namespace

UPFNonlocalSpecies prepare_upf_nonlocal_species(const UPFData& upf) {
    const int nprojectors = upf.header.number_of_projectors;
    if (nprojectors != static_cast<int>(upf.projectors.size())) {
        throw std::runtime_error(
            "UPF header and PP_BETA projector counts do not match."
        );
    }
    if (upf.mesh.r.size() != upf.mesh.rab.size()) {
        throw std::runtime_error(
            "UPF nonlocal radial mesh arrays do not have equal sizes."
        );
    }
    if (nprojectors * nprojectors
            != static_cast<int>(upf.dij_ry.size())) {
        throw std::runtime_error(
            "UPF PP_DIJ size does not match the projector count."
        );
    }

    UPFNonlocalSpecies species;
    species.element = upf.header.element;
    species.radial_grid_bohr = upf.mesh.r;
    species.quadrature_weights = make_upf_simpson_weights(upf.mesh.rab);
    species.radial_projectors = upf.projectors;
    species.dij_hartree = Eigen::MatrixXd::Zero(nprojectors, nprojectors);

    for (int i = 0; i < nprojectors; ++i) {
        for (int j = 0; j < nprojectors; ++j) {
            species.dij_hartree(i, j) =
                RYDBERG_TO_HARTREE * upf.dij(i, j);
        }
    }

    validate_species(species);
    return species;
}

std::vector<double> qe_real_spherical_harmonics(
    int l,
    const Eigen::Vector3d& G) {

    if (l < 0) {
        throw std::runtime_error("Spherical-harmonic l cannot be negative.");
    }
    if (!G.allFinite()) {
        throw std::runtime_error("Spherical-harmonic direction must be finite.");
    }

    std::vector<double> values(2 * l + 1, 0.0);
    const double radius = G.norm();
    if (radius < zero_tolerance) {
        if (l == 0) {
            values[0] = 1.0 / std::sqrt(4.0 * M_PI);
        }
        return values;
    }

    const double cos_theta = G[2] / radius;
    const double sin_theta = std::sqrt(std::max(
        0.0,
        1.0 - cos_theta * cos_theta
    ));
    const double phi = std::atan2(G[1], G[0]);

    /*
     * Q_lm = sqrt((l-m)!/(l+m)!) P_l^m(cos theta), including the
     * Condon--Shortley phase. This is the recurrence used by QE's ylmr2.
     */
    std::vector<std::vector<double>> Q(
        l + 1,
        std::vector<double>(l + 1, 0.0)
    );
    Q[0][0] = 1.0;
    if (l >= 1) {
        Q[1][0] = cos_theta;
        Q[1][1] = -sin_theta / std::sqrt(2.0);
    }
    for (int order = 2; order <= l; ++order) {
        for (int m = 0; m <= order - 2; ++m) {
            const double denominator =
                std::sqrt(
                    static_cast<double>((order + m) * (order - m))
                );
            const double previous_factor =
                std::sqrt(
                    static_cast<double>(
                        (order + m - 1) * (order - m - 1)
                    )
                );
            Q[order][m] =
                ((2.0 * order - 1.0) * cos_theta * Q[order - 1][m]
                 - previous_factor * Q[order - 2][m])
                / denominator;
        }
        Q[order][order - 1] =
            cos_theta * std::sqrt(2.0 * order - 1.0)
            * Q[order - 1][order - 1];
        Q[order][order] =
            -sin_theta
            * std::sqrt((2.0 * order - 1.0) / (2.0 * order))
            * Q[order - 1][order - 1];
    }

    const double normalization =
        std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
    values[0] = normalization * Q[l][0];
    for (int m = 1; m <= l; ++m) {
        const double pair_normalization =
            normalization * std::sqrt(2.0) * Q[l][m];
        values[2 * m - 1] = pair_normalization * std::cos(m * phi);
        values[2 * m] = pair_normalization * std::sin(m * phi);
    }
    return values;
}

std::vector<NonlocalProjector> build_upf_nonlocal_projectors(
    const Lattice& lattice,
    const PlaneWaveBasis3D& basis,
    const std::vector<UPFNonlocalSpecies>& species,
    const std::vector<UPFLocalIon>& ions,
    int thread_count) {

    if (thread_count <= 0) {
        throw std::runtime_error(
            "UPF nonlocal-projector thread count must be positive."
        );
    }
    for (const UPFNonlocalSpecies& one_species : species) {
        validate_species(one_species);
    }
    for (const UPFLocalIon& ion : ions) {
        if (ion.species_index < 0 ||
            ion.species_index >= static_cast<int>(species.size())) {
            throw std::runtime_error(
                "UPF nonlocal ion has an invalid species index."
            );
        }
        if (!ion.frac_position.allFinite()) {
            throw std::runtime_error(
                "UPF nonlocal ion has a non-finite fractional position."
            );
        }
    }

    std::vector<std::vector<SpeciesProjectorTemplate>> species_templates(
        species.size()
    );
    const ReciprocalRadiusClasses radius_classes =
        make_radius_classes(basis);
    for (int ispecies = 0;
         ispecies < static_cast<int>(species.size());
         ++ispecies) {
        species_templates[ispecies] = build_species_templates(
            lattice,
            basis,
            radius_classes,
            species[ispecies],
            thread_count
        );
    }

    std::vector<int> ion_offsets(ions.size() + 1, 0);
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        ion_offsets[iion + 1] =
            ion_offsets[iion]
            + static_cast<int>(
                species_templates[ions[iion].species_index].size()
            );
    }
    std::vector<NonlocalProjector> result(ion_offsets.back());
    const long long phase_work =
        static_cast<long long>(basis.size())
        * static_cast<long long>(ions.size());
#pragma omp parallel for schedule(static) \
    if(thread_count > 1 && phase_work >= openmp_minimum_work) \
    num_threads(thread_count)
    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        const UPFLocalIon& ion = ions[iion];
        const Eigen::Vector3d R =
            lattice.cart_from_frac(ion.frac_position);
        Eigen::VectorXcd phase(basis.size());
        for (int ig = 0; ig < basis.size(); ++ig) {
            phase[ig] = translation_phase(
                basis.gvectors[ig].q_cart,
                R
            );
        }

        const auto& templates =
            species_templates[ion.species_index];
        for (int itemplate = 0;
             itemplate < static_cast<int>(templates.size());
             ++itemplate) {
            NonlocalProjector& projector =
                result[ion_offsets[iion] + itemplate];
            projector.beta_G =
                templates[itemplate].beta_G.cwiseProduct(phase);
            projector.D = templates[itemplate].D;
            projector.ion_index = iion;
        }
    }
    return result;
}
