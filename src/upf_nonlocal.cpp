#include "upf_nonlocal.hpp"

#include "radial_transform.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

constexpr double zero_tolerance = 1.0e-14;

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
    const std::vector<UPFLocalIon>& ions) {

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

    std::vector<NonlocalProjector> result;
    const double inverse_sqrt_volume = 1.0 / std::sqrt(lattice.volume());

    for (int iion = 0; iion < static_cast<int>(ions.size()); ++iion) {
        const UPFLocalIon& ion = ions[iion];
        const UPFNonlocalSpecies& one_species = species[ion.species_index];
        const Eigen::Vector3d R =
            lattice.cart_from_frac(ion.frac_position);
        const auto groups = group_projectors_by_l(one_species);

        std::vector<std::vector<double>> radial_transforms(
            one_species.radial_projectors.size(),
            std::vector<double>(basis.size(), 0.0)
        );
        for (int ip = 0;
             ip < static_cast<int>(one_species.radial_projectors.size());
             ++ip) {
            const UPFProjector& projector =
                one_species.radial_projectors[ip];
            for (int ig = 0; ig < basis.size(); ++ig) {
                radial_transforms[ip][ig] =
                    radial_fourier_bessel_transform_from_r_times_function(
                        projector.angular_momentum,
                        basis.gvectors[ig].q_cart.norm(),
                        one_species.radial_grid_bohr,
                        one_species.quadrature_weights,
                        projector.r_times_beta
                    );
            }
        }

        for (const auto& group : groups) {
            const int l = group.first;
            const std::vector<int>& indices = group.second;
            const int nradial = static_cast<int>(indices.size());
            Eigen::MatrixXd dij_block(nradial, nradial);
            for (int i = 0; i < nradial; ++i) {
                for (int j = 0; j < nradial; ++j) {
                    dij_block(i, j) =
                        one_species.dij_hartree(indices[i], indices[j]);
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

            const std::complex<double> angular_phase =
                minus_i_to_power(l);
            const int nm = 2 * l + 1;
            for (int m = 0; m < nm; ++m) {
                std::vector<Eigen::VectorXcd> raw_projectors;
                raw_projectors.reserve(nradial);
                for (int iradial = 0; iradial < nradial; ++iradial) {
                    Eigen::VectorXcd raw =
                        Eigen::VectorXcd::Zero(basis.size());
                    const int ip = indices[iradial];
                    for (int ig = 0; ig < basis.size(); ++ig) {
                        const Eigen::Vector3d& q =
                            basis.gvectors[ig].q_cart;
                        const std::vector<double> harmonics =
                            qe_real_spherical_harmonics(l, q);
                        raw[ig] =
                            inverse_sqrt_volume
                            * angular_phase
                            * harmonics[m]
                            * radial_transforms[ip][ig]
                            * translation_phase(q, R);
                    }
                    raw_projectors.push_back(std::move(raw));
                }

                for (int channel = 0; channel < nradial; ++channel) {
                    const double eigenvalue =
                        eigensolver.eigenvalues()[channel];
                    if (std::abs(eigenvalue) < zero_tolerance) {
                        continue;
                    }

                    NonlocalProjector projector;
                    projector.beta_G =
                        Eigen::VectorXcd::Zero(basis.size());
                    projector.D = eigenvalue;
                    projector.ion_index = iion;
                    for (int iradial = 0;
                         iradial < nradial;
                         ++iradial) {
                        projector.beta_G +=
                            eigensolver.eigenvectors()(iradial, channel)
                            * raw_projectors[iradial];
                    }
                    result.push_back(std::move(projector));
                }
            }
        }
    }

    return result;
}
