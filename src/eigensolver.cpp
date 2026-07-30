#include "eigensolver.hpp"
#include "hamiltonian.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

double elapsed_seconds(
    std::chrono::steady_clock::time_point start) {

    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

void update_projected_hamiltonian(
    Eigen::MatrixXcd& projected,
    int& projected_size,
    const Eigen::MatrixXcd& V,
    const Eigen::MatrixXcd& W,
    int subspace_size,
    double& hermiticity_error,
    DavidsonTimingBreakdown& timing) {

    if (projected_size < 0 ||
        projected_size > subspace_size ||
        subspace_size > projected.rows() ||
        subspace_size > projected.cols()) {
        throw std::runtime_error(
            "Invalid incremental projected-Hamiltonian dimensions."
        );
    }
    if (projected_size == subspace_size) {
        return;
    }

    const int old_size = projected_size;
    const int added = subspace_size - old_size;

    /*
     * Hsub is Hermitian. On expansion V'=[V,T], W'=[W,HT], retain the
     * existing old-old block and compute only
     *
     *   V_old† HT  and  T† HT.
     *
     * The lower cross block is filled by Hermitian symmetry, avoiding the
     * redundant T†W_old product. This changes repeated full V†W rebuilds
     * into one initial build plus incremental upper-triangle updates.
     */
    if (old_size > 0) {
        Eigen::MatrixXcd cross(old_size, added);
        cross.noalias() =
            V.leftCols(old_size).adjoint()
            * W.middleCols(old_size, added);
        projected.block(0, old_size, old_size, added) = cross;
        projected.block(old_size, 0, added, old_size) =
            cross.adjoint();
        timing.projected_matrix_incremental_updates += 1;
    } else {
        timing.projected_matrix_full_builds += 1;
    }

    Eigen::MatrixXcd diagonal_block(added, added);
    diagonal_block.noalias() =
        V.middleCols(old_size, added).adjoint()
        * W.middleCols(old_size, added);
    hermiticity_error =
        (diagonal_block - diagonal_block.adjoint()).norm();
    projected.block(old_size, old_size, added, added) =
        0.5 * (diagonal_block + diagonal_block.adjoint());
    projected_size = subspace_size;
}

void reuse_ritz_projected_hamiltonian(
    Eigen::MatrixXcd& projected,
    int& projected_size,
    const Eigen::VectorXd& eigenvalues,
    DavidsonTimingBreakdown& timing) {

    const int count = eigenvalues.size();
    projected.topLeftCorner(count, count).setZero();
    projected.diagonal().head(count) =
        eigenvalues.cast<std::complex<double>>();
    projected_size = count;
    timing.projected_matrix_ritz_reuses += 1;
}

} // namespace

Eigen::MatrixXcd orthonormalize_columns(
    const Eigen::MatrixXcd& A,
    double drop_tol) {

    const int nrows = A.rows();
    const int ncols = A.cols();

    std::vector<Eigen::VectorXcd> qcols;
    qcols.reserve(ncols);

    for (int j = 0; j < ncols; ++j) {
        Eigen::VectorXcd v = A.col(j);

        // Do two Gram-Schmidt passes for numerical stability.
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& q : qcols) {
                v -= q * (q.adjoint() * v)(0, 0);
            }
        }

        const double nrm = v.norm();

        if (nrm > drop_tol) {
            qcols.push_back(v / nrm);
        }
    }

    Eigen::MatrixXcd Q(nrows, static_cast<int>(qcols.size()));

    for (int j = 0; j < static_cast<int>(qcols.size()); ++j) {
        Q.col(j) = qcols[j];
    }

    return Q;
}

Eigen::MatrixXcd append_columns(
    const Eigen::MatrixXcd& A,
    const std::vector<Eigen::VectorXcd>& new_cols) {

    const int nrows = A.rows();
    const int old_cols = A.cols();
    const int add_cols = static_cast<int>(new_cols.size());

    Eigen::MatrixXcd B(nrows, old_cols + add_cols);

    B.leftCols(old_cols) = A;

    for (int j = 0; j < add_cols; ++j) {
        B.col(old_cols + j) = new_cols[j];
    }

    return B;
}

Eigen::MatrixXcd orthonormalize_correction_block(
    const Eigen::Ref<const Eigen::MatrixXcd>& subspace,
    const Eigen::Ref<const Eigen::MatrixXcd>& raw_corrections,
    double drop_tol,
    CorrectionOrthogonalizationInfo* info) {

    if (info != nullptr) {
        info->projection_passes = 0;
    }

    if (subspace.rows() != raw_corrections.rows()) {
        throw std::runtime_error(
            "Correction block and Davidson subspace row sizes differ."
        );
    }
    if (!std::isfinite(drop_tol) || drop_tol < 0.0) {
        throw std::runtime_error(
            "Correction block drop tolerance must be finite and nonnegative."
        );
    }
    if (raw_corrections.cols() == 0) {
        return Eigen::MatrixXcd(raw_corrections.rows(), 0);
    }

    Eigen::MatrixXcd T = raw_corrections;

    /*
     * Block classical Gram--Schmidt with the Daniel--Gragg--Kaufman--Stewart
     * (DGKS) reorthogonalization criterion. The first pass is mandatory:
     *
     *   T <- T - V(V†T).
     *
     * A second block pass is needed only when a column loses more than half
     * of its norm, which signals substantial cancellation against V. This
     * retains the stable two-pass path for difficult corrections while
     * avoiding two GEMMs for already-nearly-orthogonal blocks.
     */
    if (subspace.cols() > 0) {
        constexpr double dgks_reorthogonalization_threshold = 0.5;
        const Eigen::VectorXd norms_before =
            T.colwise().norm().transpose();
        Eigen::MatrixXcd coefficients(
            subspace.cols(),
            T.cols()
        );
        coefficients.noalias() = subspace.adjoint() * T;
        T.noalias() -= subspace * coefficients;
        if (info != nullptr) {
            info->projection_passes = 1;
        }

        const Eigen::VectorXd norms_after =
            T.colwise().norm().transpose();
        bool needs_reorthogonalization = false;
        for (int j = 0; j < T.cols(); ++j) {
            if (norms_before[j] > 0.0 &&
                norms_after[j]
                    < dgks_reorthogonalization_threshold
                        * norms_before[j]) {
                needs_reorthogonalization = true;
                break;
            }
        }
        if (needs_reorthogonalization) {
            coefficients.noalias() = subspace.adjoint() * T;
            T.noalias() -= subspace * coefficients;
            if (info != nullptr) {
                info->projection_passes = 2;
            }
        }
    }

    /*
     * Symmetric block orthogonalization from the small Gram matrix:
     *
     *   S = T†T = U Lambda U†
     *   Q = T U Lambda^{-1/2}.
     *
     * Eigenvalues below the absolute vector drop tolerance or the roundoff
     * floor of S are discarded, so dependent correction directions do not
     * enter the Davidson subspace.
     */
    Eigen::MatrixXcd gram(T.cols(), T.cols());
    gram.noalias() = T.adjoint() * T;
    gram = 0.5 * (gram + gram.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(gram);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Correction-block Gram diagonalization failed."
        );
    }

    const Eigen::VectorXd eigenvalues = solver.eigenvalues();
    const double largest_eigenvalue =
        std::max(0.0, eigenvalues[eigenvalues.size() - 1]);
    const double roundoff_threshold =
        64.0
        * std::numeric_limits<double>::epsilon()
        * static_cast<double>(T.cols())
        * largest_eigenvalue;
    const double eigenvalue_threshold = std::max(
        drop_tol * drop_tol,
        roundoff_threshold
    );

    int first_kept = 0;
    while (first_kept < eigenvalues.size() &&
           eigenvalues[first_kept] <= eigenvalue_threshold) {
        ++first_kept;
    }

    const int kept = eigenvalues.size() - first_kept;
    if (kept == 0) {
        return Eigen::MatrixXcd(T.rows(), 0);
    }

    Eigen::MatrixXcd transform =
        solver.eigenvectors().rightCols(kept);
    for (int j = 0; j < kept; ++j) {
        transform.col(j) /=
            std::sqrt(eigenvalues[first_kept + j]);
    }

    Eigen::MatrixXcd Q(T.rows(), kept);
    Q.noalias() = T * transform;
    return Q;
}

Eigen::MatrixXcd initial_low_kinetic_trials(
    int nbasis,
    int ntrial) {

    if (ntrial > nbasis) {
        throw std::runtime_error("ntrial cannot exceed nbasis.");
    }

    Eigen::MatrixXcd V =
        Eigen::MatrixXcd::Zero(nbasis, ntrial);

    for (int i = 0; i < ntrial; ++i) {
        V(i, i) = std::complex<double>(1.0, 0.0);
    }

    return V;
}

DavidsonResult davidson_lowest_eigenstates(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    int nbands,
    const Eigen::MatrixXcd& initial_trials,
    int max_iter,
    int max_subspace,
    double residual_tol,
    double denom_floor,
    const std::vector<NonlocalProjector>* projectors,
    bool verbose,
    const std::vector<double>* band_residual_tolerances) {

    const int nbasis = basis.size();

    if (nbands <= 0) {
        throw std::runtime_error("nbands must be positive.");
    }

    if (nbands > nbasis) {
        throw std::runtime_error("nbands cannot exceed basis size.");
    }

    if (initial_trials.rows() != nbasis) {
        throw std::runtime_error("Initial trial vector row size mismatch.");
    }

    if (initial_trials.cols() < nbands) {
        throw std::runtime_error("Number of trial vectors must be >= nbands.");
    }
    if (max_iter <= 0) {
        throw std::runtime_error("Davidson max_iter must be positive.");
    }
    if (max_subspace < nbands) {
        throw std::runtime_error(
            "Davidson max_subspace cannot be smaller than the band count."
        );
    }
    if (!std::isfinite(residual_tol) || residual_tol <= 0.0) {
        throw std::runtime_error(
            "Davidson residual tolerance must be positive and finite."
        );
    }
    if (band_residual_tolerances != nullptr) {
        if (static_cast<int>(band_residual_tolerances->size()) != nbands) {
            throw std::runtime_error(
                "Davidson band-residual tolerance count must match nbands."
            );
        }
        for (double tolerance : *band_residual_tolerances) {
            if (!std::isfinite(tolerance) || tolerance <= 0.0) {
                throw std::runtime_error(
                    "Davidson band-residual tolerances must be positive "
                    "and finite."
                );
            }
        }
    }
    if (!std::isfinite(denom_floor) || denom_floor <= 0.0) {
        throw std::runtime_error(
            "Davidson denominator floor must be positive and finite."
        );
    }

    DavidsonResult result;
    const auto davidson_start = std::chrono::steady_clock::now();

    auto phase_start = std::chrono::steady_clock::now();
    Eigen::MatrixXcd initial_subspace =
        orthonormalize_columns(initial_trials);
    result.timing.initial_orthonormalization_seconds +=
        elapsed_seconds(phase_start);

    if (initial_subspace.cols() < nbands) {
        throw std::runtime_error(
            "Initial trials became rank-deficient after orthonormalization."
        );
    }
    if (initial_subspace.cols() > max_subspace) {
        throw std::runtime_error(
            "Initial Davidson subspace exceeds max_subspace."
        );
    }

    /*
     * Cache W = H V. When correction vectors are appended later, only the
     * new columns need an H application. The old implementation recomputed
     * the complete block on every Davidson iteration.
     */
    auto apply_hamiltonian_counted = [&](const Eigen::MatrixXcd& block) {
        const auto start = std::chrono::steady_clock::now();
        Eigen::MatrixXcd image = apply_hamiltonian_to_block(
            basis,
            fft,
            V_r,
            block,
            projectors
        );
        const auto stop = std::chrono::steady_clock::now();
        result.hamiltonian_seconds +=
            std::chrono::duration<double>(stop - start).count();
        result.hamiltonian_applications += block.cols();
        result.hamiltonian_block_calls += 1;
        return image;
    };

    /*
     * Allocate the maximum Davidson capacity once. Only the first
     * subspace_size columns are active; expansion and thick restart write
     * into these buffers without reallocating and copying all old columns.
     */
    Eigen::MatrixXcd V(nbasis, max_subspace);
    Eigen::MatrixXcd W(nbasis, max_subspace);
    Eigen::MatrixXcd projected =
        Eigen::MatrixXcd::Zero(max_subspace, max_subspace);
    int projected_size = 0;
    int subspace_size = initial_subspace.cols();
    V.leftCols(subspace_size) = initial_subspace;
    W.leftCols(subspace_size) =
        apply_hamiltonian_counted(initial_subspace);

    double residual_at_last_empty_restart =
        std::numeric_limits<double>::infinity();
    int unsuccessful_empty_restarts = 0;
    constexpr int maximum_unsuccessful_empty_restarts = 3;
    constexpr double correction_drop_tolerance = 1.0e-14;

    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        result.final_subspace_size = subspace_size;

        phase_start = std::chrono::steady_clock::now();
        /*
         * 1. Incrementally update H_sub = V†W, then copy only the active
         * small matrix for diagonalization.
         */
        update_projected_hamiltonian(
            projected,
            projected_size,
            V,
            W,
            subspace_size,
            result.projected_hermiticity_error,
            result.timing
        );
        const Eigen::MatrixXcd Hsub =
            projected.topLeftCorner(subspace_size, subspace_size);
        result.timing.projected_matrix_seconds +=
            elapsed_seconds(phase_start);

        /*
         * 2. Diagonalize subspace Hamiltonian.
         */
        const auto diagonalization_start =
            std::chrono::steady_clock::now();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(Hsub);
        const auto diagonalization_stop =
            std::chrono::steady_clock::now();
        result.subspace_diagonalization_seconds +=
            std::chrono::duration<double>(
                diagonalization_stop - diagonalization_start
            ).count();

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Davidson subspace diagonalization failed.");
        }

        const Eigen::VectorXd eps_all = solver.eigenvalues();
        const Eigen::MatrixXcd A_all = solver.eigenvectors();

        Eigen::VectorXd eps = eps_all.head(nbands);
        Eigen::MatrixXcd A = A_all.leftCols(nbands);

        /*
         * 3. Ritz vectors:
         *
         *    X  = V A
         *    HX = W A
         */
        phase_start = std::chrono::steady_clock::now();
        Eigen::MatrixXcd X(nbasis, nbands);
        Eigen::MatrixXcd HX(nbasis, nbands);
        X.noalias() = V.leftCols(subspace_size) * A;
        HX.noalias() = W.leftCols(subspace_size) * A;
        result.timing.ritz_rotation_seconds +=
            elapsed_seconds(phase_start);

        /*
         * 4. Residuals:
         *
         *    r_i = Hx_i - eps_i x_i
         */
        phase_start = std::chrono::steady_clock::now();
        std::vector<double> residual_norms(nbands, 0.0);
        Eigen::MatrixXcd raw_corrections(nbasis, nbands);
        int correction_count = 0;

        bool all_converged = true;
        double maximum_residual = 0.0;
        int maximum_residual_band = -1;

        for (int ib = 0; ib < nbands; ++ib) {
            const double band_tolerance =
                band_residual_tolerances == nullptr
                ? residual_tol
                : (*band_residual_tolerances)[ib];
            Eigen::VectorXcd r =
                HX.col(ib) - eps[ib] * X.col(ib);

            const double rnorm = r.norm();
            residual_norms[ib] = rnorm;
            if (rnorm > maximum_residual) {
                maximum_residual = rnorm;
                maximum_residual_band = ib;
            }

            if (rnorm > band_tolerance) {
                all_converged = false;

                /*
                 * 5. Davidson correction:
                 *
                 *    t(G) = -r(G) / (D(G) - eps)
                 *
                 * where D(G) ≈ kinetic(G).
                 */
                for (int ig = 0; ig < nbasis; ++ig) {
                    double denom =
                        basis.gvectors[ig].kinetic - eps[ib];

                    if (std::abs(denom) < denom_floor) {
                        denom = (denom >= 0.0)
                              ? denom_floor
                              : -denom_floor;
                    }

                    raw_corrections(ig, correction_count) =
                        -r[ig] / denom;
                }

                ++correction_count;
            }
        }
        result.timing.residual_preconditioner_seconds +=
            elapsed_seconds(phase_start);

        phase_start = std::chrono::steady_clock::now();
        result.eigenvalues = eps;
        result.eigenvectors = X;
        result.residual_norms = residual_norms;
        result.converged = all_converged;
        result.max_residual_band = maximum_residual_band;
        result.timing.result_copy_seconds +=
            elapsed_seconds(phase_start);

        if (verbose) {
            std::cout << "Davidson iter " << std::setw(3) << iter + 1
                      << "  subspace = " << std::setw(4) << subspace_size
                      << "  eps[0] = " << std::setw(20) << eps[0]
                      << "  max_res = " << maximum_residual
                      << "  band = " << maximum_residual_band
                      << "  nHpsi = " << result.hamiltonian_applications
                      << "\n";
        }

        if (all_converged) {
            result.total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - davidson_start
            ).count();
            return result;
        }

        /*
         * 6. Orthogonalize corrections against V and one another before
         * deciding whether the subspace needs a restart. DGKS requests a
         * second pass only after substantial cancellation in the first.
         */
        phase_start = std::chrono::steady_clock::now();
        CorrectionOrthogonalizationInfo correction_info;
        Eigen::MatrixXcd T = orthonormalize_correction_block(
            V.leftCols(subspace_size),
            raw_corrections.leftCols(correction_count),
            correction_drop_tolerance,
            &correction_info
        );
        result.timing.correction_blocks += 1;
        if (correction_info.projection_passes > 1) {
            result.timing.correction_reorthogonalizations += 1;
        }
        result.timing.correction_orthogonalization_seconds +=
            elapsed_seconds(phase_start);

        /*
         * If every raw correction was removed by orthogonalization, the old
         * code appended them anyway and orthonormalize_columns silently
         * dropped them. V then kept the same size forever. Restart from the
         * Ritz block and stop with a diagnostic after repeated no-progress
         * restarts.
         */
        if (T.cols() == 0) {
            result.stagnation_restarts += 1;
            result.subspace_restarts += 1;

            const bool improved_since_last_empty =
                maximum_residual
                < 0.999 * residual_at_last_empty_restart;
            unsuccessful_empty_restarts = improved_since_last_empty
                ? 1
                : unsuccessful_empty_restarts + 1;
            residual_at_last_empty_restart = maximum_residual;

            phase_start = std::chrono::steady_clock::now();
            V.leftCols(nbands) = X;
            W.leftCols(nbands) = HX;
            subspace_size = nbands;
            reuse_ritz_projected_hamiltonian(
                projected,
                projected_size,
                eps,
                result.timing
            );
            result.projected_hermiticity_error = 0.0;
            result.timing.restart_seconds +=
                elapsed_seconds(phase_start);
            result.final_subspace_size = subspace_size;

            if (unsuccessful_empty_restarts
                >= maximum_unsuccessful_empty_restarts) {
                result.stagnated = true;
                result.total_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - davidson_start
                ).count();
                return result;
            }
            continue;
        }

        /*
         * 7. Thick restart: X and HX are already matching Ritz/image pairs,
         * so no new H application is required by the restart itself.
         */
        if (subspace_size + T.cols() > max_subspace) {
            phase_start = std::chrono::steady_clock::now();
            V.leftCols(nbands) = X;
            W.leftCols(nbands) = HX;
            subspace_size = nbands;
            reuse_ritz_projected_hamiltonian(
                projected,
                projected_size,
                eps,
                result.timing
            );
            result.projected_hermiticity_error = 0.0;
            result.timing.restart_seconds +=
                elapsed_seconds(phase_start);
            result.subspace_restarts += 1;
            result.final_subspace_size = subspace_size;
            continue;
        }

        /*
         * 8. Apply H only to genuinely new orthonormal directions, then
         * append the matching (V, HV) column blocks.
         */
        const Eigen::MatrixXcd HT = apply_hamiltonian_counted(T);

        phase_start = std::chrono::steady_clock::now();
        V.middleCols(subspace_size, T.cols()) = T;
        W.middleCols(subspace_size, T.cols()) = HT;
        subspace_size += T.cols();
        result.timing.subspace_expansion_seconds +=
            elapsed_seconds(phase_start);
        result.final_subspace_size = subspace_size;
    }

    result.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - davidson_start
    ).count();
    return result;
}

Eigen::MatrixXcd orthonormalize_columns_RMM(
    const Eigen::MatrixXcd& A,
    double drop_tol) {

    const int nrows = A.rows();
    const int ncols = A.cols();

    std::vector<Eigen::VectorXcd> qcols;
    qcols.reserve(ncols);

    for (int j = 0; j < ncols; ++j) {
        Eigen::VectorXcd v = A.col(j);

        // Do two Gram-Schmidt passes for numerical stability.
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& q : qcols) {
                v -= q * (q.adjoint() * v)(0, 0);
            }
        }

        

        
        qcols.push_back(v);
        
    }

    Eigen::MatrixXcd Q(nrows, static_cast<int>(qcols.size()));

    for (int j = 0; j < static_cast<int>(qcols.size()); ++j) {
        Q.col(j) = qcols[j];
    }

    return Q;
}

CanonicalBlock canonicalize_block(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& C_in) {

    CanonicalBlock out;

    Eigen::MatrixXcd C = orthonormalize_columns(C_in);///////////////////

    Eigen::MatrixXcd HC =
        apply_hamiltonian_to_block(basis, fft, V_r, C);

    Eigen::MatrixXcd Hsub = C.adjoint() * HC;
    Hsub = 0.5 * (Hsub + Hsub.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(Hsub);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Subspace diagonalization failed.");
    }

    Eigen::MatrixXcd U = solver.eigenvectors();
    Eigen::VectorXd eps = solver.eigenvalues();

    out.C = C * U;
    out.HC = HC * U;
    out.eps = eps;

    const int nbands = out.C.cols();

    out.R = out.HC;

    for (int ib = 0; ib < nbands; ++ib) {
        out.R.col(ib) -= out.eps[ib] * out.C.col(ib);
    }

    out.residual_norms.resize(nbands);

    for (int ib = 0; ib < nbands; ++ib) {
        out.residual_norms[ib] = out.R.col(ib).norm();
    }

    return out;
}

Eigen::MatrixXcd build_preconditioned_correction_block(
    const PlaneWaveBasis3D& basis,
    const Eigen::MatrixXcd& R,
    const Eigen::VectorXd& eps,
    double denom_floor) {

    const int nbasis = basis.size();
    const int nbands = R.cols();

    Eigen::MatrixXcd T(nbasis, nbands);

    for (int ib = 0; ib < nbands; ++ib) {
        for (int ig = 0; ig < nbasis; ++ig) {
            double denom = basis.gvectors[ig].kinetic - eps[ib];

            if (std::abs(denom) < denom_floor) {
                denom = (denom >= 0.0) ? denom_floor : -denom_floor;
            }

            T(ig, ib) = -R(ig, ib) / denom;
        }
    }

    return T;
}

Eigen::MatrixXcd orthogonalize_correction_against_C(
    const Eigen::MatrixXcd& C,
    const Eigen::MatrixXcd& T_raw,
    double drop_tol) {

    Eigen::MatrixXcd T = T_raw;

    /*
     * Remove components parallel to current occupied / target subspace:
     *
     *   T <- T - C(C†T)
     */
    T -= C * (C.adjoint() * T);

    /*
     * Orthonormalize correction columns internally.
     */
    T = orthonormalize_columns_RMM(T, drop_tol);

    return T;
}

Eigen::MatrixXcd project_correction_against_C_keep_magnitude(
    const Eigen::MatrixXcd& C,
    const Eigen::MatrixXcd& T_raw) {

    Eigen::MatrixXcd T = T_raw;

    /*
     * Remove components inside current orbital subspace:
     *
     *   T <- T - C(C†T)
     *
     * Here C is assumed to be orthonormal.
     */
    T -= C * (C.adjoint() * T);

    return T;
}

double max_residual_norm(
    const std::vector<double>& residual_norms) {

    double max_res = 0.0;

    for (double r : residual_norms) {
        max_res = std::max(max_res, r);
    }

    return max_res;
}

CanonicalBlock augmented_rmm_step(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const CanonicalBlock& cur,
    double denom_floor,
    double residual_tol) {

    const int nbasis = basis.size();
    const int nbands = cur.C.cols();

    /*
     * 1. Build preconditioned residual correction:
     *
     *    T_raw(G, ib) = -R(G, ib) / (D_G - eps_ib)
     */
    Eigen::MatrixXcd T_raw =
        build_preconditioned_correction_block(
            basis,
            cur.R,
            cur.eps,
            denom_floor
        );

    /*
     * 2. Project correction out of current orbital subspace.
     */
    Eigen::MatrixXcd T =
        project_correction_against_C_keep_magnitude(
            cur.C,
            T_raw
        );

    /*
     * 3. Collect useful correction directions.
     *
     * Do not add correction for already converged bands.
     * Also skip numerically tiny directions.
     */
    std::vector<Eigen::VectorXcd> correction_cols;

    const double correction_drop_tol = 1.0e-14;

    for (int ib = 0; ib < nbands; ++ib) {
        if (cur.residual_norms[ib] < residual_tol) {
            continue;
        }

        const double nrm = T.col(ib).norm();

        if (nrm > correction_drop_tol) {
            correction_cols.push_back(T.col(ib));
        }
    }

    /*
     * If all target bands are already converged or no useful correction
     * direction exists, return current block.
     */
    if (correction_cols.empty()) {
        return cur;
    }

    /*
     * 4. Build augmented subspace:
     *
     *    Q = [C, T_1, T_2, ...]
     */
    const int ncor = static_cast<int>(correction_cols.size());

    Eigen::MatrixXcd Q(nbasis, nbands + ncor);

    Q.leftCols(nbands) = cur.C;

    for (int j = 0; j < ncor; ++j) {
        Q.col(nbands + j) = correction_cols[j];
    }

    /*
     * 5. Orthonormalize augmented subspace.
     *
     * Here it is fine to normalize correction directions because they
     * are not directly added to orbitals. They only define a subspace.
     */
    Q = orthonormalize_columns(Q);

    if (Q.cols() < nbands) {
        throw std::runtime_error(
            "Augmented subspace rank dropped below nbands."
        );
    }

    /*
     * 6. Apply H to the whole augmented subspace.
     */
    Eigen::MatrixXcd HQ =
        apply_hamiltonian_to_block(
            basis,
            fft,
            V_r,
            Q
        );

    /*
     * 7. Rayleigh-Ritz in augmented subspace.
     */
    Eigen::MatrixXcd Hsub = Q.adjoint() * HQ;
    Hsub = 0.5 * (Hsub + Hsub.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(Hsub);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Augmented RMM subspace diagonalization failed."
        );
    }

    /*
     * 8. Take the lowest nbands Ritz vectors.
     */
    Eigen::MatrixXcd U =
        solver.eigenvectors().leftCols(nbands);

    Eigen::MatrixXcd C_new = Q * U;

    /*
     * 9. Canonicalize again in the nbands-dimensional subspace.
     *
     * This gives sorted eps, HC, residuals.
     */
    return canonicalize_block(
        basis,
        fft,
        V_r,
        C_new
    );
}

Eigen::VectorXd solve_block_diis_coefficients(
    const std::vector<Eigen::MatrixXcd>& R_hist,
    double regularization) {
    

    const int m = static_cast<int>(R_hist.size());

    if (m == 0) {
        throw std::runtime_error("Empty block DIIS history.");
    }

    if (m == 1) {
        Eigen::VectorXd c(1);
        c[0] = 1.0;
        return c;
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            std::complex<double> val =
                (R_hist[i].adjoint() * R_hist[j]).trace();

            A(i, j) = val.real();
        }

        A(i, i) += regularization;

        A(i, m) = -1.0;
        A(m, i) = -1.0;
    }

    b[m] = -1.0;

    Eigen::VectorXd x = A.fullPivLu().solve(b);

    return x.head(m);
}

Eigen::MatrixXcd combine_block_diis_orbitals(
    const std::vector<Eigen::MatrixXcd>& C_hist,
    const Eigen::VectorXd& coeffs) {

    const int m = static_cast<int>(C_hist.size());

    if (coeffs.size() != m) {
        throw std::runtime_error("Block DIIS coefficient size mismatch.");
    }

    Eigen::MatrixXcd C =
        Eigen::MatrixXcd::Zero(C_hist[0].rows(), C_hist[0].cols());

    for (int i = 0; i < m; ++i) {
        C += coeffs[i] * C_hist[i];
    }

    C = orthonormalize_columns(C);

    return C;
}


BlockRMMResult block_rmm_diis_eigenstates(
    const PlaneWaveBasis3D& basis,
    FFTWorkspace& fft,
    const std::vector<double>& V_r,
    const Eigen::MatrixXcd& C_initial,
    int max_iter,
    int history_size,
    double residual_tol,
    double denom_floor,
    double step_scale) {

    const int nbasis = basis.size();
    const int nbands = C_initial.cols();

    if (C_initial.rows() != nbasis) {
        throw std::runtime_error("Initial C row size mismatch.");
    }

    BlockRMMResult result;

    Eigen::MatrixXcd C = orthonormalize_columns(C_initial);

    if (C.cols() != nbands) {
        throw std::runtime_error("Initial C is rank deficient.");
    }

    std::vector<Eigen::MatrixXcd> C_hist;
    std::vector<Eigen::MatrixXcd> R_hist;

    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        /*
         * 1. Canonicalize current block.
         */
        CanonicalBlock cur =
            canonicalize_block(basis, fft, V_r, C);

        double max_res = 0.0;
        for (double rn : cur.residual_norms) {
            max_res = std::max(max_res, rn);
        }

        result.eigenvalues = cur.eps;
        result.eigenvectors = cur.C;
        result.residual_norms = cur.residual_norms;
        result.converged = (max_res < residual_tol);

        std::cout << "Block RMM-DIIS iter "
                  << std::setw(3) << iter + 1
                  << "  eps[7] = " << std::setw(20) << cur.eps[7]
                  << "  max_res = " << max_res
                  << "  hist = " << C_hist.size()
                  << "\n";

        if (result.converged) {
            return result;
        }

        CanonicalBlock next =
        augmented_rmm_step(
            basis,
            fft,
            V_r,
            cur,
            denom_floor,
            residual_tol
        );

        C = next.C;
    }

    CanonicalBlock final =
        canonicalize_block(basis, fft, V_r, C);

    double max_res = 0.0;
    for (double rn : final.residual_norms) {
        max_res = std::max(max_res, rn);
    }

    result.eigenvalues = final.eps;
    result.eigenvectors = final.C;
    result.residual_norms = final.residual_norms;
    result.converged = (max_res < residual_tol);
    result.iterations = max_iter;

    return result;
}
