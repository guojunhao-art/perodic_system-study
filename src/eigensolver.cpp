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
    bool verbose) {

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
    if (!std::isfinite(denom_floor) || denom_floor <= 0.0) {
        throw std::runtime_error(
            "Davidson denominator floor must be positive and finite."
        );
    }

    DavidsonResult result;
    const auto davidson_start = std::chrono::steady_clock::now();

    auto phase_start = std::chrono::steady_clock::now();
    Eigen::MatrixXcd V = orthonormalize_columns(initial_trials);
    result.timing.initial_orthonormalization_seconds +=
        elapsed_seconds(phase_start);

    if (V.cols() < nbands) {
        throw std::runtime_error(
            "Initial trials became rank-deficient after orthonormalization."
        );
    }
    if (V.cols() > max_subspace) {
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

    Eigen::MatrixXcd W = apply_hamiltonian_counted(V);
    double residual_at_last_empty_restart =
        std::numeric_limits<double>::infinity();
    int unsuccessful_empty_restarts = 0;
    constexpr int maximum_unsuccessful_empty_restarts = 3;
    constexpr double correction_drop_tolerance = 1.0e-14;

    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        result.final_subspace_size = V.cols();

        phase_start = std::chrono::steady_clock::now();
        /*
         * 1. H_sub = V† H V = V† W
         */
        Eigen::MatrixXcd Hsub = V.adjoint() * W;
        result.projected_hermiticity_error =
            (Hsub - Hsub.adjoint()).norm();

        /*
         * Enforce Hermiticity numerically.
         * This removes tiny roundoff asymmetry.
         */
        Hsub = 0.5 * (Hsub + Hsub.adjoint());
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
        Eigen::MatrixXcd X = V * A;
        Eigen::MatrixXcd HX = W * A;
        result.timing.ritz_rotation_seconds +=
            elapsed_seconds(phase_start);

        /*
         * 4. Residuals:
         *
         *    r_i = Hx_i - eps_i x_i
         */
        phase_start = std::chrono::steady_clock::now();
        std::vector<double> residual_norms(nbands, 0.0);
        std::vector<Eigen::VectorXcd> correction_vectors;

        bool all_converged = true;
        double maximum_residual = 0.0;
        int maximum_residual_band = -1;

        for (int ib = 0; ib < nbands; ++ib) {
            Eigen::VectorXcd r =
                HX.col(ib) - eps[ib] * X.col(ib);

            const double rnorm = r.norm();
            residual_norms[ib] = rnorm;
            if (rnorm > maximum_residual) {
                maximum_residual = rnorm;
                maximum_residual_band = ib;
            }

            if (rnorm > residual_tol) {
                all_converged = false;

                /*
                 * 5. Davidson correction:
                 *
                 *    t(G) = -r(G) / (D(G) - eps)
                 *
                 * where D(G) ≈ kinetic(G).
                 */
                Eigen::VectorXcd t(nbasis);

                for (int ig = 0; ig < nbasis; ++ig) {
                    double denom =
                        basis.gvectors[ig].kinetic - eps[ib];

                    if (std::abs(denom) < denom_floor) {
                        denom = (denom >= 0.0)
                              ? denom_floor
                              : -denom_floor;
                    }

                    t[ig] = -r[ig] / denom;
                }

                correction_vectors.push_back(t);
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
                      << "  subspace = " << std::setw(4) << V.cols()
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
         * deciding whether the subspace needs a restart. Two passes suppress
         * loss of orthogonality for nearly converged vectors.
         */
        phase_start = std::chrono::steady_clock::now();
        std::vector<Eigen::VectorXcd> accepted;
        accepted.reserve(correction_vectors.size());
        for (const Eigen::VectorXcd& t_raw : correction_vectors) {
            Eigen::VectorXcd t = t_raw;

            for (int pass = 0; pass < 2; ++pass) {
                t -= V * (V.adjoint() * t);
                for (const Eigen::VectorXcd& q : accepted) {
                    t -= q * q.dot(t);
                }
            }

            const double nrm = t.norm();
            if (nrm > correction_drop_tolerance) {
                accepted.push_back(t / nrm);
            }
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
        if (accepted.empty()) {
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
            V = X;
            W = HX;
            result.timing.restart_seconds +=
                elapsed_seconds(phase_start);
            result.final_subspace_size = V.cols();

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
        if (V.cols() + static_cast<int>(accepted.size()) > max_subspace) {
            phase_start = std::chrono::steady_clock::now();
            V = X;
            W = HX;
            result.timing.restart_seconds +=
                elapsed_seconds(phase_start);
            result.subspace_restarts += 1;
            result.final_subspace_size = V.cols();
            continue;
        }

        /*
         * 8. Apply H only to genuinely new orthonormal directions, then
         * append the matching (V, HV) column blocks.
         */
        phase_start = std::chrono::steady_clock::now();
        Eigen::MatrixXcd T(nbasis, static_cast<int>(accepted.size()));
        for (int j = 0; j < static_cast<int>(accepted.size()); ++j) {
            T.col(j) = accepted[j];
        }
        result.timing.correction_block_assembly_seconds +=
            elapsed_seconds(phase_start);

        const Eigen::MatrixXcd HT = apply_hamiltonian_counted(T);

        phase_start = std::chrono::steady_clock::now();
        V = append_columns(V, accepted);

        std::vector<Eigen::VectorXcd> haccepted;
        haccepted.reserve(HT.cols());
        for (int j = 0; j < HT.cols(); ++j) {
            haccepted.push_back(HT.col(j));
        }
        W = append_columns(W, haccepted);
        result.timing.subspace_expansion_seconds +=
            elapsed_seconds(phase_start);
        result.final_subspace_size = V.cols();
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
