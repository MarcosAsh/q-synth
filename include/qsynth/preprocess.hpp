// Preprocessing pipeline from the paper (Sec. III-A):
//   SelectKBest (ANOVA F)  ->  keep fraud rows  ->  standardize
//   ->  PCA(n=4)  ->  per-component max-abs normalization into [-1, 1].
//
// The fitted transform is stored so synthetic samples produced in the bounded
// representation space can be mapped back to the original feature space.
#pragma once

#include "linalg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace qsynth {

// Result of the full preprocessing fit/transform.
struct Preprocessor {
    int k_best = 10;       // features kept by SelectKBest
    int n_components = 4;  // PCA target dimensionality (== qubit count)
    double eps = 1e-8;     // normalization floor

    std::vector<int> selected;        // indices into original columns, size k_best
    Vec mu;                           // standardization mean, size k_best
    Vec sigma;                        // standardization std,  size k_best
    Matrix components;                // PCA projection W, (k_best x n_components)
    Vec max_abs;                      // per-component normalizer, size n_components

    // Fraud samples in the bounded [-1,1] representation space (real data).
    Matrix x_norm;
};

// ANOVA F-statistic of one feature column against a binary label.
inline double anova_f(const Vec &x, const std::vector<int> &y)
{
    double mean_all = 0.0;
    std::array<double, 2> sum{0.0, 0.0};
    std::array<long, 2> cnt{0, 0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        mean_all += x[i];
        sum[y[i]] += x[i];
        cnt[y[i]] += 1;
    }
    const double n = static_cast<double>(x.size());
    mean_all /= n;
    if (cnt[0] == 0 || cnt[1] == 0) {
        return 0.0;
    }
    std::array<double, 2> mean{sum[0] / cnt[0], sum[1] / cnt[1]};

    double ss_between = 0.0;
    for (int g = 0; g < 2; ++g) {
        double d = mean[g] - mean_all;
        ss_between += cnt[g] * d * d;
    }
    double ss_within = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        double d = x[i] - mean[y[i]];
        ss_within += d * d;
    }
    const double df_between = 1.0;             // groups - 1
    const double df_within = n - 2.0;          // n - groups
    if (ss_within <= 0.0 || df_within <= 0.0) {
        return 0.0;
    }
    return (ss_between / df_between) / (ss_within / df_within);
}

// Jacobi eigenvalue decomposition of a symmetric matrix. Returns eigenvalues
// (descending) with eigenvectors as columns of `vecs`. Plenty accurate for the
// small (k x k) covariance matrices used here.
inline void jacobi_eigen(Matrix a, Vec &vals, Matrix &vecs,
                         int max_sweeps = 100)
{
    const std::size_t n = a.rows;
    vecs = Matrix(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        vecs(i, i) = 1.0;
    }
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                off += a(p, q) * a(p, q);
            }
        }
        if (off < 1e-20) {
            break;
        }
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::fabs(a(p, q)) < 1e-18) {
                    continue;
                }
                double theta = (a(q, q) - a(p, p)) / (2.0 * a(p, q));
                double t = (theta >= 0 ? 1.0 : -1.0)
                           / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                for (std::size_t i = 0; i < n; ++i) {
                    double aip = a(i, p), aiq = a(i, q);
                    a(i, p) = c * aip - s * aiq;
                    a(i, q) = s * aip + c * aiq;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    double api = a(p, i), aqi = a(q, i);
                    a(p, i) = c * api - s * aqi;
                    a(q, i) = s * api + c * aqi;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    double vip = vecs(i, p), viq = vecs(i, q);
                    vecs(i, p) = c * vip - s * viq;
                    vecs(i, q) = s * vip + c * viq;
                }
            }
        }
    }
    vals.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        vals[i] = a(i, i);
    }
    // Sort eigenpairs by descending eigenvalue.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t i, std::size_t j) { return vals[i] > vals[j]; });
    Vec sorted_vals(n);
    Matrix sorted_vecs(n, n);
    for (std::size_t c = 0; c < n; ++c) {
        sorted_vals[c] = vals[order[c]];
        for (std::size_t r = 0; r < n; ++r) {
            sorted_vecs(r, c) = vecs(r, order[c]);
        }
    }
    vals = sorted_vals;
    vecs = sorted_vecs;
}

// Fit the full preprocessing pipeline on a labeled dataset and return the
// bounded fraud representation along with the inverse-transform parameters.
inline Preprocessor fit_preprocess(const Matrix &X, const std::vector<int> &y,
                                   int k_best = 10, int n_components = 4,
                                   double eps = 1e-8)
{
    Preprocessor pp;
    pp.k_best = k_best;
    pp.n_components = n_components;
    pp.eps = eps;

    // 1) SelectKBest by ANOVA F-statistic over the full labeled dataset.
    std::vector<std::pair<double, int>> scored;
    scored.reserve(X.cols);
    for (std::size_t c = 0; c < X.cols; ++c) {
        scored.emplace_back(anova_f(X.col(c), y), static_cast<int>(c));
    }
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b) { return a.first > b.first; });
    int keep = std::min<int>(k_best, static_cast<int>(X.cols));
    for (int i = 0; i < keep; ++i) {
        pp.selected.push_back(scored[i].second);
    }
    std::sort(pp.selected.begin(), pp.selected.end());

    // 2) Restrict to fraud rows over the selected columns.
    std::vector<std::size_t> fraud_rows;
    for (std::size_t r = 0; r < X.rows; ++r) {
        if (y[r] == 1) {
            fraud_rows.push_back(r);
        }
    }
    const std::size_t n1 = fraud_rows.size();
    const std::size_t kc = pp.selected.size();
    Matrix x_sel(n1, kc);
    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < kc; ++j) {
            x_sel(i, j) = X(fraud_rows[i], pp.selected[j]);
        }
    }

    // 3) Standardize feature-wise.
    pp.mu = column_mean(x_sel);
    pp.sigma = column_std(x_sel, pp.mu);
    Matrix x_std(n1, kc);
    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < kc; ++j) {
            double s = pp.sigma[j] > eps ? pp.sigma[j] : 1.0;
            x_std(i, j) = (x_sel(i, j) - pp.mu[j]) / s;
        }
    }

    // 4) PCA: project onto the leading n_components eigenvectors.
    Matrix cov = covariance(x_std);
    Vec evals;
    Matrix evecs;
    jacobi_eigen(cov, evals, evecs);
    pp.components = Matrix(kc, n_components);
    for (std::size_t r = 0; r < kc; ++r) {
        for (int c = 0; c < n_components; ++c) {
            pp.components(r, c) = evecs(r, c);
        }
    }
    Matrix x_pca(n1, n_components, 0.0);
    for (std::size_t i = 0; i < n1; ++i) {
        for (int c = 0; c < n_components; ++c) {
            double acc = 0.0;
            for (std::size_t r = 0; r < kc; ++r) {
                acc += x_std(i, r) * pp.components(r, c);
            }
            x_pca(i, c) = acc;
        }
    }

    // 5) Per-component max-abs normalization into [-1, 1].
    pp.max_abs.assign(n_components, eps);
    for (int c = 0; c < n_components; ++c) {
        double m = eps;
        for (std::size_t i = 0; i < n1; ++i) {
            m = std::max(m, std::fabs(x_pca(i, c)));
        }
        pp.max_abs[c] = m;
    }
    pp.x_norm = Matrix(n1, n_components);
    for (std::size_t i = 0; i < n1; ++i) {
        for (int c = 0; c < n_components; ++c) {
            pp.x_norm(i, c) = x_pca(i, c) / pp.max_abs[c];
        }
    }
    return pp;
}

// Forward-transform an original full-feature row into the bounded [-1,1]
// representation space using the fitted pipeline (used for non-fraud rows in
// downstream evaluation).
inline Vec transform_to_norm(const Preprocessor &pp, const Matrix &X,
                             std::size_t row)
{
    const std::size_t kc = pp.selected.size();
    Vec xs(kc);
    for (std::size_t j = 0; j < kc; ++j) {
        double s = pp.sigma[j] > pp.eps ? pp.sigma[j] : 1.0;
        xs[j] = (X(row, pp.selected[j]) - pp.mu[j]) / s;
    }
    Vec out(pp.n_components);
    for (int c = 0; c < pp.n_components; ++c) {
        double acc = 0.0;
        for (std::size_t r = 0; r < kc; ++r) {
            acc += xs[r] * pp.components(r, c);
        }
        out[c] = std::max(-1.0, std::min(1.0, acc / pp.max_abs[c]));
    }
    return out;
}

// Map a bounded sample x_norm in [-1,1]^n back to the selected feature space:
// undo normalization, inverse-PCA, then de-standardize.
inline Vec inverse_transform(const Preprocessor &pp, const Vec &x_norm)
{
    const int n_comp = pp.n_components;
    const std::size_t kc = pp.selected.size();
    Vec pca(n_comp);
    for (int c = 0; c < n_comp; ++c) {
        pca[c] = x_norm[c] * pp.max_abs[c];
    }
    // Inverse PCA: x_std = W * pca (W has orthonormal columns).
    Vec x_std(kc, 0.0);
    for (std::size_t r = 0; r < kc; ++r) {
        double acc = 0.0;
        for (int c = 0; c < n_comp; ++c) {
            acc += pp.components(r, c) * pca[c];
        }
        x_std[r] = acc;
    }
    Vec feat(kc);
    for (std::size_t j = 0; j < kc; ++j) {
        feat[j] = x_std[j] * pp.sigma[j] + pp.mu[j];
    }
    return feat;
}

} // namespace qsynth
