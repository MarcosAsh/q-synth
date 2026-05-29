// Minimal dense linear algebra: a row-major matrix and the handful of
// operations the pipeline needs. Kept dependency-free on purpose so the
// project builds with nothing but a C++17 compiler.
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qsynth {

using Vec = std::vector<double>;

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data; // row-major, size rows*cols

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill)
    {
    }

    double &operator()(std::size_t r, std::size_t c)
    {
        return data[r * cols + c];
    }
    double operator()(std::size_t r, std::size_t c) const
    {
        return data[r * cols + c];
    }

    // Extract a single column as a vector.
    Vec col(std::size_t c) const
    {
        Vec out(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            out[r] = (*this)(r, c);
        }
        return out;
    }

    // Extract a single row as a vector.
    Vec row(std::size_t r) const
    {
        Vec out(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            out[c] = (*this)(r, c);
        }
        return out;
    }
};

// Per-column mean over the rows of a matrix.
inline Vec column_mean(const Matrix &m)
{
    Vec mu(m.cols, 0.0);
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            mu[c] += m(r, c);
        }
    }
    for (auto &v : mu) {
        v /= static_cast<double>(m.rows);
    }
    return mu;
}

// Per-column population standard deviation.
inline Vec column_std(const Matrix &m, const Vec &mu)
{
    Vec sd(m.cols, 0.0);
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            double d = m(r, c) - mu[c];
            sd[c] += d * d;
        }
    }
    for (auto &v : sd) {
        v = std::sqrt(v / static_cast<double>(m.rows));
    }
    return sd;
}

// Symmetric matrix-vector helpers used by the PCA eigensolver.
inline Matrix covariance(const Matrix &standardized)
{
    // Assumes columns are already zero-mean (standardized input).
    const std::size_t d = standardized.cols;
    const std::size_t n = standardized.rows;
    Matrix cov(d, d, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < d; ++i) {
            double xi = standardized(r, i);
            for (std::size_t j = i; j < d; ++j) {
                cov(i, j) += xi * standardized(r, j);
            }
        }
    }
    double denom = static_cast<double>(n > 1 ? n - 1 : 1);
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = i; j < d; ++j) {
            double v = cov(i, j) / denom;
            cov(i, j) = v;
            cov(j, i) = v;
        }
    }
    return cov;
}

} // namespace qsynth
