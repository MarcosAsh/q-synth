// Evaluation metrics (Sec. III-D): two-sample Kolmogorov-Smirnov statistic and
// p-value, 1-Wasserstein distance, and AUC-ROC.
#pragma once

#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace qsynth {

// Asymptotic Kolmogorov distribution Q_KS(x) = P(sup|.| > x).
inline double kolmogorov_q(double x)
{
    if (x < 1e-12) {
        return 1.0;
    }
    double sum = 0.0;
    double sign = 1.0;
    for (int k = 1; k <= 100; ++k) {
        double term = sign * std::exp(-2.0 * k * k * x * x);
        sum += term;
        if (std::fabs(term) < 1e-12) {
            break;
        }
        sign = -sign;
    }
    double p = 2.0 * sum;
    return std::clamp(p, 0.0, 1.0);
}

struct KsResult {
    double statistic;
    double pvalue;
};

inline KsResult ks_two_sample(Vec a, Vec b)
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const std::size_t n1 = a.size(), n2 = b.size();
    std::size_t i = 0, j = 0;
    double d = 0.0;
    while (i < n1 && j < n2) {
        double x = std::min(a[i], b[j]);
        while (i < n1 && a[i] <= x) {
            ++i;
        }
        while (j < n2 && b[j] <= x) {
            ++j;
        }
        double cdf1 = static_cast<double>(i) / n1;
        double cdf2 = static_cast<double>(j) / n2;
        d = std::max(d, std::fabs(cdf1 - cdf2));
    }
    double en = std::sqrt(static_cast<double>(n1) * n2 / (n1 + n2));
    double p = kolmogorov_q((en + 0.12 + 0.11 / en) * d);
    return {d, p};
}

// 1-Wasserstein distance between two empirical 1D distributions.
inline double wasserstein1(Vec a, Vec b)
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    Vec all;
    all.reserve(a.size() + b.size());
    all.insert(all.end(), a.begin(), a.end());
    all.insert(all.end(), b.begin(), b.end());
    std::sort(all.begin(), all.end());

    auto cdf = [](const Vec &s, double x) {
        return static_cast<double>(
                   std::upper_bound(s.begin(), s.end(), x) - s.begin())
               / static_cast<double>(s.size());
    };
    double area = 0.0;
    for (std::size_t k = 0; k + 1 < all.size(); ++k) {
        double width = all[k + 1] - all[k];
        if (width <= 0.0) {
            continue;
        }
        area += width * std::fabs(cdf(a, all[k]) - cdf(b, all[k]));
    }
    return area;
}

// AUC-ROC via the Mann-Whitney U statistic (handles ties with average ranks).
// `scores` paired with binary `labels`; positive class == 1.
inline double auc_roc(const Vec &scores, const std::vector<int> &labels)
{
    const std::size_t n = scores.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
    Vec rank(n, 0.0);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j < n && scores[idx[j]] == scores[idx[i]]) {
            ++j;
        }
        double avg = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
        for (std::size_t k = i; k < j; ++k) {
            rank[idx[k]] = avg;
        }
        i = j;
    }
    double sum_pos = 0.0;
    long n_pos = 0, n_neg = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (labels[k] == 1) {
            sum_pos += rank[k];
            ++n_pos;
        } else {
            ++n_neg;
        }
    }
    if (n_pos == 0 || n_neg == 0) {
        return 0.5;
    }
    double u = sum_pos - n_pos * (n_pos + 1) / 2.0;
    return u / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

// Median of a vector (copy, partial sort).
inline double median(Vec v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double percentile(Vec v, double q) // q in [0,100]
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    double pos = q / 100.0 * (v.size() - 1);
    std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    double frac = pos - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

} // namespace qsynth
