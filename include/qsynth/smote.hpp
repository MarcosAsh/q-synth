// SMOTE oversampling baseline (Chawla et al., 2002) in the bounded
// representation space: interpolate between a fraud sample and one of its
// k nearest fraud neighbors.
#pragma once

#include "linalg.hpp"
#include "rng.hpp"

#include <algorithm>
#include <vector>

namespace qsynth {

inline Matrix smote_oversample(const Matrix &fraud, int n_synth, int k,
                               Rng &rng)
{
    const int N = static_cast<int>(fraud.rows);
    const int d = static_cast<int>(fraud.cols);
    k = std::min(k, N - 1);
    Matrix out(n_synth, d);
    for (int s = 0; s < n_synth; ++s) {
        int i = rng.randint(0, N);
        // k nearest neighbors of i (brute force over the small fraud set).
        std::vector<std::pair<double, int>> dist;
        dist.reserve(N);
        for (int j = 0; j < N; ++j) {
            if (j == i) {
                continue;
            }
            double sq = 0.0;
            for (int c = 0; c < d; ++c) {
                double dd = fraud(i, c) - fraud(j, c);
                sq += dd * dd;
            }
            dist.emplace_back(sq, j);
        }
        std::partial_sort(dist.begin(), dist.begin() + k, dist.end());
        int nb = dist[rng.randint(0, k)].second;
        double lam = rng.uniform();
        for (int c = 0; c < d; ++c) {
            double v = fraud(i, c) + lam * (fraud(nb, c) - fraud(i, c));
            out(s, c) = std::clamp(v, -1.0, 1.0);
        }
    }
    return out;
}

} // namespace qsynth
