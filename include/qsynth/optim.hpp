// Adam optimizer operating on flat parameter buffers.
#pragma once

#include "linalg.hpp"

#include <cmath>

namespace qsynth {

struct Adam {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    long t = 0;
    Vec m;
    Vec v;

    Adam() = default;
    Adam(double learning_rate, double b1, double b2, double e)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(e)
    {
    }

    void init(std::size_t n)
    {
        m.assign(n, 0.0);
        v.assign(n, 0.0);
        t = 0;
    }

    // In-place update p <- p - lr * mhat / (sqrt(vhat) + eps).
    void step(Vec &p, const Vec &g)
    {
        ++t;
        double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
        double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
        for (std::size_t i = 0; i < p.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            double mhat = m[i] / bc1;
            double vhat = v[i] / bc2;
            p[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

} // namespace qsynth
