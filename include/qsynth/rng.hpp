// Centralized random number generation for reproducible experiments.
#pragma once

#include <random>
#include <vector>

namespace qsynth {

class Rng {
public:
    explicit Rng(unsigned int seed) : gen_(seed) {}

    double uniform(double lo = 0.0, double hi = 1.0)
    {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(gen_);
    }

    double normal(double mean = 0.0, double stddev = 1.0)
    {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(gen_);
    }

    int randint(int lo, int hi) // inclusive lo, exclusive hi
    {
        std::uniform_int_distribution<int> dist(lo, hi - 1);
        return dist(gen_);
    }

    // Draw a length-n vector of i.i.d. standard normal samples.
    std::vector<double> normal_vec(std::size_t n, double stddev = 1.0)
    {
        std::vector<double> v(n);
        for (auto &x : v) {
            x = normal(0.0, stddev);
        }
        return v;
    }

    std::mt19937 &engine() { return gen_; }

private:
    std::mt19937 gen_;
};

} // namespace qsynth
