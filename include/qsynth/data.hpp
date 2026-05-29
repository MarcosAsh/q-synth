// Dataset loading. Reads the Kaggle Credit Card Fraud CSV when available
// (header row, comma-separated, last column = Class in {0,1}); otherwise
// synthesizes an extremely imbalanced tabular dataset with a heterogeneous,
// multi-modal minority class so the full pipeline can run end to end offline.
#pragma once

#include "linalg.hpp"
#include "rng.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace qsynth {

struct Dataset {
    Matrix X;
    std::vector<int> y;
    long n_fraud = 0;
    long n_legit = 0;
    bool synthetic = false;
};

inline bool load_csv(const std::string &path, Dataset &ds)
{
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    std::getline(in, line); // header
    std::vector<std::vector<double>> rows;
    std::vector<int> labels;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> vals;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty() && cell.front() == '"') {
                cell = cell.substr(1, cell.size() - 2);
            }
            vals.push_back(cell.empty() ? 0.0 : std::stod(cell));
        }
        if (vals.size() < 2) {
            continue;
        }
        int label = static_cast<int>(vals.back() + 0.5);
        vals.pop_back();
        rows.push_back(std::move(vals));
        labels.push_back(label);
    }
    if (rows.empty()) {
        return false;
    }
    const std::size_t n = rows.size();
    const std::size_t d = rows[0].size();
    ds.X = Matrix(n, d);
    ds.y = labels;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d && j < rows[i].size(); ++j) {
            ds.X(i, j) = rows[i][j];
        }
    }
    ds.n_fraud = 0;
    for (int l : labels) {
        ds.n_fraud += (l == 1);
    }
    ds.n_legit = static_cast<long>(n) - ds.n_fraud;
    ds.synthetic = false;
    return true;
}

// Build a synthetic imbalanced dataset resembling the fraud setting: ~0.5%
// minority rate, 28 features, legit class one broad gaussian, fraud class a
// mixture of a few shifted, anisotropic gaussians.
inline Dataset make_synthetic(Rng &rng, int n_total = 30000, int dim = 28,
                              double fraud_rate = 0.0166)
{
    Dataset ds;
    ds.synthetic = true;
    int n_fraud = std::max(50, static_cast<int>(n_total * fraud_rate));
    int n_legit = n_total - n_fraud;
    ds.X = Matrix(n_total, dim);
    ds.y.assign(n_total, 0);

    // Legit cluster: centered near origin, moderate spread.
    for (int i = 0; i < n_legit; ++i) {
        for (int j = 0; j < dim; ++j) {
            ds.X(i, j) = rng.normal(0.0, 1.0);
        }
        ds.y[i] = 0;
    }

    // Fraud: mixture of 3 modes with distinct means/scales on informative dims.
    const int n_modes = 3;
    std::vector<Vec> centers(n_modes, Vec(dim, 0.0));
    for (int mdl = 0; mdl < n_modes; ++mdl) {
        for (int j = 0; j < dim; ++j) {
            centers[mdl][j] = (j < 10) ? rng.normal(2.5, 0.7) : rng.normal(0.0, 0.4);
        }
    }
    for (int i = 0; i < n_fraud; ++i) {
        int mode = rng.randint(0, n_modes);
        int r = n_legit + i;
        for (int j = 0; j < dim; ++j) {
            double scale = (j < 10) ? 0.6 : 1.0;
            ds.X(r, j) = centers[mode][j] + rng.normal(0.0, scale);
        }
        ds.y[r] = 1;
    }
    ds.n_fraud = n_fraud;
    ds.n_legit = n_legit;
    return ds;
}

} // namespace qsynth
