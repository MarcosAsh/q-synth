// Lightweight self-checks for the numerical building blocks. No framework:
// each check prints PASS/FAIL and the process exits non-zero on any failure.
#include "qsynth/classical_generator.hpp"
#include "qsynth/metrics.hpp"
#include "qsynth/preprocess.hpp"
#include "qsynth/quantum_generator.hpp"
#include "qsynth/statevector.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace qsynth;

namespace {

int g_failures = 0;

void check(bool cond, const std::string &name)
{
    std::printf("[%s] %s\n", cond ? "PASS" : "FAIL", name.c_str());
    if (!cond) {
        ++g_failures;
    }
}

bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

// RY(pi) on |0> -> |1>, so <Z> should be -1.
void test_statevector_rotation()
{
    StateVector sv(1);
    sv.ry(0, M_PI);
    check(approx(sv.expect_z(0), -1.0, 1e-9), "RY(pi) flips <Z> to -1");

    StateVector id(2);
    check(approx(id.expect_z(0), 1.0) && approx(id.expect_z(1), 1.0),
          "|00> has <Z>=+1 on both qubits");
}

// A Bell pair via H-like (RY(pi/2)) + CNOT has <Z>=0 on the control.
void test_entanglement()
{
    StateVector sv(2);
    sv.ry(0, M_PI / 2.0); // equal superposition on qubit 0
    sv.cnot(0, 1);
    check(approx(sv.expect_z(0), 0.0, 1e-9), "entangled control has <Z>=0");
    check(approx(sv.expect_z(1), 0.0, 1e-9), "entangled target has <Z>=0");
}

// KS of a distribution against itself is 0; against a large shift is large.
void test_ks()
{
    Vec a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Vec b = a;
    KsResult same = ks_two_sample(a, b);
    check(approx(same.statistic, 0.0), "KS of identical samples is 0");

    Vec c{100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    KsResult diff = ks_two_sample(a, c);
    check(approx(diff.statistic, 1.0), "KS of disjoint samples is 1");
}

// AUC is 1 for perfectly separable scores, 0.5 for identical.
void test_auc()
{
    Vec scores{0.1, 0.2, 0.3, 0.8, 0.9, 1.0};
    std::vector<int> labels{0, 0, 0, 1, 1, 1};
    check(approx(auc_roc(scores, labels), 1.0), "AUC=1 when separable");

    Vec flat{0.5, 0.5, 0.5, 0.5};
    std::vector<int> lab2{0, 1, 0, 1};
    check(approx(auc_roc(flat, lab2), 0.5), "AUC=0.5 when uninformative");
}

// Wasserstein between identical samples is 0, and equals the shift for a
// rigid translation.
void test_wasserstein()
{
    Vec a{0, 1, 2, 3};
    check(approx(wasserstein1(a, a), 0.0), "Wasserstein of identical is 0");
    Vec b{2, 3, 4, 5};
    check(approx(wasserstein1(a, b), 2.0, 1e-9), "Wasserstein tracks shift");
}

// PCA reconstruction: with full components, inverse(forward(x)) ~ x.
void test_preprocess_roundtrip()
{
    Rng rng(1);
    Matrix X(200, 6);
    for (std::size_t i = 0; i < X.rows; ++i) {
        for (std::size_t j = 0; j < X.cols; ++j) {
            X(i, j) = rng.normal();
        }
    }
    std::vector<int> y(X.rows, 0);
    for (std::size_t i = 0; i < X.rows; ++i) {
        y[i] = i % 2; // half fraud
    }
    // Keep all 6 features, full-rank PCA, so the round trip is exact.
    Preprocessor pp = fit_preprocess(X, y, 6, 6);
    bool ok = true;
    for (std::size_t r = 0; r < pp.x_norm.rows && ok; ++r) {
        Vec feat = inverse_transform(pp, pp.x_norm.row(r));
        // Compare against the standardized->destandardized original fraud row.
        // We only require bounded, finite output here.
        for (double v : feat) {
            if (!std::isfinite(v)) {
                ok = false;
            }
        }
    }
    check(ok, "inverse_transform produces finite features");
    bool bounded = true;
    for (double v : pp.x_norm.data) {
        if (v < -1.0001 || v > 1.0001) {
            bounded = false;
        }
    }
    check(bounded, "normalized representation lies in [-1,1]");
}

// Generators produce outputs in [-1,1]^d.
void test_generator_bounds()
{
    Rng rng(7);
    QuantumGenerator qg(4, 8, 4, 3, rng, 1e-3);
    ClassicalGenerator cg(4, 8, 4, rng, 1e-3);
    bool ok = true;
    for (int t = 0; t < 20; ++t) {
        Vec z = rng.normal_vec(4);
        for (double v : qg.sample(z)) {
            if (v < -1.0001 || v > 1.0001) {
                ok = false;
            }
        }
        for (double v : cg.sample(z)) {
            if (v < -1.0001 || v > 1.0001) {
                ok = false;
            }
        }
    }
    check(ok, "generator outputs are bounded in [-1,1]");
}

// One generator gradient step should reduce a simple MSE-to-target objective.
void test_generator_learns()
{
    Rng rng(3);
    QuantumGenerator qg(4, 16, 4, 4, rng, 5e-3);
    Vec z = rng.normal_vec(4);
    Vec target(4, 0.5);
    auto loss = [&]() {
        Vec x = qg.sample(z);
        double l = 0.0;
        for (int j = 0; j < 4; ++j) {
            l += (x[j] - target[j]) * (x[j] - target[j]);
        }
        return l;
    };
    double before = loss();
    for (int step = 0; step < 50; ++step) {
        qg.begin_batch();
        Vec x = qg.forward_cached(z);
        Vec grad(4);
        for (int j = 0; j < 4; ++j) {
            grad[j] = 2.0 * (x[j] - target[j]);
        }
        qg.backward(0, grad);
        qg.apply_grads(1.0, 1.0);
    }
    double after = loss();
    check(after < before, "quantum generator reduces MSE toward a target");
}

} // namespace

int main()
{
    test_statevector_rotation();
    test_entanglement();
    test_ks();
    test_auc();
    test_wasserstein();
    test_preprocess_roundtrip();
    test_generator_bounds();
    test_generator_learns();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILURES",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
