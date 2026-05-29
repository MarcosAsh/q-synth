// Joint evaluation protocol (Sec. III-D): per-dimension distributional
// similarity (KS, Wasserstein) and external real-vs-synthetic detectability
// (AUC of an independent logistic-regression detector).
#pragma once

#include "classifiers.hpp"
#include "linalg.hpp"
#include "metrics.hpp"
#include "rng.hpp"

#include <algorithm>
#include <vector>

namespace qsynth {

struct FidelityReport {
    double ks_median;
    double ks_pvalue;   // median p-value across dimensions
    double wass_median;
    double wass_p75;
    double auc;         // external detectability, closer to 0.5 is better
};

inline FidelityReport evaluate_fidelity(const Matrix &real, const Matrix &synth,
                                        Rng &rng)
{
    const std::size_t d = real.cols;
    Vec ks(d), pv(d), ws(d);
    for (std::size_t j = 0; j < d; ++j) {
        Vec a = real.col(j);
        Vec b = synth.col(j);
        KsResult r = ks_two_sample(a, b);
        ks[j] = r.statistic;
        pv[j] = r.pvalue;
        ws[j] = wasserstein1(a, b);
    }
    FidelityReport rep;
    rep.ks_median = median(ks);
    rep.ks_pvalue = median(pv);
    rep.wass_median = median(ws);
    rep.wass_p75 = percentile(ws, 75.0);

    // External detector: label real=1, synthetic=0; split train/test; AUC.
    std::size_t n = std::min(real.rows, synth.rows);
    std::vector<Vec> Xtr, Xte;
    std::vector<int> ytr, yte;
    for (std::size_t i = 0; i < n; ++i) {
        bool test = rng.uniform() < 0.3;
        (test ? Xte : Xtr).push_back(real.row(i));
        (test ? yte : ytr).push_back(1);
        (test ? Xte : Xtr).push_back(synth.row(i));
        (test ? yte : ytr).push_back(0);
    }
    if (Xtr.size() < 4 || Xte.size() < 2) {
        rep.auc = 0.5;
        return rep;
    }
    LogisticRegression det(static_cast<int>(d), 0.1, 800, 1e-3);
    det.fit(Xtr, ytr);
    Vec probs = det.predict(Xte);
    rep.auc = auc_roc(probs, yte);
    return rep;
}

} // namespace qsynth
