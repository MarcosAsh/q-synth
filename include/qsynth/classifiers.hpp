// Downstream classifiers and the external real-vs-synthetic detector.
// Provides logistic regression and a small ReLU MLP, plus a metrics bundle
// (per-class precision/recall/F1, accuracy, AUC).
#pragma once

#include "linalg.hpp"
#include "metrics.hpp"
#include "optim.hpp"
#include "rng.hpp"

#include <cmath>
#include <vector>

namespace qsynth {

struct ClassMetrics {
    // Index 0 = non-fraud (y=0), index 1 = fraud (y=1).
    double prec[2] = {0, 0};
    double rec[2] = {0, 0};
    double f1[2] = {0, 0};
    double accuracy = 0.0;
    double auc = 0.0;
};

inline ClassMetrics score_predictions(const Vec &probs,
                                      const std::vector<int> &y,
                                      double thresh = 0.5)
{
    long tp = 0, tn = 0, fp = 0, fn = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        int pred = probs[i] >= thresh ? 1 : 0;
        if (y[i] == 1 && pred == 1) {
            ++tp;
        } else if (y[i] == 0 && pred == 0) {
            ++tn;
        } else if (y[i] == 0 && pred == 1) {
            ++fp;
        } else {
            ++fn;
        }
    }
    ClassMetrics m;
    auto safe = [](double a, double b) { return b > 0 ? a / b : 0.0; };
    // Fraud (positive) class.
    m.prec[1] = safe(tp, tp + fp);
    m.rec[1] = safe(tp, tp + fn);
    m.f1[1] = safe(2 * m.prec[1] * m.rec[1], m.prec[1] + m.rec[1]);
    // Non-fraud class.
    m.prec[0] = safe(tn, tn + fn);
    m.rec[0] = safe(tn, tn + fp);
    m.f1[0] = safe(2 * m.prec[0] * m.rec[0], m.prec[0] + m.rec[0]);
    m.accuracy = safe(tp + tn, tp + tn + fp + fn);
    m.auc = auc_roc(probs, y);
    return m;
}

// L2-regularized logistic regression trained by full-batch gradient descent.
class LogisticRegression {
public:
    LogisticRegression(int dim, double lr = 0.1, int iters = 5000,
                       double l2 = 1e-4)
        : w_(dim, 0.0), b_(0.0), lr_(lr), iters_(iters), l2_(l2)
    {
    }

    void fit(const std::vector<Vec> &X, const std::vector<int> &y)
    {
        const std::size_t n = X.size();
        const std::size_t d = w_.size();
        for (int it = 0; it < iters_; ++it) {
            Vec gw(d, 0.0);
            double gb = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double p = predict_one(X[i]);
                double err = p - y[i];
                for (std::size_t j = 0; j < d; ++j) {
                    gw[j] += err * X[i][j];
                }
                gb += err;
            }
            for (std::size_t j = 0; j < d; ++j) {
                w_[j] -= lr_ * (gw[j] / n + l2_ * w_[j]);
            }
            b_ -= lr_ * gb / n;
        }
    }

    double predict_one(const Vec &x) const
    {
        double s = b_;
        for (std::size_t j = 0; j < w_.size(); ++j) {
            s += w_[j] * x[j];
        }
        return 1.0 / (1.0 + std::exp(-s));
    }

    Vec predict(const std::vector<Vec> &X) const
    {
        Vec out(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            out[i] = predict_one(X[i]);
        }
        return out;
    }

private:
    Vec w_;
    double b_;
    double lr_;
    int iters_;
    double l2_;
};

// Small fully-connected ReLU classifier with a sigmoid output, Adam-trained.
class MlpClassifier {
public:
    MlpClassifier(int dim, int hidden, Rng &rng, double lr = 1e-3,
                  int epochs = 30, int batch = 64)
        : d_(dim), h_(hidden), epochs_(epochs), batch_(batch),
          W1_(hidden, dim), b1_(hidden, 0.0), W2_(1, hidden), b2_(0.0),
          rng_(&rng)
    {
        auto init = [&](Matrix &m) {
            double s = std::sqrt(2.0 / m.cols);
            for (auto &x : m.data) {
                x = rng.normal(0.0, s);
            }
        };
        init(W1_);
        init(W2_);
        aW1_.init(W1_.data.size());
        ab1_.init(b1_.size());
        aW2_.init(W2_.data.size());
        ab2_.init(1);
        for (Adam *a : {&aW1_, &ab1_, &aW2_, &ab2_}) {
            a->lr = lr;
        }
    }

    void fit(const std::vector<Vec> &X, const std::vector<int> &y)
    {
        std::vector<int> idx(X.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            idx[i] = static_cast<int>(i);
        }
        for (int ep = 0; ep < epochs_; ++ep) {
            std::shuffle(idx.begin(), idx.end(), rng_->engine());
            for (std::size_t s = 0; s < idx.size(); s += batch_) {
                std::size_t end = std::min(idx.size(), s + batch_);
                Matrix gW1(h_, d_, 0.0);
                Vec gb1(h_, 0.0);
                Matrix gW2(1, h_, 0.0);
                double gb2 = 0.0;
                int B = static_cast<int>(end - s);
                for (std::size_t t = s; t < end; ++t) {
                    const Vec &x = X[idx[t]];
                    Vec hpre(h_), hpost(h_);
                    for (int i = 0; i < h_; ++i) {
                        double acc = b1_[i];
                        for (int j = 0; j < d_; ++j) {
                            acc += W1_(i, j) * x[j];
                        }
                        hpre[i] = acc;
                        hpost[i] = acc > 0 ? acc : 0.0;
                    }
                    double slog = b2_;
                    for (int i = 0; i < h_; ++i) {
                        slog += W2_(0, i) * hpost[i];
                    }
                    double p = 1.0 / (1.0 + std::exp(-slog));
                    double ds = p - y[idx[t]];
                    gb2 += ds;
                    for (int i = 0; i < h_; ++i) {
                        gW2(0, i) += ds * hpost[i];
                    }
                    for (int i = 0; i < h_; ++i) {
                        double dh = ds * W2_(0, i) * (hpre[i] > 0 ? 1.0 : 0.0);
                        gb1[i] += dh;
                        for (int j = 0; j < d_; ++j) {
                            gW1(i, j) += dh * x[j];
                        }
                    }
                }
                double sc = 1.0 / B;
                for (auto &v : gW1.data) {
                    v *= sc;
                }
                for (auto &v : gb1) {
                    v *= sc;
                }
                for (auto &v : gW2.data) {
                    v *= sc;
                }
                gb2 *= sc;
                aW1_.step(W1_.data, gW1.data);
                ab1_.step(b1_, gb1);
                aW2_.step(W2_.data, gW2.data);
                Vec bb{b2_}, gg{gb2};
                ab2_.step(bb, gg);
                b2_ = bb[0];
            }
        }
    }

    double predict_one(const Vec &x) const
    {
        double slog = b2_;
        for (int i = 0; i < h_; ++i) {
            double acc = b1_[i];
            for (int j = 0; j < d_; ++j) {
                acc += W1_(i, j) * x[j];
            }
            double hp = acc > 0 ? acc : 0.0;
            slog += W2_(0, i) * hp;
        }
        return 1.0 / (1.0 + std::exp(-slog));
    }

    Vec predict(const std::vector<Vec> &X) const
    {
        Vec out(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            out[i] = predict_one(X[i]);
        }
        return out;
    }

private:
    int d_, h_, epochs_, batch_;
    Matrix W1_;
    Vec b1_;
    Matrix W2_;
    double b2_;
    Adam aW1_, ab1_, aW2_, ab2_;
    Rng *rng_;
};

} // namespace qsynth
