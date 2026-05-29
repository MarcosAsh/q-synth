// Classical discriminator D: R^d -> (0,1)  (Sec. III-B-2).
//   u1 = W1 x + b1 ; v1 = LeakyReLU(u1)
//   u2 = W2 v1 + b2 ; h  = LeakyReLU(u2)   <- feature map f(x)
//   (dropout on h during training)
//   s  = w . h~ + b ; D = sigmoid(s)
#pragma once

#include "linalg.hpp"
#include "optim.hpp"
#include "rng.hpp"

#include <cmath>
#include <vector>

namespace qsynth {

class Discriminator {
public:
    Discriminator(int in_dim, int k1, int k2, double leaky_alpha, double dropout,
                  Rng &rng, double lr)
        : d_(in_dim), k1_(k1), k2_(k2), alpha_(leaky_alpha), drop_(dropout),
          rng_(&rng), W1_(k1, in_dim), b1_(k1, 0.0), W2_(k2, k1), b2_(k2, 0.0),
          w_(k2, 0.0), b_(0.0)
    {
        auto init = [&](Matrix &m) {
            double scale = std::sqrt(2.0 / static_cast<double>(m.cols));
            for (auto &x : m.data) {
                x = rng.normal(0.0, scale);
            }
        };
        init(W1_);
        init(W2_);
        double ws = std::sqrt(2.0 / static_cast<double>(k2));
        for (auto &x : w_) {
            x = rng.normal(0.0, ws);
        }
        adam_W1_.init(W1_.data.size());
        adam_b1_.init(b1_.size());
        adam_W2_.init(W2_.data.size());
        adam_b2_.init(b2_.size());
        adam_w_.init(w_.size());
        adam_b_.init(1);
        for (Adam *a : {&adam_W1_, &adam_b1_, &adam_W2_, &adam_b2_, &adam_w_,
                        &adam_b_}) {
            a->lr = lr;
            a->beta1 = 0.5;
            a->beta2 = 0.9;
        }
        zero_grads();
    }

    void set_dropout(double p) { drop_ = p; }

    struct Cache {
        Vec x;
        Vec u1, v1; // first layer pre/post activation
        Vec u2, h;  // second layer pre/post activation == feature f(x)
        Vec mask;   // dropout mask applied to h (empty when disabled)
        Vec hd;     // h after (inverted) dropout
        double logit = 0.0;
        double prob = 0.0;
    };

    double leaky(double t) const { return t > 0.0 ? t : alpha_ * t; }
    double dleaky(double t) const { return t > 0.0 ? 1.0 : alpha_; }

    // Forward pass. When `train` is true, inverted dropout is applied to h.
    double forward(const Vec &x, Cache &c, bool train) const
    {
        c.x = x;
        c.u1.assign(k1_, 0.0);
        c.v1.assign(k1_, 0.0);
        for (int i = 0; i < k1_; ++i) {
            double acc = b1_[i];
            for (int j = 0; j < d_; ++j) {
                acc += W1_(i, j) * x[j];
            }
            c.u1[i] = acc;
            c.v1[i] = leaky(acc);
        }
        c.u2.assign(k2_, 0.0);
        c.h.assign(k2_, 0.0);
        for (int i = 0; i < k2_; ++i) {
            double acc = b2_[i];
            for (int j = 0; j < k1_; ++j) {
                acc += W2_(i, j) * c.v1[j];
            }
            c.u2[i] = acc;
            c.h[i] = leaky(acc);
        }
        c.hd = c.h;
        c.mask.clear();
        if (train && drop_ > 0.0) {
            c.mask.assign(k2_, 1.0);
            for (int i = 0; i < k2_; ++i) {
                double keep = rng_->uniform() >= drop_ ? 1.0 : 0.0;
                c.mask[i] = keep / (1.0 - drop_);
                c.hd[i] = c.h[i] * c.mask[i];
            }
        }
        double s = b_;
        for (int i = 0; i < k2_; ++i) {
            s += w_[i] * c.hd[i];
        }
        c.logit = s;
        c.prob = 1.0 / (1.0 + std::exp(-s));
        return c.prob;
    }

    // Accumulate parameter gradients for a BCE term with the given target
    // label. dL/ds = D - y for binary cross-entropy with a sigmoid output.
    void accumulate_param_grad(const Cache &c, double target)
    {
        double ds = c.prob - target;
        gb_ += ds;
        for (int i = 0; i < k2_; ++i) {
            gw_[i] += ds * c.hd[i];
        }
        // back through dropout into h
        Vec dh(k2_, 0.0);
        for (int i = 0; i < k2_; ++i) {
            double m = c.mask.empty() ? 1.0 : c.mask[i];
            dh[i] = ds * w_[i] * m * dleaky(c.u2[i]);
        }
        for (int i = 0; i < k2_; ++i) {
            gb2_[i] += dh[i];
            for (int j = 0; j < k1_; ++j) {
                gW2_(i, j) += dh[i] * c.v1[j];
            }
        }
        Vec dv1(k1_, 0.0);
        for (int j = 0; j < k1_; ++j) {
            double acc = 0.0;
            for (int i = 0; i < k2_; ++i) {
                acc += W2_(i, j) * dh[i];
            }
            dv1[j] = acc * dleaky(c.u1[j]);
        }
        for (int i = 0; i < k1_; ++i) {
            gb1_[i] += dv1[i];
            for (int j = 0; j < d_; ++j) {
                gW1_(i, j) += dv1[i] * c.x[j];
            }
        }
    }

    // dL/dx for a BCE term D(x) vs target (parameters held fixed). Used to feed
    // the adversarial signal back into the generator.
    Vec input_grad_bce(const Cache &c, double target) const
    {
        double ds = c.prob - target;
        Vec dh(k2_, 0.0);
        for (int i = 0; i < k2_; ++i) {
            double m = c.mask.empty() ? 1.0 : c.mask[i];
            dh[i] = ds * w_[i] * m * dleaky(c.u2[i]);
        }
        return backprop_to_input(c, dh);
    }

    // dL/dx given a seed on the feature vector h (for feature matching). Uses
    // the deterministic (no-dropout) activations.
    Vec input_grad_feature(const Cache &c, const Vec &seed_h) const
    {
        Vec dh(k2_, 0.0);
        for (int i = 0; i < k2_; ++i) {
            dh[i] = seed_h[i] * dleaky(c.u2[i]);
        }
        return backprop_to_input(c, dh);
    }

    void zero_grads()
    {
        gW1_ = Matrix(k1_, d_, 0.0);
        gb1_.assign(k1_, 0.0);
        gW2_ = Matrix(k2_, k1_, 0.0);
        gb2_.assign(k2_, 0.0);
        gw_.assign(k2_, 0.0);
        gb_ = 0.0;
    }

    void apply_grads(double scale)
    {
        for (auto *g : {&gW1_.data, &gb1_, &gW2_.data, &gb2_, &gw_}) {
            for (auto &v : *g) {
                v *= scale;
            }
        }
        gb_ *= scale;
        adam_W1_.step(W1_.data, gW1_.data);
        adam_b1_.step(b1_, gb1_);
        adam_W2_.step(W2_.data, gW2_.data);
        adam_b2_.step(b2_, gb2_);
        adam_w_.step(w_, gw_);
        Vec bb{b_}, gg{gb_};
        adam_b_.step(bb, gg);
        b_ = bb[0];
        zero_grads();
    }

    int feature_dim() const { return k2_; }

private:
    // Shared backprop from a seed on the second pre-activation u2 down to x.
    Vec backprop_to_input(const Cache &c, const Vec &dh) const
    {
        Vec dv1(k1_, 0.0);
        for (int j = 0; j < k1_; ++j) {
            double acc = 0.0;
            for (int i = 0; i < k2_; ++i) {
                acc += W2_(i, j) * dh[i];
            }
            dv1[j] = acc * dleaky(c.u1[j]);
        }
        Vec dx(d_, 0.0);
        for (int j = 0; j < d_; ++j) {
            double acc = 0.0;
            for (int i = 0; i < k1_; ++i) {
                acc += W1_(i, j) * dv1[i];
            }
            dx[j] = acc;
        }
        return dx;
    }

    int d_, k1_, k2_;
    double alpha_, drop_;
    Rng *rng_;
    Matrix W1_;
    Vec b1_;
    Matrix W2_;
    Vec b2_;
    Vec w_;
    double b_;

    Matrix gW1_;
    Vec gb1_;
    Matrix gW2_;
    Vec gb2_;
    Vec gw_;
    double gb_ = 0.0;

    Adam adam_W1_, adam_b1_, adam_W2_, adam_b2_, adam_w_, adam_b_;
};

} // namespace qsynth
