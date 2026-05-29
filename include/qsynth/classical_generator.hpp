// Classical-GAN generator baseline: a 2-layer tanh MLP mapping latent noise to
// a bounded sample in [-1,1]^d. Capacity is matched to the hybrid generator by
// the caller's choice of hidden size. Shares the IGenerator interface so the
// same adversarial trainer drives both.
#pragma once

#include "generator_base.hpp"
#include "linalg.hpp"
#include "optim.hpp"
#include "rng.hpp"

#include <cmath>
#include <vector>

namespace qsynth {

class ClassicalGenerator : public IGenerator {
public:
    ClassicalGenerator(int latent_dim, int hidden, int out_dim, Rng &rng,
                       double lr)
        : m_(latent_dim), r_(hidden), d_(out_dim), W1_(hidden, latent_dim),
          b1_(hidden, 0.0), W2_(out_dim, hidden), b2_(out_dim, 0.0)
    {
        auto init = [&](Matrix &w) {
            double s = std::sqrt(2.0 / static_cast<double>(w.cols));
            for (auto &x : w.data) {
                x = rng.normal(0.0, s);
            }
        };
        init(W1_);
        init(W2_);
        adam_W1_.init(W1_.data.size());
        adam_b1_.init(b1_.size());
        adam_W2_.init(W2_.data.size());
        adam_b2_.init(b2_.size());
        for (Adam *a : {&adam_W1_, &adam_b1_, &adam_W2_, &adam_b2_}) {
            a->lr = lr;
            a->beta1 = 0.5;
            a->beta2 = 0.9;
        }
        zero_grads();
    }

    int latent_dim() const override { return m_; }

    struct Cache {
        Vec z, h, x;
    };

    Vec forward(const Vec &z, Cache &c) const
    {
        c.z = z;
        c.h.assign(r_, 0.0);
        for (int i = 0; i < r_; ++i) {
            double acc = b1_[i];
            for (int j = 0; j < m_; ++j) {
                acc += W1_(i, j) * z[j];
            }
            c.h[i] = std::tanh(acc);
        }
        c.x.assign(d_, 0.0);
        for (int i = 0; i < d_; ++i) {
            double acc = b2_[i];
            for (int j = 0; j < r_; ++j) {
                acc += W2_(i, j) * c.h[j];
            }
            c.x[i] = std::tanh(acc);
        }
        return c.x;
    }

    void begin_batch() override { caches_.clear(); }

    Vec forward_cached(const Vec &z) override
    {
        caches_.emplace_back();
        return forward(z, caches_.back());
    }

    Vec sample(const Vec &z) override
    {
        Cache c;
        return forward(z, c);
    }

    void backward(int idx, const Vec &grad_x) override
    {
        const Cache &c = caches_[idx];
        // out = tanh(W2 h + b2)
        Vec dout(d_, 0.0);
        for (int i = 0; i < d_; ++i) {
            dout[i] = grad_x[i] * (1.0 - c.x[i] * c.x[i]);
        }
        for (int i = 0; i < d_; ++i) {
            gb2_[i] += dout[i];
            for (int j = 0; j < r_; ++j) {
                gW2_(i, j) += dout[i] * c.h[j];
            }
        }
        Vec dh(r_, 0.0);
        for (int j = 0; j < r_; ++j) {
            double acc = 0.0;
            for (int i = 0; i < d_; ++i) {
                acc += W2_(i, j) * dout[i];
            }
            dh[j] = acc * (1.0 - c.h[j] * c.h[j]);
        }
        for (int i = 0; i < r_; ++i) {
            gb1_[i] += dh[i];
            for (int j = 0; j < m_; ++j) {
                gW1_(i, j) += dh[i] * c.z[j];
            }
        }
    }

    void apply_grads(double scale, double clip_norm) override
    {
        for (auto *g : {&gW1_.data, &gb1_, &gW2_.data, &gb2_}) {
            for (auto &v : *g) {
                v *= scale;
            }
        }
        double sq = 0.0;
        for (auto *g : {&gW1_.data, &gb1_, &gW2_.data, &gb2_}) {
            for (double v : *g) {
                sq += v * v;
            }
        }
        double norm = std::sqrt(sq);
        if (clip_norm > 0.0 && norm > clip_norm) {
            double f = clip_norm / norm;
            for (auto *g : {&gW1_.data, &gb1_, &gW2_.data, &gb2_}) {
                for (auto &v : *g) {
                    v *= f;
                }
            }
        }
        adam_W1_.step(W1_.data, gW1_.data);
        adam_b1_.step(b1_, gb1_);
        adam_W2_.step(W2_.data, gW2_.data);
        adam_b2_.step(b2_, gb2_);
        zero_grads();
    }

private:
    void zero_grads()
    {
        gW1_ = Matrix(r_, m_, 0.0);
        gb1_.assign(r_, 0.0);
        gW2_ = Matrix(d_, r_, 0.0);
        gb2_.assign(d_, 0.0);
    }

    int m_, r_, d_;
    Matrix W1_;
    Vec b1_;
    Matrix W2_;
    Vec b2_;
    Matrix gW1_;
    Vec gb1_;
    Matrix gW2_;
    Vec gb2_;
    Adam adam_W1_, adam_b1_, adam_W2_, adam_b2_;
    std::vector<Cache> caches_;
};

} // namespace qsynth
