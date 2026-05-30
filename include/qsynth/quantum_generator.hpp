// Hybrid quantum-classical generator (Sec. III-B-1).
//
//   z  -- classical MLP -->  angle matrix Theta (d x 3)
//   z  -- fixed projection -> embedding angles z' (d)
//   |0>^d -- RY(z') -- [ L x ( RX,RY,RZ(Theta) + ring CNOT ) ] -- <Z> --> x_hat
//
// The MLP is trained by analytic backprop; gradients of the circuit output
// w.r.t. the rotation angles are obtained by central finite differences on the
// exact statevector simulator (the shared-angle structure makes the
// parameter-shift bookkeeping awkward, and finite differences are exact enough
// on a noiseless simulator).
#pragma once

#include "generator_base.hpp"
#include "linalg.hpp"
#include "optim.hpp"
#include "rng.hpp"
#include "statevector.hpp"

#include <cmath>
#include <vector>

namespace qsynth {

class QuantumGenerator : public IGenerator {
public:
    QuantumGenerator(int latent_dim, int hidden, int n_qubits, int layers,
                     Rng &rng, double lr)
        : m_(latent_dim), r_(hidden), d_(n_qubits), L_(layers),
          W1_(hidden, latent_dim), b1_(hidden, 0.0),
          W2_(3 * n_qubits, hidden), b2_(3 * n_qubits, 0.0),
          P_(n_qubits, latent_dim), S_(n_qubits, 0.0)
    {
        auto kaiming = [&](Matrix &w) {
            double scale = std::sqrt(2.0 / static_cast<double>(w.cols));
            for (auto &x : w.data) {
                x = rng.normal(0.0, scale);
            }
        };
        kaiming(W1_);
        kaiming(W2_);
        // Fixed (non-trained) embedding projection and diagonal scaling.
        for (auto &x : P_.data) {
            x = rng.normal(0.0, 1.0 / std::sqrt(latent_dim));
        }
        for (int q = 0; q < d_; ++q) {
            S_[q] = M_PI; // map projected latent roughly into [-pi, pi]
        }

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
    int n_qubits() const { return d_; }

    // IGenerator batch interface.
    void begin_batch() override { caches_.clear(); }

    Vec forward_cached(const Vec &z) override
    {
        caches_.emplace_back();
        return forward(z, caches_.back());
    }

    void backward(int idx, const Vec &grad_x) override
    {
        backward(caches_[idx], grad_x);
    }

    Vec sample(const Vec &z) override
    {
        Cache c;
        return forward(z, c);
    }

    // Embedding angles z' = S * (P z) (fixed transform).
    Vec embed(const Vec &z) const
    {
        Vec zp(d_, 0.0);
        for (int q = 0; q < d_; ++q) {
            double acc = 0.0;
            for (int i = 0; i < m_; ++i) {
                acc += P_(q, i) * z[i];
            }
            zp[q] = std::tanh(S_[q] * acc); // bounded embedding angle
        }
        return zp;
    }

    // Run the variational circuit and read out <Z_q> for every qubit.
    Vec run_circuit(const Vec &zp, const Vec &theta_flat) const
    {
        StateVector sv(d_);
        for (int q = 0; q < d_; ++q) {
            sv.ry(q, zp[q]); // angle embedding
        }
        for (int l = 0; l < L_; ++l) {
            for (int q = 0; q < d_; ++q) {
                sv.rx(q, theta_flat[q * 3 + 0]);
                sv.ry(q, theta_flat[q * 3 + 1]);
                sv.rz(q, theta_flat[q * 3 + 2]);
            }
            for (int q = 0; q < d_; ++q) {
                sv.cnot(q, (q + 1) % d_); // ring entanglement
            }
        }
        Vec out(d_);
        for (int q = 0; q < d_; ++q) {
            out[q] = sv.expect_z(q);
        }
        return out;
    }

    // Cached forward pass for one latent vector. Stores intermediates so a
    // later backward() call can accumulate parameter gradients.
    struct Cache {
        Vec z;
        Vec h;          // tanh(W1 z + b1)
        Vec a;          // W2 h + b2  (== flattened Theta)
        Vec zp;         // embedding angles
        Vec x;          // generator output
    };

    Vec forward(const Vec &z, Cache &cache) const
    {
        cache.z = z;
        cache.h.assign(r_, 0.0);
        for (int i = 0; i < r_; ++i) {
            double acc = b1_[i];
            for (int j = 0; j < m_; ++j) {
                acc += W1_(i, j) * z[j];
            }
            cache.h[i] = std::tanh(acc);
        }
        cache.a.assign(3 * d_, 0.0);
        for (int i = 0; i < 3 * d_; ++i) {
            double acc = b2_[i];
            for (int j = 0; j < r_; ++j) {
                acc += W2_(i, j) * cache.h[j];
            }
            cache.a[i] = acc;
        }
        cache.zp = embed(z);
        cache.x = run_circuit(cache.zp, cache.a);
        return cache.x;
    }

    // Accumulate parameter gradients given dL/dx_hat (size d).
    void backward(const Cache &cache, const Vec &grad_x)
    {
        // Jacobian dx/da via central finite differences (d x 3d).
        const double fd = 1e-3;
        Vec da(3 * d_, 0.0);
        Vec theta = cache.a;
        for (int k = 0; k < 3 * d_; ++k) {
            double orig = theta[k];
            theta[k] = orig + fd;
            Vec xp = run_circuit(cache.zp, theta);
            theta[k] = orig - fd;
            Vec xm = run_circuit(cache.zp, theta);
            theta[k] = orig;
            double g = 0.0;
            for (int q = 0; q < d_; ++q) {
                g += grad_x[q] * (xp[q] - xm[q]) / (2.0 * fd);
            }
            da[k] = g;
        }
        // a = W2 h + b2.
        for (int i = 0; i < 3 * d_; ++i) {
            gb2_[i] += da[i];
            for (int j = 0; j < r_; ++j) {
                gW2_(i, j) += da[i] * cache.h[j];
            }
        }
        Vec dh(r_, 0.0);
        for (int j = 0; j < r_; ++j) {
            double acc = 0.0;
            for (int i = 0; i < 3 * d_; ++i) {
                acc += W2_(i, j) * da[i];
            }
            dh[j] = acc;
        }
        // h = tanh(W1 z + b1).
        for (int i = 0; i < r_; ++i) {
            double dpre = dh[i] * (1.0 - cache.h[i] * cache.h[i]);
            gb1_[i] += dpre;
            for (int j = 0; j < m_; ++j) {
                gW1_(i, j) += dpre * cache.z[j];
            }
        }
    }

    void zero_grads()
    {
        gW1_ = Matrix(r_, m_, 0.0);
        gb1_.assign(r_, 0.0);
        gW2_ = Matrix(3 * d_, r_, 0.0);
        gb2_.assign(3 * d_, 0.0);
    }

    // Scale accumulated grads (e.g. by 1/batch), clip global norm, then step.
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
    int m_, r_, d_, L_;
    Matrix W1_;
    Vec b1_;
    Matrix W2_;
    Vec b2_;
    Matrix P_;
    Vec S_;

    Matrix gW1_;
    Vec gb1_;
    Matrix gW2_;
    Vec gb2_;

    Adam adam_W1_, adam_b1_, adam_W2_, adam_b2_;

    std::vector<Cache> caches_; // per-sample caches for the current mini-batch
};

} // namespace qsynth
