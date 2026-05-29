// Adversarial training with adaptive regularization (Sec. III-C).
//
// Implements: instance noise + projection onto [-1,1]^d, BCE discriminator
// objective, generator objective = adversarial (label-smoothed) + feature
// matching + moment matching, generator gradient-norm clipping, and adaptive
// adjustment of (noise sigma, label-smoothing gamma, dropout p) at checkpoints.
#pragma once

#include "discriminator.hpp"
#include "generator_base.hpp"
#include "linalg.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace qsynth {

struct QGanConfig {
    int latent_dim = 4;
    int batch_size = 64;
    int epochs = 100;
    double lr_g = 7e-4;
    double lr_d = 2e-4;

    double gamma = 0.88;            // label smoothing target
    double gamma_lo = 0.80, gamma_hi = 0.94;
    double dropout = 0.10;
    double dropout_lo = 0.10, dropout_hi = 0.16;
    double noise_base = 0.014;      // instance noise base scale
    double noise_end_bonus = 0.016; // extra noise ramped in by end of training
    double step_noise = 0.004;
    double step_gamma = 0.02;
    double step_dropout = 0.03;

    double fm_weight = 0.10;        // feature matching
    double mm_alpha = 0.05;         // moment matching (mean)
    double mm_beta = 0.03;          // moment matching (std)
    double mm_std_floor = 1e-6;
    double grad_clip = 1.0;
    int eval_every = 10;
    bool verbose = true;
};

struct TrainHistory {
    std::vector<double> g_loss;
    std::vector<double> d_loss;
};

inline double clip_unit(double v) { return std::clamp(v, -1.0, 1.0); }

inline int sign_d(double x) { return (x > 0.0) - (x < 0.0); }

// Forward declarations for helpers defined after the training routine.
inline Vec xb_row(const Matrix &m, int r, int d);
inline void gamma_step(QGanConfig &cfg, int dir);
inline void cfg_mut_dropout(Discriminator &disc, const QGanConfig &cfg, int dir);

// Train `gen` against a freshly-managed discriminator on bounded real data
// `real` (rows = samples, cols = d). The discriminator is created internally so
// both the quantum and classical generators get an identical training setup.
inline TrainHistory train_qgan(IGenerator &gen, const Matrix &real,
                               const QGanConfig &cfg, Rng &rng)
{
    const int d = static_cast<int>(real.cols);
    const int N = static_cast<int>(real.rows);
    const int B = std::min(cfg.batch_size, N);
    const int m = gen.latent_dim();

    Discriminator disc(d, 16, 8, 0.2, cfg.dropout, rng, cfg.lr_d);
    const int k2 = disc.feature_dim();

    TrainHistory hist;
    double sigma = cfg.noise_base;

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng.engine());
        double sigma_eff =
            sigma + cfg.noise_end_bonus * (static_cast<double>(epoch) /
                                           std::max(1, cfg.epochs - 1));
        sigma_eff = std::clamp(sigma_eff, 0.0, 0.08);

        double ep_g = 0.0, ep_d = 0.0;
        int steps = 0;
        for (int start = 0; start + B <= N; start += B) {
            // Gather a real mini-batch.
            std::vector<Vec> xb(B, Vec(d));
            for (int i = 0; i < B; ++i) {
                for (int j = 0; j < d; ++j) {
                    xb[i][j] = real(idx[start + i], j);
                }
            }

            // ---- Discriminator step ----------------------------------------
            disc.set_dropout(cfg.dropout);
            double d_loss = 0.0;
            for (int i = 0; i < B; ++i) {
                Vec noisy(d);
                for (int j = 0; j < d; ++j) {
                    noisy[j] = clip_unit(xb[i][j] + sigma_eff * rng.normal());
                }
                Discriminator::Cache c;
                double p = disc.forward(noisy, c, true);
                d_loss += -std::log(std::max(p, 1e-12));
                disc.accumulate_param_grad(c, 1.0);
            }
            std::vector<Vec> zb(B, Vec(m));
            std::vector<Vec> fake(B, Vec(d));
            for (int i = 0; i < B; ++i) {
                zb[i] = rng.normal_vec(m);
                fake[i] = gen.sample(zb[i]);
                Vec noisy(d);
                for (int j = 0; j < d; ++j) {
                    noisy[j] = clip_unit(fake[i][j] + sigma_eff * rng.normal());
                }
                Discriminator::Cache c;
                double p = disc.forward(noisy, c, true);
                d_loss += -std::log(std::max(1.0 - p, 1e-12));
                disc.accumulate_param_grad(c, 0.0);
            }
            disc.apply_grads(1.0 / B);
            ep_d += d_loss / (2.0 * B);

            // ---- Generator step --------------------------------------------
            gen.begin_batch();
            std::vector<Vec> xhat(B, Vec(d));
            for (int i = 0; i < B; ++i) {
                zb[i] = rng.normal_vec(m);
                xhat[i] = gen.forward_cached(zb[i]);
            }

            // Batch moments of real and generated samples.
            Vec mu_r(d, 0.0), mu_g(d, 0.0), s_r(d, 0.0), s_g(d, 0.0);
            for (int i = 0; i < B; ++i) {
                for (int j = 0; j < d; ++j) {
                    mu_r[j] += xb[i][j];
                    mu_g[j] += xhat[i][j];
                }
            }
            for (int j = 0; j < d; ++j) {
                mu_r[j] /= B;
                mu_g[j] /= B;
            }
            for (int i = 0; i < B; ++i) {
                for (int j = 0; j < d; ++j) {
                    s_r[j] += (xb[i][j] - mu_r[j]) * (xb[i][j] - mu_r[j]);
                    s_g[j] += (xhat[i][j] - mu_g[j]) * (xhat[i][j] - mu_g[j]);
                }
            }
            for (int j = 0; j < d; ++j) {
                s_r[j] = std::sqrt(s_r[j] / B + cfg.mm_std_floor);
                s_g[j] = std::sqrt(s_g[j] / B + cfg.mm_std_floor);
            }

            // Mean discriminator features (non-noisy, deterministic).
            disc.set_dropout(0.0);
            Vec fmean_r(k2, 0.0), fmean_g(k2, 0.0);
            std::vector<Discriminator::Cache> cg(B);
            for (int i = 0; i < B; ++i) {
                Discriminator::Cache cr;
                disc.forward(xb[i], cr, false);
                for (int t = 0; t < k2; ++t) {
                    fmean_r[t] += cr.h[t];
                }
                disc.forward(xhat[i], cg[i], false);
                for (int t = 0; t < k2; ++t) {
                    fmean_g[t] += cg[i].h[t];
                }
            }
            for (int t = 0; t < k2; ++t) {
                fmean_r[t] /= B;
                fmean_g[t] /= B;
            }
            Vec fm_seed(k2, 0.0);
            for (int t = 0; t < k2; ++t) {
                // d/dfmean_g of lambda*||fmean_r - fmean_g||_1, per-sample 1/B.
                fm_seed[t] = -cfg.fm_weight * sign_d(fmean_r[t] - fmean_g[t]) / B;
            }

            // Per-sample generator gradient dL/dxhat.
            double g_loss = 0.0;
            for (int i = 0; i < B; ++i) {
                Vec gx(d, 0.0);

                // Adversarial term (label-smoothed BCE on noisy input).
                Vec noisy(d);
                Vec cmask(d, 1.0);
                for (int j = 0; j < d; ++j) {
                    double raw = xhat[i][j] + sigma_eff * rng.normal();
                    noisy[j] = clip_unit(raw);
                    cmask[j] = (raw > -1.0 && raw < 1.0) ? 1.0 : 0.0;
                }
                Discriminator::Cache ca;
                double p = disc.forward(noisy, ca, false);
                g_loss += -std::log(std::max(p, 1e-12));
                Vec adv = disc.input_grad_bce(ca, cfg.gamma);
                for (int j = 0; j < d; ++j) {
                    gx[j] += adv[j] * cmask[j] / B;
                }

                // Feature matching term.
                Vec fmg = disc.input_grad_feature(cg[i], fm_seed);
                for (int j = 0; j < d; ++j) {
                    gx[j] += fmg[j];
                }

                // Moment matching term.
                for (int j = 0; j < d; ++j) {
                    double dmu = cfg.mm_alpha * (-sign_d(mu_r[j] - mu_g[j])) / B;
                    double dstd = -cfg.mm_beta * sign_d(s_r[j] - s_g[j])
                                  * (xhat[i][j] - mu_g[j]) / (B * s_g[j]);
                    gx[j] += dmu + dstd;
                }

                gen.backward(i, gx);
            }
            gen.apply_grads(1.0, cfg.grad_clip);
            ep_g += g_loss / B;
            ++steps;
        }

        hist.g_loss.push_back(ep_g / std::max(1, steps));
        hist.d_loss.push_back(ep_d / std::max(1, steps));

        // ---- Adaptive regularization at checkpoints ------------------------
        if ((epoch + 1) % cfg.eval_every == 0) {
            disc.set_dropout(0.0);
            double dr = 0.0, df = 0.0;
            int probe = std::min(N, 256);
            for (int i = 0; i < probe; ++i) {
                Discriminator::Cache cr, cf;
                dr += disc.forward(xb_row(real, idx[i % N], d), cr, false);
                df += disc.forward(gen.sample(rng.normal_vec(m)), cf, false);
            }
            dr /= probe;
            df /= probe;
            double ease = dr - df; // large => discrimination too easy
            if (ease > 0.6) {
                sigma = std::clamp(sigma + cfg.step_noise, 0.0, 0.06);
                cfg_mut_dropout(disc, cfg, +1);
                gamma_step(const_cast<QGanConfig &>(cfg), -1);
            } else if (ease < 0.2) {
                sigma = std::clamp(sigma - cfg.step_noise, 0.0, 0.06);
                cfg_mut_dropout(disc, cfg, -1);
                gamma_step(const_cast<QGanConfig &>(cfg), +1);
            }
            if (cfg.verbose) {
                std::printf("  [epoch %3d] G=%.4f D=%.4f  D(real)=%.3f "
                            "D(fake)=%.3f sigma=%.3f gamma=%.3f p=%.3f\n",
                            epoch + 1, hist.g_loss.back(), hist.d_loss.back(),
                            dr, df, sigma, cfg.gamma, cfg.dropout);
            }
        }
    }
    return hist;
}

// --- small helpers kept out of the main loop for readability --------------
inline Vec xb_row(const Matrix &m, int r, int d)
{
    Vec v(d);
    for (int j = 0; j < d; ++j) {
        v[j] = m(r, j);
    }
    return v;
}

inline void gamma_step(QGanConfig &cfg, int dir)
{
    cfg.gamma = std::clamp(cfg.gamma + dir * cfg.step_gamma, cfg.gamma_lo,
                           cfg.gamma_hi);
}

inline void cfg_mut_dropout(Discriminator &disc, const QGanConfig &cfg, int dir)
{
    QGanConfig &mut = const_cast<QGanConfig &>(cfg);
    mut.dropout = std::clamp(mut.dropout + dir * cfg.step_dropout,
                             cfg.dropout_lo, cfg.dropout_hi);
    disc.set_dropout(mut.dropout);
}

} // namespace qsynth
