// Common interface so the adversarial trainer can drive either the hybrid
// quantum generator or the classical-GAN baseline without templating.
#pragma once

#include "linalg.hpp"

namespace qsynth {

class IGenerator {
public:
    virtual ~IGenerator() = default;

    virtual int latent_dim() const = 0;

    // Begin accumulating per-sample caches for a fresh mini-batch.
    virtual void begin_batch() = 0;

    // Forward one latent vector, caching intermediates; returns the sample.
    virtual Vec forward_cached(const Vec &z) = 0;

    // Backprop dL/dx_hat for the cache at position `idx` in the current batch.
    virtual void backward(int idx, const Vec &grad_x) = 0;

    // Scale accumulated grads, clip global norm, and apply the optimizer step.
    virtual void apply_grads(double scale, double clip_norm) = 0;

    // Forward without caching (evaluation / sampling).
    virtual Vec sample(const Vec &z) = 0;
};

} // namespace qsynth
