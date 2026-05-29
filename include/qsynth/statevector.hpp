// A tiny exact statevector simulator for the few-qubit generator circuit.
// Supports the gate set used by Q-SYNTH: RX, RY, RZ single-qubit rotations,
// a ring of CNOTs, and Pauli-Z expectation-value readout.
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace qsynth {

using Complex = std::complex<double>;

class StateVector {
public:
    explicit StateVector(int n_qubits)
        : n_(n_qubits), amp_(std::size_t(1) << n_qubits, Complex(0.0, 0.0))
    {
        amp_[0] = Complex(1.0, 0.0); // |0...0>
    }

    int n_qubits() const { return n_; }

    void reset()
    {
        std::fill(amp_.begin(), amp_.end(), Complex(0.0, 0.0));
        amp_[0] = Complex(1.0, 0.0);
    }

    // Apply a single-qubit 2x2 gate [[a,b],[c,d]] to qubit q.
    void apply_1q(int q, Complex a, Complex b, Complex c, Complex d)
    {
        const std::size_t bit = std::size_t(1) << q;
        for (std::size_t i = 0; i < amp_.size(); ++i) {
            if (i & bit) {
                continue; // process each pair from its lower index only
            }
            std::size_t j = i | bit;
            Complex x0 = amp_[i];
            Complex x1 = amp_[j];
            amp_[i] = a * x0 + b * x1;
            amp_[j] = c * x0 + d * x1;
        }
    }

    void rx(int q, double phi)
    {
        double c = std::cos(phi / 2.0);
        double s = std::sin(phi / 2.0);
        apply_1q(q, Complex(c, 0.0), Complex(0.0, -s), Complex(0.0, -s),
                 Complex(c, 0.0));
    }

    void ry(int q, double phi)
    {
        double c = std::cos(phi / 2.0);
        double s = std::sin(phi / 2.0);
        apply_1q(q, Complex(c, 0.0), Complex(-s, 0.0), Complex(s, 0.0),
                 Complex(c, 0.0));
    }

    void rz(int q, double phi)
    {
        Complex em(std::cos(-phi / 2.0), std::sin(-phi / 2.0));
        Complex ep(std::cos(phi / 2.0), std::sin(phi / 2.0));
        apply_1q(q, em, Complex(0, 0), Complex(0, 0), ep);
    }

    void cnot(int control, int target)
    {
        const std::size_t cb = std::size_t(1) << control;
        const std::size_t tb = std::size_t(1) << target;
        for (std::size_t i = 0; i < amp_.size(); ++i) {
            if ((i & cb) && !(i & tb)) {
                std::size_t j = i | tb;
                std::swap(amp_[i], amp_[j]);
            }
        }
    }

    // <Z_q> = sum_i |amp_i|^2 * (+1 if bit q is 0 else -1).
    double expect_z(int q) const
    {
        const std::size_t bit = std::size_t(1) << q;
        double acc = 0.0;
        for (std::size_t i = 0; i < amp_.size(); ++i) {
            double p = std::norm(amp_[i]);
            acc += (i & bit) ? -p : p;
        }
        return acc;
    }

private:
    int n_;
    std::vector<Complex> amp_;
};

} // namespace qsynth
