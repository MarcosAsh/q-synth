# Q-SYNTH (C++)

A dependency-free C++17 implementation of **Q-SYNTH**, a hybrid
quantum-classical adversarial framework for synthesizing minority-class fraud
samples in imbalanced tabular data. A parameterized quantum circuit acts as the
generator; a classical neural network acts as the discriminator.

Reimplements the pipeline from Innan et al., *"Q-SYNTH: Hybrid
Quantum-Classical Adversarial Augmentation for Imbalanced Fraud Detection."*

## What it does

```
raw transactions
  -> SelectKBest (ANOVA F) -> keep fraud rows -> standardize
  -> PCA(n=4) -> max-abs normalize into [-1,1]^4        (preprocess.hpp)
  -> hybrid QGAN training in the bounded space          (qgan.hpp)
       generator: classical MLP -> 4-qubit variational circuit -> <Z> readout
       discriminator: 2-layer LeakyReLU MLP -> sigmoid
       losses: BCE + feature matching + moment matching, instance noise,
               label smoothing, adaptive regularization
  -> generate synthetic fraud, inverse-transform to feature space
  -> joint evaluation                                   (eval.hpp, metrics.hpp)
       fidelity: per-dim KS statistic/p-value, 1-Wasserstein
       detectability: AUC of an external logistic-regression detector
       downstream: LogReg + ANN fraud-class precision/recall/F1/AUC
  -> baselines: SMOTE and a classical GAN, same protocol
```

Everything is exact-statevector quantum simulation on the CPU; there are no
external dependencies beyond the standard library.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build        # unit tests
```

To build with Clang and emit `compile_commands.json` for your editor:

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
ln -sf build/compile_commands.json .
```

## Run

```sh
# Real data: download the Kaggle Credit Card Fraud CSV to data/creditcard.csv
./build/qsynth --data data/creditcard.csv --epochs 100 --out out

# No data on hand: synthesize an imbalanced dataset and run the full pipeline
./build/qsynth --synthetic --epochs 100 --out out
```

Flags: `--data PATH`, `--synthetic`, `--seed N`, `--epochs E`, `--out DIR`.

Artifacts written to `out/`: generator/discriminator loss curves, synthetic
fraud in both the bounded space and the inverse-transformed feature space.

## Layout

| File | Role |
|------|------|
| `statevector.hpp` | exact few-qubit statevector simulator (RX/RY/RZ/CNOT, `<Z>`) |
| `quantum_generator.hpp` | hybrid MLP + variational-circuit generator |
| `classical_generator.hpp` | classical-GAN generator baseline |
| `discriminator.hpp` | classical discriminator with backprop-to-input |
| `qgan.hpp` | adversarial trainer with the full regularized objective |
| `preprocess.hpp` | SelectKBest, standardize, PCA (Jacobi), normalize, inverse |
| `smote.hpp` | SMOTE oversampling baseline |
| `classifiers.hpp` | logistic regression, ReLU MLP, classification metrics |
| `metrics.hpp` | KS statistic + p-value, 1-Wasserstein, AUC-ROC |
| `eval.hpp` | fidelity + external-detectability protocol |

## Scope notes

This is a faithful reimplementation of the methodology, not a bit-for-bit
reproduction of the paper's tables. Differences by design: gradients of the
circuit angles use central finite differences on the noiseless simulator rather
than PennyLane autodiff; the downstream classifier suite is LogReg + ANN (the
paper also reports a QNN, random forest, and XGBoost). And without the Kaggle
CSV the data is synthetic. The quantum generator shares one angle matrix across
all variational layers, exactly as specified in the paper.
