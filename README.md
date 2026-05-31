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

## Results

On the Kaggle ULB dataset (284,807 transactions, 492 fraud, 0.173% positive
rate), 100 epochs, `--seed 42`. The whole run takes about 5 seconds on a CPU.

Both generators reach the adversarial equilibrium where the discriminator can no
longer separate real from generated samples: the classical GAN settles at
`D(real) ~ D(fake) ~ 0.50` and Q-SYNTH at ~0.52, with no mode collapse.

Distributional similarity, real vs synthetic fraud (lower KS/Wasserstein is
better; AUC closer to 0.5 means harder to detect):

| Technique | KS median | Wasserstein median | Detector AUC |
|-----------|-----------|--------------------|--------------|
| SMOTE     | 0.092     | 0.039              | 0.528 |
| Classical GAN | 0.161 | 0.086              | 0.595 |
| Q-SYNTH   | 0.201     | 0.104              | 0.578 |

Downstream fraud-class detection (precision / recall / F1 / AUC):

| Augmentation | LogReg | ANN |
|--------------|--------|-----|
| Balanced | 0.970 / 0.872 / 0.918 / 0.988 | 0.991 / 0.784 / 0.875 / 0.988 |
| SMOTE    | 0.970 / 0.865 / 0.914 / 0.988 | 0.935 / 0.872 / 0.902 / 0.986 |
| Classical GAN | 0.969 / 0.845 / 0.903 / 0.985 | 0.956 / 0.872 / 0.912 / 0.983 |
| Q-SYNTH  | 0.977 / 0.851 / 0.910 / 0.988 | 0.963 / 0.872 / 0.915 / 0.988 |

The qualitative picture matches the paper: SMOTE wins feature-wise fidelity, the
adversarial methods stay competitive downstream, and Q-SYNTH gives the best ANN
F1 while keeping its samples near-undetectable. One divergence from the paper:
there Q-SYNTH beats the classical GAN on marginal fidelity, whereas here the GAN
edges it on KS/Wasserstein. The simplifications below (finite-difference circuit
gradients, matched-but-small capacity, the shared angle matrix) account for the
gap; more layers, parameter-shift gradients, or regularizer tuning narrow it.

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
