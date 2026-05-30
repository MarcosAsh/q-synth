// Q-SYNTH end-to-end driver: preprocess -> train hybrid QGAN and classical-GAN
// baseline -> generate synthetic fraud -> joint fidelity/detectability/
// downstream evaluation against SMOTE and a balanced baseline.
//
// Usage:
//   qsynth [--data path/to/creditcard.csv] [--seed N] [--out DIR]
//          [--epochs E] [--synthetic]
// With no CSV (or --synthetic) an imbalanced dataset is generated internally so
// the full pipeline runs offline.

#include "qsynth/classical_generator.hpp"
#include "qsynth/classifiers.hpp"
#include "qsynth/data.hpp"
#include "qsynth/eval.hpp"
#include "qsynth/preprocess.hpp"
#include "qsynth/qgan.hpp"
#include "qsynth/quantum_generator.hpp"
#include "qsynth/smote.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace qsynth;

namespace {

struct Args {
    std::string data;
    std::string out = "out";
    unsigned int seed = 42;
    int epochs = 100;
    bool force_synth = false;
};

Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (s == "--data") {
            a.data = next();
        } else if (s == "--out") {
            a.out = next();
        } else if (s == "--seed") {
            a.seed = static_cast<unsigned int>(std::stoul(next()));
        } else if (s == "--epochs") {
            a.epochs = std::stoi(next());
        } else if (s == "--synthetic") {
            a.force_synth = true;
        }
    }
    return a;
}

// Generate `n` synthetic fraud samples from a trained generator.
Matrix generate(IGenerator &gen, int n, int d, Rng &rng)
{
    Matrix out(n, d);
    for (int i = 0; i < n; ++i) {
        Vec z = rng.normal_vec(gen.latent_dim());
        Vec x = gen.sample(z);
        for (int j = 0; j < d; ++j) {
            out(i, j) = x[j];
        }
    }
    return out;
}

void write_vector_csv(const std::string &path, const std::string &header,
                      const Vec &v)
{
    std::ofstream f(path);
    f << header << "\n";
    for (std::size_t i = 0; i < v.size(); ++i) {
        f << i << "," << v[i] << "\n";
    }
}

void write_matrix_csv(const std::string &path, const Matrix &m)
{
    std::ofstream f(path);
    for (std::size_t c = 0; c < m.cols; ++c) {
        f << "f" << c << (c + 1 < m.cols ? "," : "\n");
    }
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            f << m(r, c) << (c + 1 < m.cols ? "," : "\n");
        }
    }
}

// Build a labeled downstream dataset of bounded vectors from fraud and
// non-fraud pools.
void assemble(const std::vector<Vec> &fraud, const std::vector<Vec> &legit,
              std::vector<Vec> &X, std::vector<int> &y)
{
    X.clear();
    y.clear();
    for (const auto &v : fraud) {
        X.push_back(v);
        y.push_back(1);
    }
    for (const auto &v : legit) {
        X.push_back(v);
        y.push_back(0);
    }
}

struct DownstreamRow {
    std::string aug;
    ClassMetrics logreg;
    ClassMetrics mlp;
};

} // namespace

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    Rng rng(args.seed);

    // Load the dataset, or synthesize one when no CSV is given.
    Dataset ds;
    bool loaded = false;
    if (!args.force_synth && !args.data.empty()) {
        loaded = load_csv(args.data, ds);
        if (!loaded) {
            std::printf("Could not read %s; falling back to synthetic data.\n",
                        args.data.c_str());
        }
    }
    if (!loaded) {
        ds = make_synthetic(rng);
    }
    std::printf("Dataset: %zu samples, %zu features, %ld fraud (%.3f%%)%s\n",
                ds.X.rows, ds.X.cols, ds.n_fraud,
                100.0 * ds.n_fraud / static_cast<double>(ds.X.rows),
                ds.synthetic ? " [synthetic]" : "");

    // Preprocess into the bounded representation space.
    Preprocessor pp = fit_preprocess(ds.X, ds.y, 10, 4);
    const int d = pp.n_components;
    const int N1 = static_cast<int>(pp.x_norm.rows);
    std::printf("Preprocessed fraud: %d samples in R^%d bounded space\n", N1, d);

    // Non-fraud pool in the same bounded space (for downstream).
    std::vector<Vec> legit_pool;
    int legit_target = std::min<int>(ds.n_legit, N1 * 4);
    for (std::size_t r = 0; r < ds.X.rows && (int)legit_pool.size() < legit_target;
         ++r) {
        if (ds.y[r] == 0) {
            legit_pool.push_back(transform_to_norm(pp, ds.X, r));
        }
    }

    // Real fraud as vectors, split into train / holdout (70/30).
    std::vector<Vec> fraud_all;
    for (int r = 0; r < N1; ++r) {
        fraud_all.push_back(pp.x_norm.row(r));
    }
    std::shuffle(fraud_all.begin(), fraud_all.end(), rng.engine());
    int n_tr = static_cast<int>(0.7 * N1);
    std::vector<Vec> fraud_tr(fraud_all.begin(), fraud_all.begin() + n_tr);
    std::vector<Vec> fraud_te(fraud_all.begin() + n_tr, fraud_all.end());

    Matrix real_tr(fraud_tr.size(), d);
    for (std::size_t i = 0; i < fraud_tr.size(); ++i) {
        for (int j = 0; j < d; ++j) {
            real_tr(i, j) = fraud_tr[i][j];
        }
    }

    // Train the hybrid quantum generator and the classical-GAN baseline.
    QGanConfig cfg;
    cfg.epochs = args.epochs;
    cfg.latent_dim = 4;

    std::printf("\n=== Training Q-SYNTH (hybrid quantum generator) ===\n");
    QuantumGenerator qgen(cfg.latent_dim, 32, d, 8, rng, cfg.lr_g);
    TrainHistory qhist = train_qgan(qgen, real_tr, cfg, rng);

    std::printf("\n=== Training classical GAN baseline ===\n");
    QGanConfig cfg2 = cfg;
    ClassicalGenerator cgen(cfg2.latent_dim, 24, d, rng, cfg2.lr_g);
    TrainHistory chist = train_qgan(cgen, real_tr, cfg2, rng);

    // Generate synthetic fraud from each trained generator.
    int n_eval = std::min(2000, std::max(200, N1));
    Matrix synth_q = generate(qgen, n_eval, d, rng);
    Matrix synth_c = generate(cgen, n_eval, d, rng);
    Matrix synth_s = smote_oversample(real_tr, n_eval, 5, rng);

    // Real reference for fidelity: held-out fraud (resampled to n_eval).
    Matrix real_ref(n_eval, d);
    for (int i = 0; i < n_eval; ++i) {
        const Vec &v = fraud_te[i % fraud_te.size()];
        for (int j = 0; j < d; ++j) {
            real_ref(i, j) = v[j];
        }
    }

    // Distributional fidelity and external detectability.
    FidelityReport f_s = evaluate_fidelity(real_ref, synth_s, rng);
    FidelityReport f_c = evaluate_fidelity(real_ref, synth_c, rng);
    FidelityReport f_q = evaluate_fidelity(real_ref, synth_q, rng);

    std::printf("\n=== Distributional similarity (real vs synthetic fraud) ===\n");
    std::printf("%-9s %10s %10s %10s %10s %8s\n", "Technique", "KS_med",
                "KS_p", "Wass_med", "Wass_P75", "AUC");
    auto prow = [](const char *name, const FidelityReport &f) {
        std::printf("%-9s %10.4f %10.4f %10.4f %10.4f %8.4f\n", name,
                    f.ks_median, f.ks_pvalue, f.wass_median, f.wass_p75, f.auc);
    };
    prow("SMOTE", f_s);
    prow("GAN", f_c);
    prow("Q-SYNTH", f_q);

    // Downstream evaluation: split non-fraud into train/test, hold out a real
    // test set, and evaluate each augmentation on that same test set.
    std::shuffle(legit_pool.begin(), legit_pool.end(), rng.engine());
    int ln_tr = static_cast<int>(0.7 * legit_pool.size());
    std::vector<Vec> legit_tr(legit_pool.begin(), legit_pool.begin() + ln_tr);
    std::vector<Vec> legit_te(legit_pool.begin() + ln_tr, legit_pool.end());

    // Test set: held-out real fraud + held-out real non-fraud.
    std::vector<Vec> Xte;
    std::vector<int> yte;
    assemble(fraud_te, legit_te, Xte, yte);

    auto build_train = [&](const Matrix *synth) {
        std::vector<Vec> fr = fraud_tr;
        if (synth) {
            for (std::size_t i = 0; i < synth->rows; ++i) {
                fr.push_back(synth->row(i));
            }
        }
        // Balance non-fraud count to the (augmented) fraud count.
        std::vector<Vec> lg;
        for (std::size_t i = 0; i < fr.size(); ++i) {
            lg.push_back(legit_tr[i % legit_tr.size()]);
        }
        std::vector<Vec> X;
        std::vector<int> y;
        assemble(fr, lg, X, y);
        return std::make_pair(X, y);
    };

    auto eval_aug = [&](const std::string &name, const Matrix *synth) {
        DownstreamRow row;
        row.aug = name;
        auto train = build_train(synth);
        LogisticRegression lr(d, 0.2, 2000, 1e-3);
        lr.fit(train.first, train.second);
        row.logreg = score_predictions(lr.predict(Xte), yte);
        MlpClassifier mlp(d, 16, rng, 1e-3, 40, 64);
        mlp.fit(train.first, train.second);
        row.mlp = score_predictions(mlp.predict(Xte), yte);
        return row;
    };

    std::vector<DownstreamRow> rows;
    rows.push_back(eval_aug("Balanced", nullptr));
    rows.push_back(eval_aug("SMOTE", &synth_s));
    rows.push_back(eval_aug("GAN", &synth_c));
    rows.push_back(eval_aug("Q-SYNTH", &synth_q));

    std::printf("\n=== Downstream fraud detection (fraud-class y=1) ===\n");
    std::printf("%-9s | %-22s | %-22s\n", "", "LogReg (P/R/F1 AUC)",
                "ANN (P/R/F1 AUC)");
    for (const auto &r : rows) {
        std::printf("%-9s | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                    r.aug.c_str(), r.logreg.prec[1], r.logreg.rec[1],
                    r.logreg.f1[1], r.logreg.auc, r.mlp.prec[1], r.mlp.rec[1],
                    r.mlp.f1[1], r.mlp.auc);
    }

    // Persist artifacts.
    std::string mkdir = "mkdir -p " + args.out;
    if (std::system(mkdir.c_str()) != 0) {
        std::printf("warning: could not create output dir %s\n",
                    args.out.c_str());
    }
    write_vector_csv(args.out + "/qsynth_generator_loss.csv", "epoch,loss",
                     qhist.g_loss);
    write_vector_csv(args.out + "/qsynth_discriminator_loss.csv", "epoch,loss",
                     qhist.d_loss);
    write_matrix_csv(args.out + "/synthetic_fraud_norm.csv", synth_q);

    // Inverse-transform synthetic fraud back to the selected feature space.
    Matrix synth_feat(synth_q.rows, pp.selected.size());
    for (std::size_t i = 0; i < synth_q.rows; ++i) {
        Vec feat = inverse_transform(pp, synth_q.row(i));
        for (std::size_t j = 0; j < feat.size(); ++j) {
            synth_feat(i, j) = feat[j];
        }
    }
    write_matrix_csv(args.out + "/synthetic_fraud_features.csv", synth_feat);

    std::printf("\nArtifacts written to %s/\n", args.out.c_str());
    return 0;
}
