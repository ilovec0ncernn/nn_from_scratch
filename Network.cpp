#include "Network.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>

namespace nn {

Network::Network(RNG& rng) {
    // число из нашего EigenRand генератора используем как seed для std::shuffle
    auto seed = static_cast<std::mt19937_64::result_type>(rng.gen());
    shuffle_eng_.seed(seed);
}

void Network::AddLayer(Index in_dim, Index out_dim, Activation sigma, RNG& rng) {
    layers_.emplace_back(in_dim, out_dim, std::move(sigma), rng);
}

Vector Network::PredictOne(const Vector& x) const {
    Vector h = x;
    for (const auto& L : layers_) {
        auto& Lnc = const_cast<Layer&>(L);
        h = Lnc.Forward(h);
    }
    return h;
}

Matrix Network::Predict(const Matrix& X) const {
    const Index n = X.rows();
    const Index dout = layers_.empty() ? 0 : layers_.back().OutDim();
    Matrix Y(n, dout);
    for (Index i = 0; i < n; ++i) {
        Y.row(i) = PredictOne(X.row(i).transpose()).transpose();
    }
    return Y;
}

void Network::Train(const Matrix& X, const Matrix& Y, const Matrix& X_val, const Matrix& Y_val, const TrainConfig& cfg,
                    LossFunction& loss) {
    assert(X.rows() == Y.rows());
    const int n = static_cast<int>(X.rows());
    const int b = cfg.batch_size;

    Accuracy acc_metric;
    CrossEntropyMetric ce_metric;

    std::vector<Index> order(n);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), shuffle_eng_);

        int correct = 0;
        int seen = 0;

        for (int i = 0; i < n; i += b) {
            const int r = std::min(b, n - i);

            for (auto& L : layers_) {
                L.ZeroGrad();
            }

            for (int j = 0; j < r; ++j) {
                Index idx = order[i + j];

                Vector x = X.row(idx).transpose();
                Vector y_true = Y.row(idx).transpose();

                // forward
                for (auto& L : layers_) {
                    x = L.Forward(x);
                }
                Vector logits = x;

                // train accuracy
                int yi = 0, pi = 0;
                y_true.maxCoeff(&yi);
                logits.maxCoeff(&pi);
                correct += (yi == pi);
                ++seen;

                Vector grad = loss.Gradient(y_true, logits);  // dL/d(logits)

                // backpropagation
                grad = layers_.back().BackwardDy(grad);
                for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
                    grad = layers_[l].BackwardDy(grad);
                }
            }

            for (auto& L : layers_) {
                L.Step(cfg.lr, r);
            }
        }

        const float train_acc = static_cast<float>(correct) / static_cast<float>(std::max(1, seen));
        Matrix logits_val = Predict(X_val);
        const float val_acc = acc_metric.Value(Y_val, logits_val);
        const float val_ce = ce_metric.Value(Y_val, logits_val);

        std::cout << "epoch " << epoch << ": train_acc=" << std::fixed << std::setprecision(4) << train_acc
                  << ", val_acc=" << std::fixed << std::setprecision(4) << val_acc << ", val_ce=" << std::fixed
                  << std::setprecision(4) << val_ce << std::endl;
    }
}

}  // namespace nn
