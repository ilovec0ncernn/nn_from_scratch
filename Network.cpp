#include "Network.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

namespace nn {

Network& Network::AddFirstLayer(Index in_dim, Index out_dim, Activation sigma, RNG& rng) {
    assert(!has_input_dim_ && "AddFirstLayer called twice");
    layers_.emplace_back(in_dim, out_dim, std::move(sigma), rng);
    last_dim_ = out_dim;
    has_input_dim_ = true;
    return *this;
}

Network& Network::AddLayer(Index out_dim, Activation sigma, RNG& rng) {
    assert(has_input_dim_ && "call AddFirstLayer() first");
    const Index in_dim = last_dim_;
    layers_.emplace_back(in_dim, out_dim, std::move(sigma), rng);
    last_dim_ = out_dim;
    return *this;
}

Matrix Network::ForwardAll(const Matrix& Xb) {
    Matrix h = Xb;
    for (auto& L : layers_)
        h = L.Forward(h);
    return h;
}

Matrix Network::BackwardAll(const Matrix& dY) {
    assert(!layers_.empty());
    Matrix grad = layers_.back().BackwardDy(dY);

    for (auto it = layers_.rbegin() + 1; it != layers_.rend(); ++it) {
        grad = it->BackwardDy(grad);
    }
    return grad;
}

void Network::StepAll(Scalar lr, int batch_size) {
    for (auto& L : layers_)
        L.Step(lr, batch_size);
}

Matrix Network::Predict(const Matrix& X_cols) {
    if (layers_.empty())
        return Matrix::Zero(0, X_cols.cols());
    return ForwardAll(X_cols);
}

Vector Network::PredictOne(const Vector& x) {
    Matrix X(x.size(), 1);
    X.col(0) = x;
    Matrix Y = Predict(X);
    return Y.col(0);
}

void Network::Train(const Matrix& X_cols, const Matrix& Y_cols, const Matrix& X_val_cols, const Matrix& Y_val_cols,
                    const TrainConfig& cfg, const Loss& loss) {

    const Index N = X_cols.cols();
    const Index din = layers_.front().InDim();
    const Index dout = layers_.back().OutDim();
    const int B = std::max(1, cfg.batch_size);
    (void)din;
    (void)dout;

    std::vector<Index> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 eng(cfg.shuffle_seed);

    Metric acc = Metric::Accuracy();
    Metric ce = Metric::CrossEntropy();

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), eng);

        Scalar sum_acc = 0;
        Index seen = 0;

        for (Index start = 0; start < N; start += B) {
            const int r = static_cast<int>(std::min<Index>(B, N - start));

            Matrix Xb(din, r);
            Matrix Yb(dout, r);
            for (int j = 0; j < r; ++j) {
                const Index idx = order[static_cast<size_t>(start + j)];
                Xb.col(j) = X_cols.col(idx);
                Yb.col(j) = Y_cols.col(idx);
            }

            Matrix logits = ForwardAll(Xb);

            const float acc_batch = acc.Value(Yb, logits);
            sum_acc += static_cast<Scalar>(acc_batch) * static_cast<Scalar>(r);
            seen += r;

            Matrix dY = loss.Gradient(Yb, logits);
            BackwardAll(dY);
            StepAll(cfg.lr, r);
        }

        const float train_acc = static_cast<float>(sum_acc / static_cast<Scalar>(std::max<Index>(1, seen)));

        Matrix logits_val = Predict(X_val_cols);
        const float val_acc = acc.Value(Y_val_cols, logits_val);
        const float val_ce = ce.Value(Y_val_cols, logits_val);

        std::cout << "epoch " << epoch << ": train_acc=" << std::fixed << std::setprecision(4) << train_acc
                  << ", val_acc=" << std::fixed << std::setprecision(4) << val_acc << ", val_ce=" << std::fixed
                  << std::setprecision(4) << val_ce << std::endl;
    }
}

}  // namespace nn
