#include "Layer.h"

#include <EigenRand/EigenRand>
#include <utility>

namespace nn {

Matrix Layer::InitA(Index out_dim, Index in_dim, RNG& rng) {
    Matrix M = Eigen::Rand::normal<Matrix>(out_dim, in_dim, rng.gen);
    return M * 0.05f;
}

Vector Layer::InitB(Index out_dim, RNG& rng) {
    Vector v = Eigen::Rand::normal<Vector>(out_dim, 1, rng.gen);
    return v * 0.01f;
}

Layer::Layer(Index in_dim, Index out_dim, Activation sigma, RNG& rng)
    : A_(InitA(out_dim, in_dim, rng))
    , b_(InitB(out_dim, rng))
    , sigma_(std::move(sigma))
    , dA_sum_(Matrix::Zero(out_dim, in_dim))
    , db_sum_(Vector::Zero(out_dim)) {
}

Vector Layer::Forward(const Vector& x) {
    Matrix Xm = x;
    Matrix Ym = Forward(Xm);
    return Ym.col(0);
}

Vector Layer::BackwardDy(const Vector& dL_dy) {
    Matrix dY = dL_dy;
    Matrix dX = BackwardDy(dY);
    return dX.col(0);
}

Matrix Layer::Forward(const Matrix& X) {
    x_ = X;

    z_.resize(OutDim(), X.cols());
    z_.colwise() = b_;
    z_.noalias() += A_ * X;

    y_.resize(OutDim(), X.cols());
    for (Index c = 0; c < X.cols(); ++c) {
        y_.col(c) = sigma_.forward(z_.col(c));
    }
    return y_;
}

Matrix Layer::BackwardDy(const Matrix& dL_dy) {
    const Index b = dL_dy.cols();

    Matrix dL_dz(OutDim(), b);
    for (Index c = 0; c < b; ++c) {
        dL_dz.col(c) = sigma_.backward(y_.col(c), dL_dy.col(c));
    }

    dA_sum_.noalias() += dL_dz * x_.transpose();
    db_sum_.noalias() += dL_dz.rowwise().sum();

    Matrix dL_dx = A_.transpose() * dL_dz;
    return dL_dx;
}

void Layer::ZeroGrad() {
    dA_sum_.setZero();
    db_sum_.setZero();
}

void Layer::Step(float lr, int batch_size) {
    const Scalar scale = lr / static_cast<Scalar>(batch_size);
    A_ -= scale * dA_sum_;
    b_ -= scale * db_sum_;
    ZeroGrad();
}

}  // namespace nn
