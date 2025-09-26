#include "Layer.h"

#include <EigenRand/EigenRand>

namespace nn {

Matrix Layer::InitA(Index out_dim, Index in_dim, RNG& rng) {
    Matrix M = Eigen::Rand::normal<Matrix>(out_dim, in_dim, rng.gen);
    return M * 0.05f;
}

Vector Layer::InitB(Index out_dim, RNG& rng) {
    Vector v = Eigen::Rand::normal<Vector>(out_dim, 1, rng.gen);
    return v * 0.01f;
}

Layer::Layer(Index in_dim, Index out_dim, const ActivationFunction* sigma, RNG& rng)
    : in_dim_(in_dim)
    , out_dim_(out_dim)
    , A_(InitA(out_dim, in_dim, rng))
    , b_(InitB(out_dim, rng))
    , sigma_(sigma)
    , dA_sum_(Matrix::Zero(out_dim, in_dim))
    , db_sum_(Vector::Zero(out_dim)) {
}

Vector Layer::Forward(const Vector& x) {
    x_ = x;
    z_ = A_ * x_ + b_;
    y_ = sigma_->Forward(z_);
    return y_;
}

Vector Layer::BackwardDy(const Vector& dL_dy) {
    // dL/dz
    Vector dL_dz;
    if (auto sm = dynamic_cast<const Softmax*>(sigma_)) {
        dL_dz = sm->BackwardFromDy(y_, dL_dy);
    } else {
        Vector d = sigma_->Derivative(z_);
        dL_dz = (dL_dy.array() * d.array()).matrix();
    }

    // градиенты параметров
    dA_sum_ += dL_dz * x_.transpose();
    db_sum_ += dL_dz;

    return A_.transpose() * dL_dz; // dL/dx
}

void Layer::ZeroGrad() {
    dA_sum_.setZero();
    db_sum_.setZero();
}

void Layer::Step(float lr, int batch_size) {
    const float scale = lr / static_cast<float>(batch_size);
    A_ -= scale * dA_sum_;
    b_ -= scale * db_sum_;
    ZeroGrad();
}

}  // namespace nn