#include "Layer.h"

#include <EigenRand/EigenRand>

namespace nn {

Matrix Layer::InitA(Index out_dim, Index in_dim, RNG& rng) {
    // Нормальное распределение N(0, 0.05)
    return Eigen::Rand::normal<Matrix>(out_dim, in_dim, 0.0, 0.05, rng.gen);
}

Vector Layer::InitB(Index out_dim, RNG& rng) {
    return Eigen::Rand::normal<Vector>(out_dim, 1, 0.0, 0.01, rng.gen);
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
    y_ = sigma_->forward(z_);
    return y_;
}

Matrix Layer::GradA(const Vector& x, const Vector& upstream) const {
    Vector z = A_ * x + b_;
    Vector y = sigma_->forward(z);
    Vector d_sigma = sigma_->derivative(y);
    Vector dL_dz = upstream.array() * d_sigma.array();
    return dL_dz * x.transpose();
}

Vector Layer::GradB(const Vector& x, const Vector& upstream) const {
    Vector z = A_ * x + b_;
    Vector y = sigma_->forward(z);
    Vector d_sigma = sigma_->derivative(y);
    Vector dL_dz = upstream.array() * d_sigma.array();
    return dL_dz;
}

Vector Layer::BackpropToPrev(const Vector& x, const Vector& upstream) const {
    Vector z = A_ * x + b_;
    Vector y = sigma_->forward(z);
    Vector d_sigma = sigma_->derivative(y);
    Vector dL_dz = upstream.array() * d_sigma.array();
    return A_.transpose() * dL_dz;
}

Vector Layer::BackwardDy(const Vector& dL_dy) {
    Vector d_sigma = sigma_->derivative(y_);
    Vector dL_dz = dL_dy.array() * d_sigma.array();
    dA_sum_ += dL_dz * x_.transpose();
    db_sum_ += dL_dz;
    return A_.transpose() * dL_dz;
}

Vector Layer::BackwardDz(const Vector& dL_dz) {
    dA_sum_ += dL_dz * x_.transpose();
    db_sum_ += dL_dz;
    return A_.transpose() * dL_dz;
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