#include "ActivationFunctions.h"

namespace nn {

Vector ReLU::Forward(const Vector& z) const {
    return z.cwiseMax(0.0f);
}

Vector ReLU::Derivative(const Vector& z) const {
    Vector d(z.size());
    for (Index i = 0; i < z.size(); ++i) {
        d[i] = (z[i] > 0.0f) ? 1.0f : 0.0f;
    }
    return d;
}

Vector Identity::Forward(const Vector& z) const {
    return z;
}

Vector Identity::Derivative(const Vector& z) const {
    return Vector::Ones(z.size());
}

Vector Softmax::Forward(const Vector& z) const {
    const Scalar m = z.maxCoeff();
    const Vector e = (z.array() - m).exp().matrix();
    const Scalar s = e.sum();
    return e / s;
}

Vector Softmax::Derivative(const Vector& z) const {
    Vector d(z.size());
    for (Index i = 0; i < z.size(); ++i) {
        d[i] = z[i] * (1.0f - z[i]);  // диагональ якобиана
    }
    return d;
}

Vector Softmax::BackwardFromDy(const Vector& y, const Vector& dL_dy) const {
    const Scalar dot = y.dot(dL_dy);
    return (y.array() * (dL_dy.array() - dot)).matrix();
}

}  // namespace nn
