#include "ActivationFunctions.h"

namespace nn {

Vector ReLU::forward(const Vector& z) const {
    return z.cwiseMax(Scalar(0));
}

Vector ReLU::derivative(const Vector& z) const {
    Vector d(z.size());
    for (Index i = 0; i < z.size(); ++i) {
        d[i] = (z[i] > Scalar(0)) ? Scalar(1) : Scalar(0);
    }
    return d;
}

Vector Identity::forward(const Vector& z) const {
    return z;
}

Vector Identity::derivative(const Vector& z) const {
    return Vector::Ones(z.size());
}

Vector Softmax::forward(const Vector& z) const {
    const Scalar m = z.maxCoeff();
    const Vector e = (z.array() - m).exp().matrix();
    const Scalar s = e.sum();
    return e / s;
}

Vector Softmax::derivative(const Vector& z) const {
    Vector d(z.size());
    for (Index i = 0; i < z.size(); ++i) {
        d[i] = z[i] * (Scalar(1) - z[i]);  // диагональ Якобиана
    }
    return d;
}

Vector Softmax::backwardFromDy(const Vector& y, const Vector& dL_dy) const {
    const Scalar dot = y.dot(dL_dy);
    return (y.array() * (dL_dy.array() - dot)).matrix();
}

}  // namespace nn