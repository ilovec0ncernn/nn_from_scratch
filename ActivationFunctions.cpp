#include "ActivationFunctions.h"

namespace nn {

Vector ReLU::forward(const Vector& z) const {
    return z.cwiseMax(0.0);
}

Vector ReLU::derivative(const Vector& z) const {
    return (z.array() > 0.0).cast<float>().matrix();
}

Vector Softmax::forward(const Vector& z) const {
    const float m = z.maxCoeff();
    const Vector e = (z.array() - m).exp();
    return e / e.sum();
}

Vector Softmax::derivative(const Vector& z) const {
    return (z.array() * (1.0 - z.array())).matrix(); // диагональная часть якобиана
}

}  // namespace nn