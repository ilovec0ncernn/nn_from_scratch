#include "ActivationFunctions.h"

namespace nn {

static Vector ReluForward(const Vector& z) {
    return z.cwiseMax(0.0f);
}
static Vector ReluBackward(const Vector& y, const Vector& dL_dy) {
    return ((y.array() > 0.0f).cast<Scalar>() * dL_dy.array()).matrix();
}

static Vector IdForward(const Vector& z) {
    return z;
}
static Vector IdBackward(const Vector&, const Vector& dL_dy) {
    return dL_dy;
}

static Vector SoftmaxForward(const Vector& z) {
    const Scalar m = z.maxCoeff();
    const Vector e = (z.array() - m).exp().matrix();
    return e / e.sum();
}
static Vector SoftmaxBackward(const Vector& y, const Vector& dL_dy) {
    const Scalar dot = y.dot(dL_dy);
    return (y.array() * (dL_dy.array() - dot)).matrix();
}

Activation Activation::ReLU() {
    return Activation{ReluForward, ReluBackward};
}
Activation Activation::Identity() {
    return Activation{IdForward, IdBackward};
}
Activation Activation::Softmax() {
    return Activation{SoftmaxForward, SoftmaxBackward};
}

}  // namespace nn
