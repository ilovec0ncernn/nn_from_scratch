#include "LossFunctions.h"

#include <cassert>
#include <cmath>

namespace nn {

Scalar MSE::Loss(const Vector& y_true, const Vector& y_pred) const {
    assert(y_true.size() == y_pred.size() && "MSE size mismatch");
    return (y_pred - y_true).squaredNorm() / Scalar(y_true.size());
}

Vector MSE::Grad(const Vector& y_true, const Vector& y_pred) const {
    assert(y_true.size() == y_pred.size() && "MSE gradient size mismatch");
    return (Scalar(2) / Scalar(y_true.size())) * (y_pred - y_true);
}

static inline Vector SoftmaxStable(const Vector& z) {
    const Scalar m = z.maxCoeff();
    const Vector e = (z.array() - m).exp().matrix();
    return e / e.sum();
}

Scalar CrossEntropyWithLogits::Loss(const Vector& y_true, const Vector& logits) const {
    Vector p = SoftmaxStable(logits);
    Scalar s = Scalar(0);
    for (Index i = 0; i < y_true.size(); ++i) {
        Scalar pi = std::max(Scalar(1e-12), std::min(Scalar(1), p[i]));
        s += -y_true[i] * static_cast<Scalar>(std::log(static_cast<double>(pi)));
    }
    return s;
}

Vector CrossEntropyWithLogits::Grad(const Vector& y_true, const Vector& logits) const {
    return SoftmaxStable(logits) - y_true;
}

}  // namespace nn