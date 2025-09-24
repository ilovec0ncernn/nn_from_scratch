#include "ActivationFunctions.h"

namespace nn {

Vector ReLU::forward(const Vector& z) const {
    Vector y = z;
    for (int i = 0; i < y.size(); ++i)
        y[i] = y[i] > 0.0 ? y[i] : 0.0;
    return y;
}

Vector ReLU::derivative(const Vector& z, const Vector& y) const {
    Vector d = z;
    for (int i = 0; i < d.size(); ++i)
        d[i] = d[i] > 0.0 ? 1.0 : 0.0;
    return d;
}

}  // namespace nn