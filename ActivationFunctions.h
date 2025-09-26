#pragma once

#include "Alias.h"

namespace nn {

class ActivationFunction {
   public:
    virtual ~ActivationFunction() = default;
    virtual Vector Forward(const Vector& z) const = 0;     // sigma(z) (element-wise)
    virtual Vector Derivative(const Vector& z) const = 0;  // dsigma/dz (element-wise)
};

class ReLU : public ActivationFunction {
   public:
    Vector Forward(const Vector& z) const override;
    Vector Derivative(const Vector& z) const override;
};

class Identity : public ActivationFunction {
   public:
    Vector Forward(const Vector& z) const override;
    Vector Derivative(const Vector& z) const override;
};

class Softmax : public ActivationFunction {
   public:
    Vector Forward(const Vector& z) const override;
    Vector Derivative(const Vector& z) const override;
    Vector BackwardFromDy(const Vector& y, const Vector& dL_dy) const;
};

}  // namespace nn