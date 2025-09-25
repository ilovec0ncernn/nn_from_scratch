#pragma once

#include "Alias.h"

namespace nn {

class LossFunction {
   public:
    virtual ~LossFunction() = default;
    virtual Scalar Loss(const Vector& y_true, const Vector& y_pred) const = 0;
    virtual Vector Grad(const Vector& y_true, const Vector& y_pred) const = 0;
};

class MSE : public LossFunction {
   public:
    Scalar Loss(const Vector& y_true, const Vector& y_pred) const override;
    Vector Grad(const Vector& y_true, const Vector& y_pred) const override;
};

class CrossEntropyWithLogits : public LossFunction {
   public:
    Scalar Loss(const Vector& y_true, const Vector& logits) const override;
    Vector Grad(const Vector& y_true, const Vector& logits) const override;
};

}  // namespace nn