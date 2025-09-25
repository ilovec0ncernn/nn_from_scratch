#pragma once
#include "Alias.h"

namespace nn {

class Metric {
   public:
    virtual ~Metric() = default;
    virtual float Value(const Matrix& Y_true_onehot, const Matrix& Y_logits) const = 0;
};

class Accuracy : public Metric {
   public:
    float Value(const Matrix& Y_true_onehot, const Matrix& Y_logits) const override;
};

class CrossEntropyMetric : public Metric {
   public:
    float Value(const Matrix& Y_true_onehot, const Matrix& Y_logits) const override;
};

}  // namespace nn
