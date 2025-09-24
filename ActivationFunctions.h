#include "Alias.h"

namespace nn {

class ActivationFunction {
   public:
    virtual ~ActivationFunction() = default;
    virtual Vector forward(const Vector& z) const = 0;                      // sigma(z)
    virtual Vector derivative(const Vector& z, const Vector& y) const = 0;  // dsigma/dz (element-wise)
};

class ReLU : public ActivationFunction {
   public:
    Vector forward(const Vector& z) const override;
    Vector derivative(const Vector& z, const Vector& y) const override;
};

}  // namespace nn