#pragma once

#include <functional>

#include "Alias.h"

namespace nn {

class Activation {
    using ForwardSig = Vector(const Vector&);
    using BackwardSig = Vector(const Vector&, const Vector&);

   public:
    Vector forward(const Vector& z) const {
        return forward_(z);
    }
    Vector backward(const Vector& y, const Vector& u) const {
        return backward_(y, u);
    }
    Activation() = default;
    Activation(std::function<ForwardSig> fwd, std::function<BackwardSig> bwd)
        : forward_(std::move(fwd)), backward_(std::move(bwd)) {
    }

    static Activation ReLU();
    static Activation Identity();
    static Activation Softmax();

   private:
    std::function<ForwardSig> forward_;
    std::function<BackwardSig> backward_;
};

}  // namespace nn
