#pragma once

#include <Eigen/Dense>

#include "ActivationFunctions.h"
#include "Alias.h"

namespace nn {

class Layer {
   public:
    Layer(Index in_dim, Index out_dim, const ActivationFunction* act, RNG& rng);

    Vector Forward(const Vector& x); // forward: y = sigma(Ax + b)
    Vector BackwardDy(const Vector& dL_dy); // backward: на вход dL/dy, возвращает dL/dx и накапливает dA, db

    // шаг по среднему градиенту за батч + обнуление
    void Step(Scalar lr, int batch_size);
    void ZeroGrad();

    const Matrix& A() const {
        return A_;
    }
    const Vector& b() const {
        return b_;
    }
    Matrix& A() {
        return A_;
    }
    Vector& b() {
        return b_;
    }

    Index InDim() const {
        return in_dim_;
    }
    Index OutDim() const {
        return out_dim_;
    }

   private:
    Index in_dim_;
    Index out_dim_;
    Matrix A_;
    Vector b_;
    const ActivationFunction* sigma_;

    Vector x_, z_, y_; // кэши для backprop

    // аккумуляторы градиентов за батч
    Matrix dA_sum_;
    Vector db_sum_;

    // инициализация
    static Matrix InitA(Index out_dim, Index in_dim, RNG& rng);
    static Vector InitB(Index out_dim, RNG& rng);
};

}  // namespace nn