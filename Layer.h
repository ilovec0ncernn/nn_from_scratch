#pragma once

#include <Eigen/Dense>

#include "ActivationFunctions.h"
#include "Alias.h"

namespace nn {

enum class In : Index {};
enum class Out : Index {};

class Layer {
   public:
    Layer(Index in_dim, Index out_dim, Activation sigma, RNG& rng);

    Vector Forward(const Vector& x);
    Vector BackwardDy(const Vector& dL_dy);

    Matrix Forward(const Matrix& X);
    Matrix BackwardDy(const Matrix& dL_dy);

    void Step(Scalar lr, int batch_size);
    void ZeroGrad();

    Index InDim() const {
        return static_cast<Index>(A_.cols());
    }
    Index OutDim() const {
        return static_cast<Index>(A_.rows());
    }

   private:
    Matrix A_;
    Vector b_;
    Activation sigma_;

    Matrix x_, z_, y_;  // кэши для backprop

    Matrix dA_sum_;
    Vector db_sum_;

    static Matrix InitA(Index out_dim, Index in_dim, RNG& rng);
    static Vector InitB(Index out_dim, RNG& rng);
};

}  // namespace nn
