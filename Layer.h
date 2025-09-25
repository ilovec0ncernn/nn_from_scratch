#include <Eigen/Dense>

#include "ActivationFunctions.h"
#include "Alias.h"

namespace nn {

class Layer {
   public:
    // конструктор через случайную инициализацию
    Layer(Index in_dim, Index out_dim, const ActivationFunction* act, RNG& rng);

    Vector Forward(const Vector& x);  // forward prop: y = sigma(Ax + b)

    // upstream == dL/dy (градиент из следующего слоя ПОСЛЕ активации текущего слоя).
    Matrix GradA(const Vector& x, const Vector& upstream) const;           // dL/dA
    Vector GradB(const Vector& x, const Vector& upstream) const;           // dL/db
    Vector BackpropToPrev(const Vector& x, const Vector& upstream) const;  // dL/dx

    // стандартный случай: на вход dL/dy (будет домножаться на sigma'(z))
    Vector BackwardDy(const Vector& dL_dy);  // возвращает dL/dx и накапливает градиенты

    // специальный случай (softmax + кросс-энтропия на последнем слое): на вход dL/dz
    Vector BackwardDz(const Vector& dL_dz);  // возвращает dL/dx и накапливает градиенты

    // применение и обнуление батчевых градиентов
    void Step(float lr, int batch_size);
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

    Vector x_, z_, y_;  // кэши для backpropagation

    // аккумуляторы градиентов за батч
    Matrix dA_sum_;
    Vector db_sum_;

    // инициализация
    static Matrix InitA(Index out_dim, Index in_dim, RNG& rng);
    static Vector InitB(Index out_dim, RNG& rng);
};

}  // namespace nn