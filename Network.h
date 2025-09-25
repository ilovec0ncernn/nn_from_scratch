#pragma once

#include <random>
#include <vector>

#include "Alias.h"
#include "Layer.h"
#include "LossFunctions.h"
#include "Metrics.h"

namespace nn {

struct TrainConfig {
    int epochs = 10;
    int batch_size = 64;
    float lr = 0.05f;
};

class Network {
   public:
    explicit Network(RNG& rng);

    // конструируем архитектуру нейросети
    void AddLayer(Index in_dim, Index out_dim, const ActivationFunction* sigma, RNG& rng);

    // обучение и вывод метрик в std::cout
    void Train(const Matrix& X, const Matrix& Y, const Matrix& X_val, const Matrix& Y_val, const TrainConfig& cfg,
               LossFunction& loss);

    // предсказания
    Vector PredictOne(const Vector& x) const;  // прямой прогон одного объекта
    Matrix Predict(const Matrix& X) const;     // прогон матрицы объектов (построчно)

   private:
    std::vector<Layer> layers_;
    std::mt19937_64 shuffle_eng_;
};

}  // namespace nn