#include "Metrics.h"

#include <algorithm>
#include <cmath>

namespace nn {

static inline Vector SoftmaxRow(const Vector& z) {
    const Scalar m = z.maxCoeff();
    Vector e = (z.array() - m).exp().matrix();
    return e / e.sum();
}

// accuracy: сравниваем argmax по строкам
float Accuracy::Value(const Matrix& y_true, const Matrix& y_logits) const {
    assert(y_true.rows() == y_logits.rows());
    int correct = 0;
    for (Index i = 0; i < y_true.rows(); ++i) {
        int yi = 0;
        int pi = 0;
        y_true.row(i).maxCoeff(&yi);
        y_logits.row(i).maxCoeff(&pi);
        correct += (yi == pi);
    }
    return static_cast<float>(correct) / static_cast<float>(y_true.rows());
}

// средняя кросс-энтропия: внутри считает softmax от логитов построчно
float CrossEntropyMetric::Value(const Matrix& y_true, const Matrix& y_logits) const {
    assert(y_true.rows() == y_logits.rows());
    Scalar sum = 0.0f;
    for (Index i = 0; i < y_true.rows(); ++i) {
        Vector p = SoftmaxRow(y_logits.row(i).transpose());
        for (Index j = 0; j < y_true.cols(); ++j) {
            Scalar prob = std::max(1e-12f, std::min(1.0f, p[j]));
            sum += -y_true(i, j) * static_cast<Scalar>(std::log(static_cast<double>(prob)));
        }
    }
    return static_cast<float>(sum / Scalar(y_true.rows()));
}

}  // namespace nn
