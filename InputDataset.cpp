#include "InputDataset.h"

#include "mnist/mnist_reader.hpp"

namespace nn {

static inline Vector ToOneHot(int label) {
    Vector y = Vector::Zero(10);
    y[label] = 1.0;
    return y;
}

static inline void FillXy(Matrix& X, Matrix& y, const std::vector<std::vector<uint8_t>>& images,
                          const std::vector<uint8_t>& labels) {
    const int n = static_cast<int>(images.size());
    X.resize(n, 784);
    y.resize(n, 10);
    for (int i = 0; i < n; ++i) {
        Vector x(784);
        for (int j = 0; j < 784; ++j)
            x[j] = images[i][j] / 255.0;  // [0,1]
        X.row(i) = x.transpose();
        y.row(i) = ToOneHot(static_cast<int>(labels[i])).transpose();
    }
}

Split InputDataset::load_mnist() {
    // читает 60k train + 10k test из *.ubyte
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    // dataset.training_images/labels, dataset.test_images/labels
    Split split;
    FillXy(split.X_train, split.y_train, dataset.training_images, dataset.training_labels);
    FillXy(split.X_test, split.y_test, dataset.test_images, dataset.test_labels);
    return split;
}

}  // namespace nn