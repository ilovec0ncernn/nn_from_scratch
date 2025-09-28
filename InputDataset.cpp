#include "InputDataset.h"

#ifdef _MSC_VER
#include "mnist/mnist_reader_less.hpp"
#else
#include "mnist/mnist_reader.hpp"
#endif

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nn {

static inline Vector ToOneHot(int label) {
    Vector y = Vector::Zero(10);
    y[static_cast<Index>(label)] = 1.0f;
    return y;
}

static inline void FillXy(Matrix& X, Matrix& y, const std::vector<std::vector<uint8_t>>& images,
                          const std::vector<uint8_t>& labels) {
    auto n = static_cast<Index>(images.size());
    X.resize(n, 784);
    y.resize(n, 10);
    for (Index i = 0; i < n; ++i) {
        // вход это строка из 784 признаков, ее нужно нормализовать в [0, 1]
        for (int j = 0; j < 784; ++j) {
            X(i, j) = static_cast<Scalar>(images[static_cast<size_t>(i)][j]) / 255.0f;
        }
        y.row(i) = ToOneHot(labels[static_cast<size_t>(i)]).transpose();
    }
}

Split InputDataset::LoadMnist(const std::filesystem::path& dir) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dir.string());
    // dataset.training_images/labels, dataset.test_images/labels
    if (dataset.training_images.empty() || dataset.test_images.empty()) {
        throw std::runtime_error("mnist files not found in \"" + dir.string());
    }
    Split split;
    FillXy(split.X_train, split.y_train, dataset.training_images, dataset.training_labels);
    FillXy(split.X_test, split.y_test, dataset.test_images, dataset.test_labels);
    return split;
}

}  // namespace nn
