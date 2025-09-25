#pragma once

#include <string>

#include "Alias.h"

#ifndef NN_MNIST_DIR
#define NN_MNIST_DIR "external_submodules/mnist"
#endif

namespace nn {

struct Split {
    Matrix X_train, y_train;
    Matrix X_test, y_test;
};

class InputDataset {
   public:
    static Split LoadMnist(const std::string& dir = NN_MNIST_DIR);
};

}  // namespace nn