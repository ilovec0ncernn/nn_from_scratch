#pragma once

#include "ActivationFunctions.h"
#include "Alias.h"
#include "InputDataset.h"
#include "LossFunctions.h"
#include "Metrics.h"
#include "Network.h"

namespace nn {

struct TestConfig {
    int epochs = 15;
    int batch_size = 64;
    float lr = 0.2f;
    int train_limit = -1;
    int test_limit = -1;
};

void RunAllTests();

void TestMnistBasic(const TestConfig& cfg);

}  // namespace nn