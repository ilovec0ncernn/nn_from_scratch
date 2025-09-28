#pragma once

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
