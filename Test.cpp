#include "Test.h"

#include <iomanip>
#include <iostream>

#include "ActivationFunctions.h"
#include "Alias.h"
#include "InputDataset.h"
#include "LossFunctions.h"
#include "Metrics.h"
#include "Network.h"

namespace nn {

static Matrix TakeRows(const Matrix& M, int n) {
    if (n < 0 || n >= M.rows()) {
        return M;
    }
    return M.topRows(n);
}

void TestMnistBasic(const TestConfig& cfg) {
    std::cout << "training model on mnist dataset" << std::endl;

    auto split = InputDataset::LoadMnist();

    Matrix X_train = TakeRows(split.X_train, cfg.train_limit);
    Matrix y_train = TakeRows(split.y_train, cfg.train_limit);
    Matrix X_test = TakeRows(split.X_test, cfg.test_limit);
    Matrix y_test = TakeRows(split.y_test, cfg.test_limit);

    RNG rng(42);
    ReLU relu;
    Softmax sm;
    Identity id;

    Network net(rng);
    net.AddLayer(784, 128, &relu, rng);  // hidden-1 ReLU
    net.AddLayer(128, 10, &sm, rng);     // hidden-2 Softmax
    net.AddLayer(10, 10, &id, rng);      // выход логиты

    // обучение
    TrainConfig tcfg;
    tcfg.epochs = cfg.epochs;
    tcfg.batch_size = cfg.batch_size;
    tcfg.lr = cfg.lr;
    MSE mse;
    CrossEntropyWithLogits celoss;
    net.Train(X_train, y_train, X_test, y_test, tcfg, celoss);

    // итоговые метрики
    Accuracy acc_metric;
    CrossEntropyMetric ce_metric;
    Matrix logits = net.Predict(X_test);
    float final_acc = acc_metric.Value(y_test, logits);
    float final_ce = ce_metric.Value(y_test, logits);

    std::cout << "[test] final test accuracy=" << std::fixed << std::setprecision(4) << final_acc
              << ", final CE=" << std::fixed << std::setprecision(4) << final_ce << std::endl;
}

void RunAllTests() {
    TestConfig cfg;
    TestMnistBasic(cfg);
}

}  // namespace nn
