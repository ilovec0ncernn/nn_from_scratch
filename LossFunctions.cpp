#include "LossFunctions.h"
#include <cassert>

namespace nn {

double MSE::loss(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const {
    assert(y_true.size() == y_pred.size() && "MSE size mismatch");
    return (y_pred - y_true).array().square().mean();
}

Eigen::VectorXd MSE::grad(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const {
    assert(y_true.size() == y_pred.size() && "MSE gradient size mismatch");
    return 2.0 * (y_pred - y_true) / static_cast<double>(y_true.size());
}

}  // namespace nn