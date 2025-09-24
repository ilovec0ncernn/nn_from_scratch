#pragma once
#include <Eigen/Dense>

namespace nn {
    
class LossFunction {
   public:
    virtual ~LossFunction() = default;
    virtual double loss(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const = 0;
    virtual Eigen::VectorXd grad(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const = 0;  // dL/dy_pred
};

class MSE : public LossFunction {
   public:
    double loss(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const override;
    Eigen::VectorXd grad(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) const override;
};

}  // namespace nn