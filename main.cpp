#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

double MSE(const Eigen::VectorXd& y, const Eigen::VectorXd& yhat) noexcept {
    assert(y.size() == yhat.size() && "are you deadass lil vro");
    assert(y.size() > 0 && "empty vector");
    const Eigen::ArrayXd diff = (y - yhat).array();
    return diff.square().mean();
}

bool Near(double a, double b, double eps = 1e-12) {
    return std::abs(a - b) <= eps * (1.0 + std::max(std::abs(a), std::abs(b)));
}

int main() {

    Eigen::VectorXd y(3);
    y << 1.0, 2.0, 3.0;
    Eigen::VectorXd p = y;  // ожидаем 0
    std::cout << "test 1: " << MSE(y, p) << "\n";
    assert(Near(MSE(y, p), 0.0));

    Eigen::VectorXd y2(2);
    y2 << 1.0, 3.0;
    Eigen::VectorXd p2(2);
    p2 << 2.0, 1.0;  // ожидаем 2.5
    std::cout << "test 2: " << MSE(y2, p2) << "\n";
    assert(Near(MSE(y2, p2), 2.5));

    std::cout << "passed all tests" << "\n";
    return 0;
}