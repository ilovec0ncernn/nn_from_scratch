#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace nn {

using Scalar = float;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;               // NO MOVIES
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;  // IM LOCKED IN
using Index = Eigen::Index;                                            // IM IN THE TRENCHES

struct RNG {
    Eigen::Rand::P8_mt19937_64 gen;
    explicit RNG(std::uint64_t seed = 42u) : gen(seed) {
    }
};

}  // namespace nn
