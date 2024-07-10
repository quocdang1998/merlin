// Copyright 2024 quocdang1998
#include "merlin/candy/intlz/gaussian.hpp"

#include <cmath>  // std::sqrt

#include "merlin/env.hpp"     // merlin::Environment
#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Gaussian
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from mean and variance
candy::intlz::Gaussian::Gaussian(double mean, double scale) {
    // argument checking
    if (mean < 0) {
        Fatal<std::invalid_argument>("Expected non-negative mean value.\n");
    }
    if (scale < 0) {
        Fatal<std::invalid_argument>("Negative std.\n");
    }
    // initialize generator
    if (mean < scale) {
        Warning("Mean is smaller than variance, distribution will be skewed.\n");
    }
    this->generator_ = std::normal_distribution<double>(mean, scale);
}

// Sample a value for initializing CP model
double candy::intlz::Gaussian::sample(void) {
    double result = -1.0;
    do {
        result = this->generator_(Environment::random_generator);
    } while (result < 0);
    return result;
}

}  // namespace merlin
