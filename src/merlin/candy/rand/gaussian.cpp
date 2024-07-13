// Copyright 2024 quocdang1998
#include "merlin/candy/rand/gaussian.hpp"

#include <cmath>    // std::sqrt
#include <sstream>  // std::ostringstream

#include "merlin/env.hpp"     // merlin::Environment
#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Gaussian
// ---------------------------------------------------------------------------------------------------------------------

// Change mean and standard deviation of the distribution
void candy::rand::Gaussian::set_params(double mean, double std_dev) {
    // argument checking
    if (mean < 0) {
        Fatal<std::invalid_argument>("Expected non-negative mean value.\n");
    }
    if (std_dev < 0) {
        Fatal<std::invalid_argument>("Negative std.\n");
    }
    // initialize generator
    if (mean < std_dev) {
        Warning("Mean is smaller than variance, distribution will be skewed.\n");
    }
    this->generator_ = std::normal_distribution<double>(mean, std_dev);
}

// Sample a value for initializing CP model
double candy::rand::Gaussian::sample(void) {
    double result = -1.0;
    do {
        result = this->generator_(Environment::random_generator);
    } while (result < 0);
    return result;
}

// String representation
std::string candy::rand::Gaussian::str(void) const {
    std::ostringstream os;
    os << "<Gaussian()>";
    return os.str();
}

}  // namespace merlin
