// Copyright 2024 quocdang1998
#include "merlin/candy/intlz/uniform.hpp"

#include <cmath>  // std::sqrt

#include "merlin/env.hpp"     // merlin::Environment
#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from mean and variance
candy::intlz::Uniform::Uniform(double mean, double scale) {
    // argument checking
    if (mean < 0) {
        Fatal<std::invalid_argument>("Expected non-negative mean value.\n");
    }
    if (scale < 0) {
        Fatal<std::invalid_argument>("Negative scale.\n");
    }
    // initialize generator
    if (scale > 1) {
        Warning("Scale is greater than 1.0, may generate negative value.\n");
    }
    this->generator_ = std::uniform_real_distribution<double>(mean - scale * mean, mean + scale * mean);
}

// Sample a value for initializing CP model
double candy::intlz::Uniform::sample(void) { return this->generator_(Environment::random_generator); }

}  // namespace merlin
