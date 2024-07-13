// Copyright 2024 quocdang1998
#include "merlin/candy/rand/uniform.hpp"

#include "merlin/env.hpp"     // merlin::Environment
#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from relative range value
candy::rand::Uniform::Uniform(double k) : k_(k) {
    if (k < 0) {
        Fatal<std::invalid_argument>("Negative relative range value.\n");
    }
    if (k > 1) {
        Warning("Relative range is greater than 1.0, may generate negative value.\n");
    }
}

// Change mean of the distribution
void candy::rand::Uniform::set_params(double mean) {
    // argument checking
    if (mean < 0) {
        Fatal<std::invalid_argument>("Expected non-negative mean value.\n");
    }
    this->generator_ = std::uniform_real_distribution<double>(mean * (1.0 - this->k_), mean * (1.0 + this->k_));
}

// Sample a value for initializing CP model
double candy::rand::Uniform::sample(void) { return this->generator_(Environment::random_generator); }

// String representation
std::string candy::rand::Uniform::str(void) const {
    std::ostringstream os;
    os << "<Uniform(k=" << this->k_ << ")>";
    return os.str();
}

}  // namespace merlin
