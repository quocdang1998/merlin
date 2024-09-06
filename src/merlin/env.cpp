// Copyright quocdang1998
#include "merlin/env.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------------------------------

// Mutex
std::mutex Environment::mutex;

// Random generator
std::mt19937_64 Environment::random_generator;

// Primary context of each GPU.
std::map<int, std::uintptr_t> Environment::primary_ctx;

}  // namespace merlin
