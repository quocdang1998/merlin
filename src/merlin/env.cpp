// Copyright quocdang1998
#include "merlin/env.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Environment properties
// ---------------------------------------------------------------------------------------------------------------------

// Mutex
std::mutex Environment::mutex;

// Random generator
std::mt19937_64 Environment::random_generator;

// ---------------------------------------------------------------------------------------------------------------------
// CUDA environment
// ---------------------------------------------------------------------------------------------------------------------

// CUDA context checker
bool Environment::is_cuda_initialized = false;

#ifndef __MERLIN_CUDA__

// Initialize CUDA context
void Environment::init_cuda(int default_gpu) {}

// Throw an error if CUDA environment has not been initialized
void check_cuda_env(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
