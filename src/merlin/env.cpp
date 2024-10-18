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

// Flag indicating if the package was compiled using CUDA
#if defined(__MERLIN_CUDA__)
bool Environment::use_cuda = true;
#else
bool Environment::use_cuda = false;
#endif  // __MERLIN_CUDA__

// Primary context of each GPU.
std::map<int, std::uintptr_t> Environment::primary_ctx;

}  // namespace merlin
