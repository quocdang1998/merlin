// Copyright quocdang1998
#include "merlin/env.hpp"

#include <cstdio>

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Default constructor
Environment::Environment(void) {
    Environment::num_instances++;
    if (Environment::is_initialized) {
        return;
    }
    initialize_cuda_context();
    Environment::is_initialized = true;
}

// Destructor
Environment::~Environment(void) {
    Environment::num_instances--;
    if (Environment::num_instances == 0) {
        alarm_cuda_error();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Class bounded properties
// ---------------------------------------------------------------------------------------------------------------------

// Check if environment is initialized
bool Environment::is_initialized = false;

// Number of instances
std::atomic_uint Environment::num_instances = 0;

// Mutex
std::mutex Environment::mutex;

// Random generator
std::mt19937_64 Environment::random_generator;

// ---------------------------------------------------------------------------------------------------------------------
// CUDA environment
// ---------------------------------------------------------------------------------------------------------------------

// Default CUDA kernel block size
std::uint64_t Environment::default_block_size = 64;

#ifndef __MERLIN_CUDA__

// Initialize CUDA context
void initialize_cuda_context(void) {}

// Alarm for CUDA error
void alarm_cuda_error(void) {}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Default environment instance
// ---------------------------------------------------------------------------------------------------------------------

// Default environment
Environment default_environment;

}  // namespace merlin
