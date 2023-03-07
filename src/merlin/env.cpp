// Copyright quocdang1998
#include "merlin/env.hpp"

namespace merlin {

// Default constructor
Environment::Environment(void) {
    if (Environment::is_initialized) {
        return;
    }
    initialize_cuda_context();
    Environment::is_initialized = true;
}

// --------------------------------------------------------------------------------------------------------------------
// Class bounded properties
// --------------------------------------------------------------------------------------------------------------------

// Check if environment is initialized
bool Environment::is_initialized = false;

// Mutex
std::mutex Environment::mutex;

// --------------------------------------------------------------------------------------------------------------------
// Array allocation limit
// --------------------------------------------------------------------------------------------------------------------

// Size in bytes of maximum allowed allocated memory
std::uint64_t Environment::cpu_mem_limit = static_cast<std::uint64_t>(20) << 30;

// --------------------------------------------------------------------------------------------------------------------
// CPU Parallelism
// --------------------------------------------------------------------------------------------------------------------

// Minimum size over which the loop is parallelized
std::uint64_t Environment::parallel_chunk = static_cast<std::uint64_t>(96);

// --------------------------------------------------------------------------------------------------------------------
// CUDA environment
// --------------------------------------------------------------------------------------------------------------------

// ID of default GPU
int Environment::default_gpu = 0;

// Constructor from elements
Environment::ContextAttribute::ContextAttribute(std::uint64_t ref_count, int gpu_id) :
reference_count(ref_count), gpu(gpu_id) {}

// Copy constructor
Environment::ContextAttribute::ContextAttribute(const Environment::ContextAttribute & src) :
reference_count(src.reference_count.load()), gpu(src.gpu) {}

// Copy assignment
Environment::ContextAttribute & Environment::ContextAttribute::operator=(const Environment::ContextAttribute & src) {
    this->reference_count.store(src.reference_count.load());
    this->gpu = src.gpu;
    return *this;
}

// Attributes of instances
std::map<std::uintptr_t, Environment::ContextAttribute> Environment::attribute;

// CUDA primary contexts
std::map<int, std::uintptr_t> Environment::primary_contexts;

// Default CUDA kernel block size
std::uint64_t Environment::default_block_size = 64;

#ifndef __MERLIN_CUDA__

// Initialize CUDA context
void initialize_cuda_context(void) {}

#endif  // __MERLIN_CUDA__

// --------------------------------------------------------------------------------------------------------------------
// Default environment instance
// --------------------------------------------------------------------------------------------------------------------

// Default environment
Environment default_environment;


}  // namespace merlin
