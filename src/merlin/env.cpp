// Copyright quocdang1998
#include "merlin/env.hpp"

#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_WINDOWS__)
    #include <io.h>     // ::_isatty
    #include <stdio.h>  // ::_fileno
#elif defined(__MERLIN_LINUX__)
    #include <unistd.h>  // ::fileno, ::isatty
#endif

namespace merlin {

// Default constructor
Environment::Environment(void) {
    // increase number of instances
    Environment::num_instances++;
    if (Environment::is_initialized) {
        return;
    }
    // get redirection
#if defined(__MERLIN_WINDOWS__)
    Environment::cout_terminal = ::_isatty(::_fileno(stdout));
    Environment::cerr_terminal = ::_isatty(::_fileno(stderr));
#elif defined(__MERLIN_LINUX__)
    Environment::cout_terminal = ::isatty(::fileno(stdout));
    Environment::cerr_terminal = ::isatty(::fileno(stderr));
#endif
    // initialize CUDA context
    initialize_cuda_context();
    Environment::is_initialized = true;
}

// Destructor
Environment::~Environment(void) {
    // decrease number of instances
    Environment::num_instances--;
    if (Environment::num_instances == 0) {
        alarm_cuda_error();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Class bounded properties
// ---------------------------------------------------------------------------------------------------------------------

// Check if ``stdout`` is redirected into a file
bool Environment::cout_terminal;

// Check if ``stderr`` is redirected into a file
bool Environment::cerr_terminal;

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
