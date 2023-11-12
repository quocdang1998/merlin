// Copyright quocdang1998
#include "merlin/env.hpp"

#include <cuda.h>  // ::cuInit

#include "merlin/logger.hpp"    // FAILURE
#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// CUDA environment
// ---------------------------------------------------------------------------------------------------------------------

// Initialize CUDA context
void initialize_cuda_context(void) {
    // initialize context
    ::cuInit(0);
    // get number of CUDA capable GPUs
    int num_gpu;
    ::cudaError_t err_ = ::cudaGetDeviceCount(&num_gpu);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get device failed with error \"%s\"\n", ::cudaGetErrorString(err_));
    }
    // return empty context if no GPU was found
    if (num_gpu == 0) {
        WARNING("No GPU was found. Return empty CUDA context (GPU functions will have no effect).\n");
        return;
    }
}

// Alarm for CUDA error
void alarm_cuda_error(void) {
    // check for any CUDA error
    ::cudaError_t err_ = ::cudaPeekAtLastError();
    if (err_ != 0) {
        WARNING("A CUDA error has occurred somewhere in the program with message \"%s\"", ::cudaGetErrorString(err_));
    }
}

}  // namespace merlin
