// Copyright quocdang1998
#include "merlin/env.hpp"

#include <cuda.h>  // ::cuInit

#include "merlin/color.hpp"     // merlin::cout_terminal, merlin::cuprintf_terminal
#include "merlin/logger.hpp"    // merlin::Fatal, merlin::Warning

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
        Fatal<cuda_runtime_error>("Get device failed with error \"%s\"\n", ::cudaGetErrorString(err_));
    }
    // return empty context if no GPU was found
    if (num_gpu == 0) {
        Warning("No GPU was found. Return empty CUDA context (GPU functions will have no effect).\n");
        return;
    }
}

// Alarm for CUDA error
void alarm_cuda_error(void) {
    // check for any CUDA error
    ::cudaError_t err_ = ::cudaPeekAtLastError();
    if (err_ != 0) {
        Warning("A CUDA error has occurred somewhere in the program with message \"%s\"", ::cudaGetErrorString(err_));
    }
}

}  // namespace merlin
