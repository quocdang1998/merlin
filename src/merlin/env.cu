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
void Environment::init_cuda(int default_gpu) {
    // skip if the context has already been defined
    if (Environment::is_cuda_initialized) {
        return;
    }
    // get number of CUDA capable GPUs
    int num_gpu;
    ::cudaError_t err_;
    err_ = ::cudaGetDeviceCount(&num_gpu);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get device failed with error \"%s\"\n", ::cudaGetErrorString(err_));
    }
    if (num_gpu == 0) {
        Warning("No GPU was found. Return empty CUDA context (GPU functions will have no effect).\n");
        return;
    }
    // default settings for each GPU
    for (int i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        // initialize device
        err_ = ::cudaInitDevice(i_gpu, 0U, 0U);
        if (err_ != 0) {
            Fatal<cuda_runtime_error>("Initialize device %d failed with error \"%s\"\n", i_gpu,
                                      ::cudaGetErrorString(err_));
        }
    }
    // set current device
    err_ = ::cudaSetDevice(default_gpu);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Set device failed with error \"%s\"\n", ::cudaGetErrorString(err_));
    }
    // set variable
    Environment::is_cuda_initialized = true;
}

// Throw an error if CUDA environment has not been initialized
void check_cuda_env(void) {
    if (!Environment::is_cuda_initialized) {
        Fatal<cuda_runtime_error>("Invoking merlin::Environment::init_cuda before using any CUDA functionality.\n");
    }
}

}  // namespace merlin
