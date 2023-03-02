// Copyright quocdang1998
#include "merlin/env.hpp"

#include <cuda.h>  // ::cuCtxGetCurrent

#include "merlin/logger.hpp"

namespace merlin {

// Initialize CUDA context
void initialize_cuda_context(void) {
    // check for number of GPU
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
    // try to get current context
    ::CUcontext current_ctx;
    err_ = static_cast<::cudaError_t>(::cuCtxGetCurrent(&current_ctx));
    if ((err_ != 3) && (err_ != 0)) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    // initialized case (return success)
    if ((err_ == 0) && (current_ctx != nullptr)) {
        return;
    }
    // uninitialized or undefined
    for (int i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        err_ = ::cudaSetDevice(i_gpu);
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Set current GPU of ID %d failed with message \"%s\".\n",
                    i_gpu, ::cudaGetErrorString(err_));
        }
        ::cuCtxGetCurrent(&current_ctx);
        Environment::primary_contexts[i_gpu] = reinterpret_cast<std::uint64_t>(current_ctx);
    }
    // set back to default device
    ::cudaSetDevice(Environment::default_gpu);
}

}  // namespace merlin
