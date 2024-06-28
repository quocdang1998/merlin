// Copyright quocdang1998
#include "merlin/env.hpp"

#include <cuda.h>  // ::cuInit, CUresult

#include "merlin/color.hpp"   // merlin::cout_terminal, merlin::cuprintf_terminal
#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Check error from CUDA driver API and throw
static inline void check_error(::CUresult err) {
    if (err != 0) {
        const char * error_message;
        ::cuGetErrorName(err, &error_message);
        Fatal<cuda_runtime_error>("Driver initialization failed with error \"%s\".\n", error_message);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CUDA environment
// ---------------------------------------------------------------------------------------------------------------------

// Initialize CUDA context
void Environment::init_cuda(void) {
    // skip if the context has already been defined
    if (Environment::is_cuda_initialized) {
        return;
    }
    // driver error check
    ::CUresult driver_err;
    // driver initialization
    driver_err = ::cuInit(0U);
    check_error(driver_err);
    // get number of CUDA capable GPUs and check if it is not zero
    int num_gpu;
    driver_err = ::cuDeviceGetCount(&num_gpu);
    check_error(driver_err);
    if (num_gpu == 0) {
        Warning("No GPU was found. Return empty CUDA context (GPU functions will have no effect).\n");
        return;
    }
#if __CUDACC_VER_MAJOR__ < 12
    // initialize primary context for each GPU
    for (int i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        ::cudaInitDevice(i_gpu, 0U, 0);
    }
#endif  // __CUDACC_VER_MAJOR__ < 12
    Environment::is_cuda_initialized = true;
}

// Throw an error if CUDA environment has not been initialized
void check_cuda_env(void) {
    if (!Environment::is_cuda_initialized) {
        Fatal<cuda_runtime_error>("Invoking merlin::Environment::init_cuda before using any CUDA functionality.\n");
    }
}

}  // namespace merlin
