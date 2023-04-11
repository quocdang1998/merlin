// Copyright quocdang1998
#include "merlin/cuda_interface.hpp"

#include "merlin/logger.hpp"  // merlin::cuda_compile_error, FAILURE

namespace merlin {

#ifndef __MERLIN_CUDA__

// Allocate memory on the current GPU
void * cuda_mem_alloc(std::uint64_t size, std::uint64_t stream_ptr) {
    FAILURE(cuda_compile_error, "Compile the library with CUDA option to enable memory allocation on GPU.\n");
    return nullptr;
}

// Copy data from CPU to GPU
void cuda_mem_cpy_host_to_device(void * destination, void * source, std::uint64_t size, std::uint64_t stream_ptr) {
    FAILURE(cuda_compile_error, "Compile the library with CUDA option to enable data transfering to GPU.\n");
}

// Copy data from GPU to CPU
void cuda_mem_cpy_device_to_host(void * destination, void * source, std::uint64_t size, std::uint64_t stream_ptr) {
    FAILURE(cuda_compile_error, "Compile the library with CUDA option to enable data transfering from GPU.\n");
}

// Call CUDA deallocation on pointer
void CudaDeleter::operator()(void * pointer) {
    FAILURE(cuda_compile_error, "Compile the library with CUDA option to enable memory deallocation on GPU.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
