// Copyright quocdang1998
#include "merlin/cuda_interface.hpp"

#include "merlin/logger.hpp"  // merlin::cuda_runtime_error, FAILURE

namespace merlin {

// Allocate memory on the current GPU
void * cuda_mem_alloc(std::uint64_t size, std::uint64_t stream_ptr) {
    void * result;
    ::cudaError_t err_ = ::cudaMallocAsync(&result, size, reinterpret_cast<::cudaStream_t>(stream_ptr));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "CUDA asynchronous memory allocation failed with error \"%s\"",
                ::cudaGetErrorString(err_));
    }
    return result;
}

// Copy data from CPU to GPU
void cuda_mem_cpy_host_to_device(void * destination, void * source, std::uint64_t size, std::uint64_t stream_ptr) {
    ::cudaError_t err_ = ::cudaMemcpyAsync(destination, source, size, ::cudaMemcpyHostToDevice,
                                           reinterpret_cast<::cudaStream_t>(stream_ptr));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "CUDA asynchronous copy data to GPU failed with error \"%s\"",
                ::cudaGetErrorString(err_));
    }
}

// Copy data from GPU to CPU
void cuda_mem_cpy_device_to_host(void * destination, void * source, std::uint64_t size, std::uint64_t stream_ptr) {
    ::cudaError_t err_ = ::cudaMemcpyAsync(destination, source, size, ::cudaMemcpyDeviceToHost,
                                           reinterpret_cast<::cudaStream_t>(stream_ptr));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "CUDA asynchronous copy data to GPU failed with error \"%s\"",
                ::cudaGetErrorString(err_));
    }
}

// Deallocate memory on the global memory space of the current GPU.
void cuda_mem_free(void * ptr, std::uint64_t stream_ptr) {
    ::cudaError_t err_ = ::cudaFreeAsync(ptr, reinterpret_cast<::cudaStream_t>(stream_ptr));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "CUDA asynchronous memory deallocation failed with error \"%s\"",
                ::cudaGetErrorString(err_));
    }
}

}  // namespace merlin
