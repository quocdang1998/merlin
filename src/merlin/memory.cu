// Copyright 2024 quocdang1998
#include "merlin/memory.hpp"

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Throw CUDA error
static inline void check_cuda_error(::cudaError_t error, const std::string & step_name) {
    if (error != 0) {
        Fatal<cuda_runtime_error>("{} failed with error \"{}\".\n", step_name, ::cudaGetErrorString(error));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU Memory Allocator
// ---------------------------------------------------------------------------------------------------------------------

// Allocate page-locked CPU memory
void * mem_alloc_host(std::size_t size) {
    void * allocated_mem;
    check_cuda_error(::cudaMallocHost(&allocated_mem, size), "Malloc page-locked memory");
    return allocated_mem;
}

// Free page-locked CPU memory
void mem_free_host(void * ptr) { check_cuda_error(::cudaFreeHost(ptr), "Free page-locked memory"); }

// Pin a pre-allocated CPU memory
bool mem_register_host(void * ptr, std::size_t size) {
    ::cudaError_t error = ::cudaHostRegister(ptr, size, 0);
    if (error == ::cudaErrorAlreadyMapped) {
        Warning("The memory has already been mapped. No registration of memory pages was perfermed.\n");
        return false;
    }
    check_cuda_error(error, "Register page-locked memory");
    return true;
}

// Unpin a pre-allocated CPU memory
bool mem_unregister_host(void * ptr) {
    ::cudaError_t error = ::cudaHostUnregister(ptr);
    if (error == ::cudaErrorNotMapped) {
        Warning("The memory has not been mapped. No unregistration operation was performed.\n");
        return false;
    }
    check_cuda_error(error, "Unregister page-locked memory");
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU Memory Allocator
// ---------------------------------------------------------------------------------------------------------------------

// Asynchronously allocate memory on the current GPU
void mem_alloc_device(void ** ptr, std::size_t size, std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaMallocAsync(ptr, size, cuda_stream), "Asynchronous memory allocation on GPU");
}

// Asynchronously deallocate memory on the current GPU
void mem_free_device(void * ptr, std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaFreeAsync(ptr, cuda_stream), "Asynchronous memory deallocation on GPU");
}

// Asynchronously deallocate memory on the current GPU (without exception)
void mem_free_device_noexcept(void * ptr, std::uintptr_t stream_ptr) noexcept {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaFreeAsync(ptr, cuda_stream);
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU-GPUs Data Transfer
// ---------------------------------------------------------------------------------------------------------------------

// Asynchronously copy data from CPU to GPU
void memcpy_cpu_to_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaMemcpyAsync(dest, src, size, ::cudaMemcpyHostToDevice, cuda_stream),
                     "Asynchronous memcpy from CPU to GPU");
}

// Asynchronously copy data from GPU to CPU
void memcpy_gpu_to_cpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaMemcpyAsync(dest, src, size, ::cudaMemcpyDeviceToHost, cuda_stream),
                     "Asynchronous memcpy from GPU to CPU");
}

// Asynchronously copy data between two locations on the global memory of a GPU
void memcpy_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaMemcpyAsync(dest, src, size, ::cudaMemcpyDeviceToDevice, cuda_stream),
                     "Asynchronous memcpy within a GPU");
}

// Asynchronously copy data between two GPUs
void memcpy_peer_gpu(void * dest, const void * src, std::size_t size, int dest_gpu, int src_gpu,
                     std::uintptr_t stream_ptr) {
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    check_cuda_error(::cudaMemcpyPeerAsync(dest, dest_gpu, src, src_gpu, size, cuda_stream),
                     "Asynchronous memcpy between two GPUs");
}

}  // namespace merlin
