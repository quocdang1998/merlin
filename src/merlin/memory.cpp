// Copyright 2024 quocdang1998
#include "merlin/memory.hpp"

#include <cstddef>  // nullptr
#include <cstdint>  // std::uint64_t

#include "merlin/logger.hpp"  // merlin::cuda_compile_error, merlin::Fatal
#include "merlin/platform.hpp"  // __MERLIN_WINDOWS__, __MERLIN_LINUX__

#if defined(__MERLIN_WINDOWS__)
    #include <malloc.h>  // ::_aligned_malloc, ::_aligned_free
#elif defined(__MERLIN_LINUX__)
    #include <cstdlib>  // std::aligned_alloc, std::free
#endif

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Wrapper for aligned_alloc
// ---------------------------------------------------------------------------------------------------------------------

// Align-alloc
static inline void * aligned_alloc(std::size_t alignment, std::size_t size) {
#if defined(__MERLIN_WINDOWS__)
    return ::_aligned_malloc(size, alignment);
#elif defined(__MERLIN_LINUX__)
    return std::aligned_alloc(alignment, size);
#endif
}

// Align-free
static inline void aligned_free(void * ptr) {
#if defined(__MERLIN_WINDOWS__)
    ::_aligned_free(ptr);
#elif defined(__MERLIN_LINUX__)
    std::free(ptr);
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU Memory Allocator
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Allocate page-locked CPU memory
void * mem_alloc_host(std::size_t size) {
    // allocate memory
    void * allocated_mem = nullptr;
    if (size % sizeof(double) == 0) {
        allocated_mem = aligned_alloc(alignof(double), size);
    } else if (size % sizeof(std::uint64_t) == 0) {
        allocated_mem = aligned_alloc(alignof(std::uint64_t), size);
    } else {
        allocated_mem = aligned_alloc(alignof(char), size);
    }
    // error checking
    if (allocated_mem == nullptr) {
        Fatal<std::bad_alloc>();
    }
    return allocated_mem;
}

// Free page-locked CPU memory
void mem_free_host(void * ptr) {
    // deallocate memory
    aligned_free(ptr);
}

// Pin a pre-allocated CPU memory
bool mem_register_host(void * ptr, std::size_t size) { return false; }

// Unpin a pre-allocated CPU memory
bool mem_unregister_host(void * ptr) { return false; }

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// GPU Memory Allocator
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Asynchronously allocate memory on the current GPU
void mem_alloc_device(void ** ptr, std::size_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot allocate memory on GPU in non-CUDA mode.\n");
}

// Asynchronously deallocate memory on the current GPU
void mem_free_device(void * ptr, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot allocate memory on GPU in non-CUDA mode.\n");
}

// Asynchronously deallocate memory on the current GPU (without exception)
void mem_free_device_noexcept(void * ptr, std::uintptr_t stream_ptr) noexcept {}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// CPU-GPUs Data Transfer
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Asynchronously copy data from CPU to GPU
void memcpy_cpu_to_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot transfer data from CPU to GPU in non-CUDA mode.\n");
}

// Asynchronously copy data from GPU to CPU
void memcpy_gpu_to_cpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot transfer data from GPU to CPU in non-CUDA mode.\n");
}

// Asynchronously copy data between two locations on the global memory of a GPU
void memcpy_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot transfer data within the same GPU in non-CUDA mode.\n");
}

// Asynchronously copy data between two GPUs
void memcpy_peer_gpu(void * dest, const void * src, std::size_t size, int dest_gpu, int src_gpu,
                     std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Cannot transfer data between different GPUs in non-CUDA mode.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
