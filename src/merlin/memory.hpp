// Copyright 2024 quocdang1998
#ifndef MERLIN_MEMORY_HPP_
#define MERLIN_MEMORY_HPP_

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uintptr_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

// CPU Memory Allocator
// --------------------

/** @brief Allocate page-locked CPU memory.
 *  @details Allocate a paged-locked memory. The memory page is pinned into the physical RAM to prevent it from being
 *  swapped out to disk by the virtual memory manager of the operating system. This is crucial for high-performance
 *  computing and systems that require asynchronous data transfers, such as between CPU and GPUs.
 *  
 *  In non-CUDA configuration, this function degenerates to the ``new`` operator.
 *  @note In both mode, the allocated memory must be freed manually.
 *  @param size Size in bytes.
 */
MERLIN_EXPORTS void * mem_alloc_host(std::size_t size);

/** @brief Free page-locked CPU memory.
 *  @details Deallocate a paged-locked memory.
 * 
 *  In non-CUDA configuration, this function degenerates to the ``delete`` operator.
 *  @note Call this function on memory allocated by ``mem_alloc_host``.
 *  @param ptr Pointer to the allocated memory.
 */
MERLIN_EXPORTS void mem_free_host(void * ptr);

/** @brief Pin a pre-allocated CPU memory.
 *  @details Register the memory pages associated an pre-allocated memory as "pinned". These pages will not be
 *  swapped out of the system memory (RAM) even when the memory is full.
 * 
 *  If the memory pinning process has successfully been executed, a ``true`` value is returned.
 *  
 *  In non-CUDA configuration, this function does nothing, because memory pinning is not necessary.
 *  @note The pinned memory must be unregistered.
 *  @param ptr Pointer to the allocated memory. Only the pointer to the first element of the allocated memory is
 *  eligible to this mapping.
 *  @param size Size in bytes.
 */
MERLIN_EXPORTS bool mem_register_host(void * ptr, std::size_t size);

/** @brief Unpin a pre-allocated CPU memory.
 *  @details Unregister a pinned memory. The memory pages can be swapped out of the system memory (RAM), similar to the
 *  default setting of the current process.
 * 
 *  If the memory unpinning process has successfully been executed, a ``true`` value is returned.
 *  
 *  In non-CUDA configuration, this function does nothing, because memory pinning is not necessary.
 *  @note Call this function on memory registered by ``mem_register_host``.
 *  @param ptr Pointer to the allocated memory. Only the pointer to the first element of the allocated memory is
 *  eligible to this un-mapping.
 */
MERLIN_EXPORTS bool mem_unregister_host(void * ptr);

// GPU Memory Allocator
// --------------------

/** @brief Asynchronously allocate memory on the current GPU.
 *  @details Asynchronously allocate memory on the global memory of the GPU attached to the main thread.
 *
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @note The allocated memory must be freed manually using ``mem_free_device``.
 *  @param ptr Reference to the pointer to allocated memory.
 *  @param size Size in bytes.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void mem_alloc_device(void ** ptr, std::size_t size, std::uintptr_t stream_ptr = 0);

/** @brief Asynchronously deallocate memory on the current GPU.
 *  @details Asynchronously deallocate memory on the global memory of the GPU attached to the main thread.
 * 
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @param ptr Pointer to the allocated memory. Only the pointer to the first element of the allocated memory is
 *  eligible to this mapping.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void mem_free_device(void * ptr, std::uintptr_t stream_ptr = 0);

/** @brief Asynchronously deallocate memory on the current GPU (without exception).
 *  @details Asynchronously deallocate memory on the global memory of the GPU attached to the main thread.
 * 
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @param ptr Pointer to the allocated memory. Only the pointer to the first element of the allocated memory is
 *  eligible to this mapping.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void mem_free_device_noexcept(void * ptr, std::uintptr_t stream_ptr = 0) noexcept;

// CPU-GPUs Data Transfer
// ----------------------

/** @brief Asynchronously copy data from CPU to GPU.
 *  @details Asynchronously copy data from RAM to GPU global memory.
 * 
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @note In order to enhance the success of the data transferring, these two requirements must be met:
 *  - The target GPU where the destination resides must be the one associated to the current CUDA context.
 *  - The source must be pinned to the physical memory, either through ``mem_alloc_host`` or ``mem_register_host``.
 *  @param dest Pointer to destination (on GPU global memory).
 *  @param src Pointer to source (on CPU main memory).
 *  @param size Size in bytes.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void memcpy_cpu_to_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr = 0);

/** @brief Asynchronously copy data from GPU to CPU.
 *  @details Asynchronously copy data from GPU global memory to RAM.
 * 
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @note In order to enhance the success of the data transferring, these two requirements must be met:
 *  - The target GPU where the destination resides must be the one associated to the current CUDA context.
 *  - The source must be pinned to the physical memory, either through ``mem_alloc_host`` or ``mem_register_host``.
 *  @param dest Pointer to destination (on CPU main memory).
 *  @param src Pointer to source (on GPU global memory).
 *  @param size Size in bytes.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void memcpy_gpu_to_cpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr = 0);

/** @brief Asynchronously copy data within the same GPU.
 *  @details Asynchronously copy data between two locations on a GPU global memory. Both the source and the destination
 *  reside on the same GPU.
 *
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @note In order to enhance the success of the data transferring, the target GPU where the destination and the source
 *  reside must be the one associated to the current CUDA context.
 *  @param dest Pointer to destination (on GPU global memory).
 *  @param src Pointer to source (on GPU global memory).
 *  @param size Size in bytes.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void memcpy_gpu(void * dest, const void * src, std::size_t size, std::uintptr_t stream_ptr = 0);

/** @brief Asynchronously copy data between two GPUs.
 *  @details Asynchronously copy data between two different GPUs.
 *
 *  This function will throw an exception of type ``cuda_compile_error`` when compiling in non-CUDA configuration.
 *  @param dest Pointer to destination (on GPU global memory).
 *  @param src Pointer to source (on GPU global memory).
 *  @param size Size in bytes.
 *  @param dest_gpu ID of the destination GPU.
 *  @param src_gpu ID of the source GPU.
 *  @param stream_ptr CUDA Stream pointer. The null stream corresponds to a synchronous copy wrt. the main thread.
 */
MERLIN_EXPORTS void memcpy_peer_gpu(void * dest, const void * src, std::size_t size, int dest_gpu, int src_gpu,
                                    std::uintptr_t stream_ptr = 0);

}  // namespace merlin

// #include "merlin/memory.tpp"

#endif  // MERLIN_MEMORY_HPP_
