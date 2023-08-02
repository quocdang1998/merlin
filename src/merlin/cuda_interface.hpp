// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_INTERFACE_HPP_
#define MERLIN_CUDA_INTERFACE_HPP_

#include <cstdint>  // std::uint64_t
#include <memory>   // std::unique_ptr

#include "merlin/exports.hpp"  // MERLINSHARED_EXPORTS

// CUDA decorator expansion when not compiling with nvcc
#ifdef __NVCC__
    #define __cuhost__ __host__
    #define __cudevice__ __device__
    #define __cuhostdev__ __host__ __device__
#else
    #define __cuhost__
    #define __cudevice__
    #define __cuhostdev__
#endif

namespace merlin {

/** @brief Allocate memory on the current GPU.
 *  @details CUDA allocation wrapper for Python interface.
 *  @param size Number of bytes to allocate in the memory.
 *  @param stream_ptr CUDA stream performing the allocation.
 */
MERLINSHARED_EXPORTS void * cuda_mem_alloc(std::uint64_t size, std::uint64_t stream_ptr = 0);

/** @brief Copy data from CPU to GPU.
 *  @details CUDA memcpy wrapper for Python interface.
 *  @param destination Pointer to memomry on GPU.
 *  @param source Pointer to memory on CPU.
 *  @param size Number of bytes to transfer to GPU.
 *  @param stream_ptr CUDA stream performing the allocation.
 */
MERLINSHARED_EXPORTS void cuda_mem_cpy_host_to_device(void * destination, void * source, std::uint64_t size,
                                                      std::uint64_t stream_ptr = 0);

/** @brief Copy data from GPU to CPU.
 *  @details CUDA memcpy wrapper for Python interface.
 *  @param destination Pointer to memomry on GPU.
 *  @param source Pointer to memory on CPU.
 *  @param size Number of bytes to transfer to CPU.
 *  @param stream_ptr CUDA stream performing the allocation.
 */
MERLINSHARED_EXPORTS void cuda_mem_cpy_device_to_host(void * destination, void * source, std::uint64_t size,
                                                      std::uint64_t stream_ptr = 0);

/** @brief CUDA memory deleter.*/
class CudaDeleter {
  public:
    /// @name Constructor and destructor
    /// @{
    /** @brief Default constructor.*/
    CudaDeleter(void) = default;
    /** @brief Copy constructor*/
    CudaDeleter(const CudaDeleter & src) = default;
    /** @brief Copy assignment.*/
    CudaDeleter & operator=(const CudaDeleter & src) = default;
    /** @brief Destructor.*/
    ~CudaDeleter(void) = default;
    /// @}

    /// @name Call fucntion
    /// @{
    /** @brief Call CUDA deallocation (synchronous) on pointer.*/
    MERLINSHARED_EXPORTS void operator()(void * pointer);
    /// @}
};

/** @brief CUDA pointer to memory deallocate itself as destruction.*/
using CudaPtrWrapper = std::unique_ptr<void, CudaDeleter>();

}  // namespace merlin

#endif  // MERLIN_CUDA_INTERFACE_HPP_
