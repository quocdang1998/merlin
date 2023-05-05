// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_MEMORY_HPP_
#define MERLIN_CUDA_MEMORY_HPP_

#include <array>  // std::array
#include <cstdint>  // std::uint64_t
#include <tuple>  // std::tuple

#include "merlin/cuda_decorator.hpp"  // __cudevice__

// CUDA nvcc guard
#ifndef __NVCC__
#error "Compile with CUDA compiler nvcc to use memory allocation tool."
#endif

namespace merlin::cuda {
template <typename ... Args>
class Memory;
}  // namespace merlin::cuda

namespace merlin {

/** @brief CUDA memory allocator.
 *  @details Copy metadata of classes to GPU.
 */
template <typename ... Args>
class cuda::Memory {
  public:
    /** @brief Default constructor.*/
    Memory(void) = default;
    /** @brief Constructor from classes.*/
    Memory(std::uintptr_t stream_ptr, const Args & ... args);

    /** @brief Copy constructor (deleted).*/
    Memory(const cuda::Memory<Args ...> & src) = delete;
    /** @brief Copy assignment (deleted).*/
    cuda::Memory<Args ...> & operator=(const cuda::Memory<Args ...> & src) = delete;
    /** @brief Move constructor (deleted).*/
    Memory(cuda::Memory<Args ...> && src) = delete;
    /** @brief Move assignment (deleted).*/
    cuda::Memory<Args ...> & operator=(cuda::Memory<Args ...> && src) = delete;

    /** @brief Get total malloc size.*/
    constexpr std::uint64_t get_total_malloc_size(void) noexcept {return this->total_malloc_size_;}
    /** @brief Get GPU pointer to element.*/
    template <std::uint64_t index>
    typename std::tuple_element<index, std::tuple<Args * ...>>::type get(void);

    /** @brief Defer the CUDA free on pointer for asynchronous launch on GPU.*/
    void defer_allocation(void);

    /** @brief Destructor.*/
    ~Memory(void);

  protected:
    /** @brief Allocated pointers to GPU memory.*/
    void * gpu_ptr_ = nullptr;
    /** @brief Array of pointers to GPU objects.*/
    std::array<std::uintptr_t, sizeof...(Args)> offset_;
    /** @brief Total malloc size.*/
    std::uint64_t total_malloc_size_;
    /** @brief CUDA stream pointer.*/
    std::uintptr_t stream_ptr_ = 0;
    /** @brief Tuple of pointers to elements for storing class type.*/
    std::tuple<Args * ...> type_ptr_;
    /** @brief Deferred deallocation.*/
    bool deferred_dealloc_ = false;
};

namespace cuda {

/** @brief Copy a list of object to GPU shared memory.
 *  @details This function provides a simple compile-time API function for call the method ``copy_to_shared_mem`` of
 *  each object in the list.
 *  @param share_ptr Pointer to available shared memory.
 *  @param args List of objects to be copied to shred memory.
 *  @returns A tuple of pointers. The first pointer points to the address of available free data after all objects
 *  have been copied. The rest is the list of pointers to copied objects on shared memory.
 */
template <typename ... Args>
__cudevice__ std::tuple<void *, Args * ...> copy_class_to_shared_mem(void * share_ptr, const Args & ... args);

}  // namespace cuda

}  // namespace merlin

#include "merlin/cuda/memory.tpp"

#endif  // MERLIN_CUDA_MEMORY_HPP_
