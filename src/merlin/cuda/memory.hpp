// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_MEMORY_HPP_
#define MERLIN_CUDA_MEMORY_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <tuple>    // std::tuple
#include <utility>  // std::exchange

#include "merlin/cuda/declaration.hpp"  // merlin::cuda::Memory
#include "merlin/cuda_interface.hpp"    // __cudevice__

// CUDA nvcc guard
#ifndef __NVCC__
    #error "Compile with CUDA compiler nvcc to use memory allocation tool."
#endif

namespace merlin {

/** @brief CUDA memory allocator.
 *  @details This template serves as a helper for allocating and freeing CUDA memory with a simple syntax. Template
 *  parameters class ``T`` must be either trivially copyable (i.e. object can be copied to a new address using directly
 *  ``std::memcpy``), or support the following methods:
 *     - ``cumalloc_size(void) -> std::uint64_t``: return the memory size (in bytes) needed for the object itself and
 *       data pointed by its members.
 *     - ``copy_to_gpu(T * gpu_ptr, void * data_ptr, std::uintptr_t) -> void *``: copy object to the address (on GPU)
 *       ``gpu_ptr``, copy data pointed by object to ``data_ptr``, and perform the copy action on a CUDA stream casted
 *       to ``std::uintptr_t``. Pointer to the end of the copied data is returned.
 *     - ``copy_by_block(T * dest_ptr, void * data_ptr, std::uint64_t thread_idx, std::uint64_t block_size) -> void *``:
 *       similar to ``copy_to_gpu``, but this function will copy an object on GPU to another memory address also on GPU
 *       (the destination can be on global, shared or thread local mempry).
 */
template <typename... Args>
class cuda::Memory {
  public:
    /** @brief Default constructor.*/
    Memory(void) = default;
    /** @brief Constructor from classes.*/
    Memory(std::uintptr_t stream_ptr, const Args &... args);

    /** @brief Copy constructor (deleted).*/
    Memory(const cuda::Memory<Args...> & src) = delete;
    /** @brief Copy assignment (deleted).*/
    cuda::Memory<Args...> & operator=(const cuda::Memory<Args...> & src) = delete;
    /** @brief Move constructor (deleted).*/
    Memory(cuda::Memory<Args...> && src) = delete;
    /** @brief Move assignment (deleted).*/
    cuda::Memory<Args...> & operator=(cuda::Memory<Args...> && src) = delete;

    /** @brief Get total malloc size.*/
    constexpr std::uint64_t get_total_malloc_size(void) noexcept { return this->total_malloc_size_; }
    /** @brief Get GPU pointer to element.*/
    template <std::uint64_t index>
    typename std::tuple_element<index, std::tuple<Args *...>>::type get(void);

    /** @brief Disown the memory.
     *  @details Release the pointer to GPU data, and the internal pointer is set to ``nullptr``. After calling this
     *  method, GPU pointers contained by the object will no longer be valid anymore.
     */
    void * disown(void) noexcept { return std::exchange(this->gpu_ptr_, nullptr); }

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
    std::tuple<Args *...> type_ptr_;
};

namespace cuda {

/** @brief Copy a list of object to GPU shared memory.
 *  @details This function provides a simple compile-time API function for call the method ``copy_by_block`` of each
 *  object in the list.
 *  @param share_ptr Pointer to available shared memory.
 *  @param args List of objects to be copied to shared memory.
 *  @returns A tuple of pointers. The first pointer points to the address of available free data after all objects
 *  have been copied. The rest is the list of pointers to copied objects on shared memory.
 */
template <typename... Args>
__cudevice__ std::tuple<void *, Args *...> copy_objects(void * share_ptr, const Args &... args);

}  // namespace cuda

}  // namespace merlin

#include "merlin/cuda/memory.tpp"

#endif  // MERLIN_CUDA_MEMORY_HPP_
