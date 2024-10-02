// Copyright 2024 quocdang1998
#ifndef MERLIN_CUDA_COPY_HELPERS_HPP_
#define MERLIN_CUDA_COPY_HELPERS_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <tuple>    // std::tuple
#include <utility>  // std::exchange

#include "merlin/config.hpp"            // __cudevice__
#include "merlin/cuda/declaration.hpp"  // merlin::cuda::Dispatcher

namespace merlin {

// Helper classes for transferring data
// ------------------------------------

/** @brief Automatic CUDA memory allocator and transfer for a tuple of classes.
 *  @details This template serves as a helper for allocating and freeing CUDA memory with a simple syntax.
 *  
 *  Each element of the variadic template ``Args`` must be either trivially copyable (i.e. object can be copied to a new
 *  address using directly ``std::memcpy``), or support the cloning interface as follows:
 *  - ``cumalloc_size(void) -> std::uint64_t``: return the memory size (in bytes) needed for the object itself and data
 *    pointed by its members.
 *  - ``copy_to_gpu(T * gpu_ptr, void * data_ptr, std::uintptr_t) -> void *``: copy object to the address (on GPU)
 *    ``gpu_ptr``, copy data pointed by object to ``data_ptr``, and perform the copy action on a CUDA stream casted to
 *    ``std::uintptr_t``. Pointer to the end of the copied data is returned.
 *  - ``copy_by_block(T * dest_ptr, void * data_ptr, std::uint64_t thread_idx, std::uint64_t block_size) -> void *``:
 *    similar to ``copy_to_gpu``, but this function will copy an object on GPU to another memory address also on GPU
 *    (the destination can be on either the global or the shared memory).
 */
template <typename... Args>
class cuda::Dispatcher {
  public:
    /** @brief Default constructor.*/
    Dispatcher(void) = default;
    /** @brief Constructor from classes.*/
    Dispatcher(std::uintptr_t stream_ptr, const Args &... args);

    /** @brief Copy constructor (deleted).*/
    Dispatcher(const cuda::Dispatcher<Args...> & src) = delete;
    /** @brief Copy assignment (deleted).*/
    cuda::Dispatcher<Args...> & operator=(const cuda::Dispatcher<Args...> & src) = delete;
    /** @brief Move constructor (deleted).*/
    Dispatcher(cuda::Dispatcher<Args...> && src) = delete;
    /** @brief Move assignment (deleted).*/
    cuda::Dispatcher<Args...> & operator=(cuda::Dispatcher<Args...> && src) = delete;

    /** @brief Get total malloc size.*/
    constexpr std::uint64_t get_total_malloc_size(void) noexcept { return this->total_malloc_size_; }
    /** @brief Get GPU pointer to element.*/
    template <std::uint64_t index>
    typename std::tuple_element<index, std::tuple<Args *...>>::type get(void);

    /** @brief Disown the memory on GPU, but not deallocate it.
     *  @details Release the pointer to GPU data, and the internal pointer is set to ``nullptr``. After calling this
     *  method, GPU pointers contained by the object will no longer be valid anymore, but the memory is not freed.
     */
    void * disown(void) noexcept { return std::exchange(this->gpu_ptr_, nullptr); }

    /** @brief Destructor.*/
    ~Dispatcher(void);

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

// Copy classes by CUDA blocks
// ---------------------------

namespace cuda {

#ifdef __NVCC__

/** @brief Copy a list of object to GPU shared memory.
 *  @details This function provides a simple compile-time API function for call the method ``copy_by_block`` of each
 *  object in the list.
 *  @param share_ptr Pointer to available shared memory.
 *  @param thread_idx Index of the current CUDA thread in the block.
 *  @param block_size Number of threads in the current CUDA block.
 *  @param args List of objects to be copied to shared memory.
 *  @returns A tuple of pointers. The first pointer points to the address of available free data after all objects
 *  have been copied. The rest is the list of pointers to copied objects on shared memory.
 */
template <typename... Args>
__cudevice__ std::tuple<void *, Args *...> copy_objects(void * share_ptr, const std::uint64_t & thread_idx,
                                                        const std::uint64_t & block_size, const Args &... args);

#endif  // __NVCC__

}  // namespace cuda

}  // namespace merlin

#include "merlin/cuda/copy_helpers.tpp"

#endif  // MERLIN_CUDA_COPY_HELPERS_HPP_
