// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_COPY_HELPERS_TPP_
#define MERLIN_CUDA_COPY_HELPERS_TPP_

#include <concepts>     // std::convertible_to, std::same_as
#include <cstddef>      // std::size_t
#include <type_traits>  // std::is_trivially_copyable

#include "merlin/memory.hpp"  // merlin::mem_alloc_device, merlin::memcpy_cpu_to_gpu, merlin::mem_free_device_noexcept

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------------------------------------------------

template <typename T>
concept HasCuMallocSize = requires(const T & obj) {
    { obj.cumalloc_size() } -> std::convertible_to<std::uint64_t>;
};

template <typename T>
concept HasCopyToGpu = requires(const T & obj, T * gpu_dest, void * object_data, std::uintptr_t stream_ptr) {
    { obj.copy_to_gpu(gpu_dest, object_data, stream_ptr) } -> std::same_as<void *>;
};

template <typename T>
concept HasCopyByBlock = requires(const T & obj, T * dest_ptr, void * data_ptr, std::uint64_t thread_idx,
                                  std::uint64_t block_size) {
    { obj.copy_by_block(dest_ptr, data_ptr, thread_idx, block_size) } -> std::same_as<void *>;
};

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Total malloc size
template <typename T, typename... Args>
requires HasCuMallocSize<T> || std::is_trivially_copyable<T>::value
std::uint64_t total_malloc_size(std::uintptr_t * arr, std::uint64_t write_index, const T & first,
                                const Args &... args) {
    std::uint64_t result = 0;
    if constexpr (HasCuMallocSize<T>) {
        result = first.cumalloc_size();
    } else if constexpr (std::is_trivially_copyable<T>::value) {
        result = sizeof(T);
    }
    if constexpr (sizeof...(args) > 0) {
        arr[write_index] = result;
        result += total_malloc_size(arr, write_index + 1, args...);
    }
    return result;
}

// Copy metadata to GPU
template <typename T, typename... Args>
requires HasCopyToGpu<T> || std::is_trivially_copyable<T>::value
void * copy_metadata_to_gpu(std::uintptr_t stream_ptr, void * data, const T & first, const Args &... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    void * result;
    if constexpr (HasCopyToGpu<T>) {
        result = first.copy_to_gpu(ptr_data, ptr_data + 1, stream_ptr);
    } else if constexpr (std::is_trivially_copyable<T>::value) {
        memcpy_cpu_to_gpu(ptr_data, &first, sizeof(T), stream_ptr);
        result = reinterpret_cast<void *>(ptr_data + 1);
    }
    if constexpr (sizeof...(args) > 0) {
        result = copy_metadata_to_gpu(stream_ptr, result, args...);
    }
    return result;
}

#ifdef __NVCC__

// Copy metadata to shared mem
template <typename T, typename... Args>
requires HasCopyByBlock<T> || std::is_trivially_copyable<T>::value
__cudevice__ std::tuple<T *, Args *...> copy_metadata_to_shmem(void * data, const std::uint64_t & thread_idx,
                                                               const std::uint64_t & block_size, void ** final,
                                                               const T & first, const Args &... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    std::tuple<T *> current(ptr_data);
    void * next;
    if constexpr (HasCopyByBlock<T>) {
        next = first.copy_by_block(ptr_data, ptr_data + 1, thread_idx, block_size);
    } else if constexpr (std::is_trivially_copyable<T>::value) {
        if (thread_idx == 0) {
            *ptr_data = first;
        }
        __syncthreads();
        next = ptr_data + 1;
    }
    *final = next;
    if constexpr (sizeof...(args) > 0) {
        return std::tuple_cat(current, copy_metadata_to_shmem(next, thread_idx, block_size, final, args...));
    } else {
        return current;
    }
}

#endif  // __NVCC__

// ---------------------------------------------------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------------------------------------------------

// Constructor
template <typename... Args>
cuda::Dispatcher<Args...>::Dispatcher(std::uintptr_t stream_ptr, const Args &... args) {
    // allocate data
    this->offset_.fill(0);
    this->stream_ptr_ = stream_ptr;
    this->total_malloc_size_ = total_malloc_size(this->offset_.data(), 1, args...);
    // this->gpu_ptr_ = mem_alloc_device(this->total_malloc_size_, stream_ptr);
    mem_alloc_device(&(this->gpu_ptr_), this->total_malloc_size_, stream_ptr);
    // storing source pointers
    this->type_ptr_ = std::make_tuple<Args *...>(const_cast<Args *>(&(args))...);
    // copy data to GPU
    copy_metadata_to_gpu(stream_ptr, this->gpu_ptr_, args...);
    // calculate cumulative sum
    for (std::size_t i = 1; i < this->offset_.size(); i++) {
        this->offset_[i] += this->offset_[i - 1];
    }
}

// Get pointer element
template <typename... Args>
template <std::uint64_t index>
typename std::tuple_element<index, std::tuple<Args *...>>::type cuda::Dispatcher<Args...>::get(void) {
    std::uintptr_t result = reinterpret_cast<std::uintptr_t>(this->gpu_ptr_) + this->offset_[index];
    return reinterpret_cast<typename std::tuple_element<index, std::tuple<Args *...>>::type>(result);
}

// Destructor
template <typename... Args>
cuda::Dispatcher<Args...>::~Dispatcher(void) {
    if (this->gpu_ptr_ != nullptr) {
        mem_free_device_noexcept(this->gpu_ptr_, this->stream_ptr_);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Copy classes by CUDA blocks
// ---------------------------------------------------------------------------------------------------------------------

#ifdef __NVCC__

// Copy class to shared memory
template <typename... Args>
__cudevice__ std::tuple<void *, Args *...> cuda::copy_objects(void * share_ptr, const std::uint64_t & thread_idx,
                                                              const std::uint64_t & block_size, const Args &... args) {
    void * final = nullptr;
    std::tuple<Args *...> result = copy_metadata_to_shmem(share_ptr, thread_idx, block_size, &final, args...);
    std::tuple<void *> final_tpl(final);
    return std::tuple_cat(final_tpl, result);
}

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_CUDA_MEMORY_TPP_
