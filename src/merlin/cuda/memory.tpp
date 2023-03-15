// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_MEMORY_TPP_
#define MERLIN_CUDA_MEMORY_TPP_

#include <algorithm>  // std::reverse
#include <cstddef>  // std::size_t
#include <type_traits>  // std::remove_pointer_t

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// Total malloc size
template <typename T, typename ... Args>
std::uint64_t total_malloc_size(std::uintptr_t * arr, std::uint64_t write_index, const T & first,
                                const Args & ... args) {
    std::uint64_t result = first.malloc_size();
    if constexpr (sizeof...(args) > 0) {
        arr[write_index] = result;
        result += total_malloc_size(arr, write_index+1, args...);
    }
    return result;
}

// Copy metadata to GPU
template <typename T, typename ... Args>
void * copy_metadata_to_gpu(void * data, const T & first, const Args & ... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    void * result = first.copy_to_gpu(ptr_data, ptr_data+1);
    if constexpr (sizeof...(args) > 0) {
        result = copy_metadata_to_gpu(result, args...);
    }
    return result;
}

// Copy metadata to shared mem
template <typename T, typename ... Args>
__cudevice__ void * copy_metadata_to_shared_mem(void * data, std::uintptr_t * arr, std::uint64_t write_index,
                                                const T & first, const Args & ... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    arr[write_index] = reinterpret_cast<std::uintptr_t>(data);
    void * result = first.copy_to_shared_mem(ptr_data, ptr_data+1);
    if constexpr (sizeof...(args) > 0) {
        result = copy_metadata_to_shared_mem(result, arr, write_index+1, args...);
    }
    return result;
}

// --------------------------------------------------------------------------------------------------------------------
// Memory
// --------------------------------------------------------------------------------------------------------------------


// Constrcutor
template <typename ... Args>
cuda::Memory<Args ...>::Memory(const Args & ... args) {
    // allocate data
    this->offset_.fill(0);
    this->total_malloc_size_ = total_malloc_size(this->offset_.data(), 1, args...);
    ::cudaError_t err_ = ::cudaMalloc(&(this->gpu_ptr_), this->total_malloc_size_);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Alloc data faile with message \"%s\"\n", ::cudaGetErrorString(err_));
    }
    // storing source pointers
    this->type_ptr_ = std::make_tuple<const Args * ...>(&(args)...);
    // copy data to GPU
    copy_metadata_to_gpu(this->gpu_ptr_, args...);
    // cummulative sum
    for (std::size_t i = 1; i < this->offset_.size(); i++) {
        this->offset_[i] += this->offset_[i-1];
    }
}

// Get pointer element
template <typename ... Args>
template <std::uint64_t index>
typename std::tuple_element<index, std::tuple<const Args * ...>>::type cuda::Memory<Args ...>::get(void) {
    std::uintptr_t result = reinterpret_cast<std::uintptr_t>(this->gpu_ptr_) + this->offset_[index];
    return reinterpret_cast<typename std::tuple_element<index, std::tuple<const Args * ...>>::type>(result);
}

// Destructor
template <typename ... Args>
cuda::Memory<Args ...>::~Memory(void) {
    if (this->gpu_ptr_ != nullptr) {
        ::cudaFree(this->gpu_ptr_);
    }
}

template <typename ... Args>
__cudevice__ std::tuple<Args * ...> cuda::copy_class_to_shared_mem(void * share_ptr, const Args & ... args) {
    std::tuple<Args * ...> result;
    const std::uint64_t n_elems = std::tuple_size_v<std::tuple<Args * ...>>;
    std::array<std::uintptr_t, sizeof...(Args)> arr;
    copy_metadata_to_shared_mem(share_ptr, arr.data(), 0, args...);
    for (std::uint64_t i = 0; i < n_elems; i++) {
        std::get<const_cast<const uint64_t>(i)>(result) = reinterpret_cast<typename std::tuple_element<const uint64_t>(i), std::tuple<Args * ...>>::type>(arr[i]);
    }
    return result;
}

}  // namespace merlin

#endif  // MERLIN_CUDA_MEMORY_TPP_
