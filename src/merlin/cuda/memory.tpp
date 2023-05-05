// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_MEMORY_TPP_
#define MERLIN_CUDA_MEMORY_TPP_

#include <algorithm>  // std::reverse
#include <cstddef>  // std::size_t
#include <type_traits>  // std::remove_pointer_t
#include <utility>  // std::make_pair

#include <cuda.h>  // ::cuCtxGetDevice

#include "merlin/env.hpp"  // merlin::Environment
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
void * copy_metadata_to_gpu(std::uintptr_t stream_ptr, void * data, const T & first, const Args & ... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    void * result = first.copy_to_gpu(ptr_data, ptr_data+1, stream_ptr);
    if constexpr (sizeof...(args) > 0) {
        result = copy_metadata_to_gpu(stream_ptr, result, args...);
    }
    return result;
}

// Copy metadata to shared mem
template <typename T, typename ... Args>
__cudevice__ std::tuple<T *, Args * ...> copy_metadata_to_shmem(void * data, void ** final,
                                                                const T & first, const Args & ... args) {
    T * ptr_data = reinterpret_cast<T *>(data);
    std::tuple<T *> current(ptr_data);
    void * next = first.copy_to_shared_mem(ptr_data, ptr_data+1);
    *final = next;
    if constexpr (sizeof...(args) > 0) {
        return std::tuple_cat(current, copy_metadata_to_shmem(next, final, args...));
    } else {
        return current;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Memory
// --------------------------------------------------------------------------------------------------------------------

// Constructor
template <typename ... Args>
cuda::Memory<Args ...>::Memory(std::uintptr_t stream_ptr, const Args & ... args) {
    // allocate data
    this->offset_.fill(0);
    this->stream_ptr_ = stream_ptr;
    this->total_malloc_size_ = total_malloc_size(this->offset_.data(), 1, args...);
    ::cudaError_t err_ = ::cudaMallocAsync(&(this->gpu_ptr_), this->total_malloc_size_,
                                           reinterpret_cast<::cudaStream_t>(stream_ptr));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Alloc data faile with message \"%s\"\n", ::cudaGetErrorString(err_));
    }
    // storing source pointers
    this->type_ptr_ = std::make_tuple<Args * ...>(const_cast<Args *>(&(args))...);
    // copy data to GPU
    copy_metadata_to_gpu(stream_ptr, this->gpu_ptr_, args...);
    // cummulative sum
    for (std::size_t i = 1; i < this->offset_.size(); i++) {
        this->offset_[i] += this->offset_[i-1];
    }
}

// Get pointer element
template <typename ... Args>
template <std::uint64_t index>
typename std::tuple_element<index, std::tuple<Args * ...>>::type cuda::Memory<Args ...>::get(void) {
    std::uintptr_t result = reinterpret_cast<std::uintptr_t>(this->gpu_ptr_) + this->offset_[index];
    return reinterpret_cast<typename std::tuple_element<index, std::tuple<Args * ...>>::type>(result);
}

// Defer the CUDA free on pointer
template <typename ... Args>
void cuda::Memory<Args ...>::defer_allocation(void) {
    if (this->gpu_ptr_ != nullptr) {
        int gpu;
        ::cuCtxGetDevice(&gpu);
        Environment::deferred_gpu_pointer.push_back(std::make_pair(gpu, this->gpu_ptr_));
        this->deferred_dealloc_ = true;
    }
}

// Destructor
template <typename ... Args>
cuda::Memory<Args ...>::~Memory(void) {
    if ((this->gpu_ptr_ != nullptr) && !(this->deferred_dealloc_)) {
        ::cudaFreeAsync(this->gpu_ptr_, reinterpret_cast<::cudaStream_t>(this->stream_ptr_));
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Shared Memory Copy
// --------------------------------------------------------------------------------------------------------------------

// Copy class to shared memory
template <typename ... Args>
__cudevice__ std::tuple<void *, Args * ...> cuda::copy_class_to_shared_mem(void * share_ptr, const Args & ... args) {
    void * final = nullptr;
    std::tuple<Args * ...> result = copy_metadata_to_shmem(share_ptr, &final, args...);
    std::tuple<void *> final_tpl(final);
    return std::tuple_cat(final_tpl, result);
}

}  // namespace merlin

#endif  // MERLIN_CUDA_MEMORY_TPP_
