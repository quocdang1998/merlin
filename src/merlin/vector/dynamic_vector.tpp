// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_DYNAMIC_VECTOR_TPP_
#define MERLIN_VECTOR_DYNAMIC_VECTOR_TPP_

#include <type_traits>  // std::is_copy_constructible_v, std::is_copy_assignable_v
#include <utility>      // std::swap, std::exchange

#include "merlin/memory.hpp"  // merlin::memcpy_cpu_to_gpu, merlin::memcpy_gpu_to_cpu

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Dynamic Vector
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from initializer list
template <typename T>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(std::initializer_list<T> data) : size_(data.size()) {
    this->data_ = new T[data.size()];
    for (std::uint64_t i = 0; i < data.size(); i++) {
        this->data_[i] = std::data(data)[i];
    }
}

// Constructor from size and fill-in value
template <typename T>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(std::uint64_t size, const T & value) : size_(size) {
    static_assert(std::is_copy_constructible_v<T>, "Element type is not copy constructible.\n");
    this->data_ = new T[size];
    for (std::uint64_t i = 0; i < size; i++) {
        new (this->data_ + i) T(value);
    }
}

// Copy constructor
template <typename T>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(const vector::DynamicVector<T> & src) : size_(src.size_) {
    static_assert(std::is_copy_constructible_v<T>, "Element type is not copy constructible.\n");
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; i < src.size_; i++) {
        new (this->data_ + i) T(src.data_[i]);
    }
}

// Copy assignment
template <typename T>
__cuhostdev__ vector::DynamicVector<T> & vector::DynamicVector<T>::operator=(const vector::DynamicVector<T> & src) {
    static_assert(std::is_copy_assignable_v<T>, "Element type is not copy assignable.\n");
    if (this == &src) {
        return *this;
    }
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        delete[] this->data_;
    }
    this->size_ = src.size_;
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; i < src.size_; i++) {
        this->data_[i] = src.data_[i];
    }
    return *this;
}

// Move constructor
template <typename T>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(vector::DynamicVector<T> && src) {
    this->data_ = std::exchange(src.data_, nullptr);
    this->size_ = std::exchange(src.size_, 0);
    this->assigned_ = std::exchange(src.assigned_, false);
}

// Move assignment
template <typename T>
__cuhostdev__ vector::DynamicVector<T> & vector::DynamicVector<T>::operator=(vector::DynamicVector<T> && src) {
    if (this == &src) {
        return *this;
    }
    std::swap(this->data_, src.data_);
    std::swap(this->size_, src.size_);
    std::swap(this->assigned_, src.assigned_);
    return *this;
}

// Constructor from a pointer to first and last element
template <typename T>
template <typename Pointer>
requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, T &>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(Pointer data, std::uint64_t size, bool assign) : size_(size) {
    if (assign) {
        this->assigned_ = true;
        this->data_ = std::to_address(data);
        return;
    }
    this->data_ = new T[size];
    for (std::uint64_t i = 0; i < size; i++) {
        new (this->data_ + i) T(*data);
        ++data;
    }
}

// Constructor from a pointer to first and last element
template <typename T>
template <typename Pointer>
requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
__cuhostdev__ vector::DynamicVector<T>::DynamicVector(Pointer first, Pointer last) {
    this->size_ = std::distance(first, last);
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; first != last; i++) {
        new (this->data_ + i) T(*first);
        ++first;
    }
}

// Resize vector
template <typename T>
__cuhostdev__ void vector::DynamicVector<T>::resize(std::uint64_t new_size) {
    if (this->size_ = new_size) {
        return;
    }
    if (this->assigned_) {
        this->size_ = new_size;
        return;
    }
    T * new_data = new T[new_size];
    for (std::uint64_t i = 0; i < this->size_; i++) {
        new (new_data + i) T(this->data_[i]);
    }
    if (this->data_ != nullptr) {
        delete[] this->data_;
    }
    this->data_ = new_data;
    this->size_ = new_size;
}

// Copy data from CPU to a global memory on GPU
template <typename T>
void * vector::DynamicVector<T>::copy_to_gpu(vector::DynamicVector<T> * gpu_ptr, void * data_ptr,
                                             std::uintptr_t stream_ptr) const {
    // copy data
    memcpy_cpu_to_gpu(data_ptr, this->data_, this->size_ * sizeof(T), stream_ptr);
    // initialize an object containing metadata and copy it to GPU
    vector::DynamicVector<T> copy_on_gpu;
    copy_on_gpu.data_ = reinterpret_cast<T *>(data_ptr);
    copy_on_gpu.size_ = this->size_;
    memcpy_cpu_to_gpu(gpu_ptr, &copy_on_gpu, sizeof(vector::DynamicVector<T>), stream_ptr);
    // nullify data on copy to avoid deallocate memory on CPU
    copy_on_gpu.data_ = nullptr;
    std::uintptr_t ptr_end = reinterpret_cast<std::uintptr_t>(data_ptr) + this->size_ * sizeof(T);
    return reinterpret_cast<void *>(ptr_end);
}

// Copy data from GPU to CPU
template <typename T>
void * vector::DynamicVector<T>::copy_from_gpu(T * gpu_ptr, std::uintptr_t stream_ptr) {
    // copy data from GPU
    memcpy_gpu_to_cpu(this->data_, gpu_ptr, this->size_ * sizeof(T), stream_ptr);
    return reinterpret_cast<void *>(gpu_ptr + this->size_);
}

#ifdef __NVCC__

// Copy to shared memory
template <typename T>
__cudevice__ void * vector::DynamicVector<T>::copy_by_block(vector::DynamicVector<T> * dest_ptr, void * data_ptr,
                                                            std::uint64_t thread_idx, std::uint64_t block_size) const {
    T * new_data = reinterpret_cast<T *>(data_ptr);
    // copy meta data
    if (thread_idx == 0) {
        dest_ptr->size_ = this->size_;
        dest_ptr->data_ = new_data;
    }
    __syncthreads();
    // copy data
    for (std::uint64_t i = thread_idx; i < this->size_; i += block_size) {
        new_data[i] = this->data_[i];
    }
    __syncthreads();
    return reinterpret_cast<void *>(new_data + this->size_);
}

// Copy to shared memory by current thread
template <typename T>
__cudevice__ void * vector::DynamicVector<T>::copy_by_thread(vector::DynamicVector<T> * dest_ptr,
                                                             void * data_ptr) const {
    // copy meta data
    T * new_data = reinterpret_cast<T *>(data_ptr);
    dest_ptr->size_ = this->size_;
    dest_ptr->data_ = new_data;
    // copy data
    for (std::uint64_t i = 0; i < this->size_; i++) {
        new_data[i] = this->data_[i];
    }
    return reinterpret_cast<void *>(new_data + this->size_);
}

#endif  // __NVCC__

// Destructor
template <typename T>
__cuhostdev__ vector::DynamicVector<T>::~DynamicVector(void) {
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        delete[] this->data_;
    }
}

}  // namespace merlin

#endif  // MERLIN_VECTOR_DYNAMIC_VECTOR_TPP_
