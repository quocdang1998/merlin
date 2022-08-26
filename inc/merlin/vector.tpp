// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_TPP_
#define MERLIN_VECTOR_TPP_

#include <cstdint>  // uintptr_t
#include <algorithm>  // std::copy, std::fill_n

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Constructor from initializer list
template <typename T>
Vector<T>::Vector(std::initializer_list<T> data) {
    this->size_ = data.size();
    this->data_ = new T[data.size()];
    std::copy(data.begin(), data.end(), this->data_);
}

// Constructor from size and fill-in value
template <typename T>
Vector<T>::Vector(unsigned long int size, T value) : size_(size) {
    this->data_ = new T[size];
    std::fill_n(this->data_, size, value);
}

// Constructor from a pointer to first and last element
template <typename T>
Vector<T>::Vector(T * ptr_first, T * ptr_last) {
    this->size_ = ptr_last - ptr_first;
    this->data_ = new T[this->size_];
    std::copy(ptr_first, ptr_last, this->data_);
}

// Copy constructor
template <typename T>
Vector<T>::Vector(const Vector<T> & src) {
    this->size_ = src.size_;
    this->data_ = new T[this->size_];
    std::copy(src.data_, src.data_+this->size_, this->data_);
}

// Copy assignment
template <typename T>
Vector<T> & Vector<T>::operator=(const Vector<T> & src) {
    // free old data
    if (this->data_ != NULL) {
        delete[] this->data_;
    }
    // copy new data
    this->size_ = src.size_;
    this->data_ = new T[this->size_];
    std::copy(src.data_, src.data_+this->size_, this->data_);
    return *this;
}

// Move constructor
template <typename T>
Vector<T>::Vector(Vector<T> && src) {
    this->size_ = src.size_;
    this->data_ = src.data_;
    src.data_ = NULL;
}

// Move assignment
template <typename T>
Vector<T> & Vector<T>::operator=(Vector<T> && src) {
    if (this->data_ != NULL) {
        delete[] this->data_;
    }
    this->size_ = src.size_;
    this->data_ = src.data_;
    src.data_ = NULL;
    return *this;
}

// Copy data to GPU
#ifndef MERLIN_CUDA_
template <typename T>
void Vector<T>::copy_to_device_ptr(Vector<T> * gpu_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access this feature.\n");
}
#elif defined(__NVCC__)
template <typename T>
void Vector<T>::copy_to_device_ptr(Vector<T> * gpu_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    Vector<T> copy_on_gpu;
    // copy data
    uintptr_t gpu_data = reinterpret_cast<uintptr_t>(gpu_ptr) + sizeof(Vector<T>);
    cudaMemcpy(reinterpret_cast<T *>(gpu_data), this->data_,
               this->size_*sizeof(T), cudaMemcpyHostToDevice);
    // copy metadata
    copy_on_gpu.data_ = reinterpret_cast<T *>(gpu_data);
    copy_on_gpu.size_ = this->size_;
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(Vector<T>), cudaMemcpyHostToDevice);
    // nullify data on copy to avoid deallocate memory on CPU
    copy_on_gpu.data_ = NULL;
}
#endif  // MERLIN_CUDA_

// Copy to shared memory
#ifdef __NVCC__
template <typename T>
__cudevice__ void Vector<T>::copy_to_shared_mem(Vector<T> * share_ptr) {
    // copy size
    share_ptr->size_ = this->size_;
    share_ptr->data_ = (T *) &share_ptr[1];
    // copy data in parallel
    bool check_zeroth_thread = (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)
                            && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        for (int i = 0; i < this->size_; i++) {
            share_ptr->data_[i] = this->data_[i];
        }
    }
    __syncthreads();
}
#endif

// Destructor
template <typename T>
Vector<T>::~Vector(void) {
    if (this->data_ != NULL) {
        delete[] this->data_;
    }
}

}  // namespace merlin

#endif  // MERLIN_VECTOR_TPP_
