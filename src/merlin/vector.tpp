// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_TPP_
#define MERLIN_VECTOR_TPP_

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Constructor from initializer list
template <typename T>
__cuhostdev__ Vector<T>::Vector(std::initializer_list<T> data) : size_(data.size()) {
    this->data_ = new T[data.size()];
    for (int i = 0; i < data.size(); i++) {
        this->data_[i] = data.begin()[i];
    }
}

// Constructor from size and fill-in value
template <typename T>
__cuhostdev__ Vector<T>::Vector(std::uint64_t size, T value) : size_(size) {
    this->data_ = new T[size];
    for (int i = 0; i < size; i++) {
        this->data_[i] = value;
    }
}

// Constructor from a pointer to first and last element
template <typename T>
template <typename Convertable>
__cuhostdev__ Vector<T>::Vector(const Convertable * ptr_first, const Convertable * ptr_last) {
    this->size_ = ptr_last - ptr_first;
    this->data_ = new T[this->size_];
    for (int i = 0; i < this->size_; i++) {
        this->data_[i] = T(ptr_first[i]);
    }
}

// Convertable constructor
template <typename T>
template <typename Convertable>
__cuhostdev__ Vector<T>::Vector(const Convertable * ptr_src, std::uint64_t size) : size_(size) {
    this->data_ = new T[this->size_];
    for (int i = 0; i < this->size_; i++) {
        this->data_[i] = T(ptr_src[i]);
    }
}

// Copy constructor
template <typename T>
__cuhostdev__ Vector<T>::Vector(const Vector<T> & src) : size_(src.size_) {
    this->data_ = new T[this->size_];
    for (int i = 0; i < src.size_; i++) {
        this->data_[i] = src.data_[i];
    }
}

// Copy assignment
template <typename T>
__cuhostdev__ Vector<T> & Vector<T>::operator=(const Vector<T> & src) {
    // free old data
    if (this->data_ != nullptr) {
        delete[] this->data_;
    }
    // copy new data
    this->size_ = src.size_;
    this->data_ = new T[this->size_];
    for (int i = 0; i < src.size_; i++) {
        this->data_[i] = src.data_[i];
    }
    return *this;
}

// Move constructor
template <typename T>
__cuhostdev__ Vector<T>::Vector(Vector<T> && src) : size_(src.size_), data_(src.data_) {
    src.data_ = nullptr;
}

// Move assignment
template <typename T>
__cuhostdev__ Vector<T> & Vector<T>::operator=(Vector<T> && src) {
    if (this->data_ != nullptr) {
        delete[] this->data_;
    }
    this->size_ = src.size_;
    this->data_ = src.data_;
    src.data_ = nullptr;
    return *this;
}

#ifndef __MERLIN_CUDA__

// Copy data from CPU to a global memory on GPU
template <typename T>
void Vector<T>::copy_to_gpu(Vector<T> * gpu_ptr, void * data_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access this feature.\n");
}

// Copy data from GPU to CPU
template <typename T>
void Vector<T>::copy_from_device(Vector<T> * gpu_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access this feature.\n");
}

#elif defined(__NVCC__)

// Copy data from CPU to a global memory on GPU
template <typename T>
void Vector<T>::copy_to_gpu(Vector<T> * gpu_ptr, void * data_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    Vector<T> copy_on_gpu;
    // copy data
    cudaMemcpy(data_ptr, this->data_, this->size_*sizeof(T), cudaMemcpyHostToDevice);
    // copy metadata
    copy_on_gpu.data_ = reinterpret_cast<T *>(data_ptr);
    copy_on_gpu.size_ = this->size_;
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(Vector<T>), cudaMemcpyHostToDevice);
    // nullify data on copy to avoid deallocate memory on CPU
    copy_on_gpu.data_ = nullptr;
}

// Copy data from GPU to CPU
template <typename T>
void Vector<T>::copy_from_device(Vector<T> * gpu_ptr) {
    // copy data
    std::uintptr_t gpu_data = reinterpret_cast<std::uintptr_t>(gpu_ptr) + sizeof(Vector<T>);
    cudaMemcpy(reinterpret_cast<T *>(gpu_data), this->data_,
               this->size_*sizeof(T), cudaMemcpyDeviceToHost);
}

#endif  // __MERLIN_CUDA__

#ifdef __NVCC__
// Copy to shared memory
template <typename T>
__cudevice__ void Vector<T>::copy_to_shared_mem(Vector<T> * share_ptr, void * data_ptr) {
    bool check_zeroth_thread = (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        // copy size
        share_ptr->size_ = this->size_;
        share_ptr->data_ = reinterpret_cast<T *>(data_ptr);
        // copy data in parallel
        for (int i = 0; i < this->size_; i++) {
            share_ptr->data_[i] = this->data_[i];
        }
    }
    __syncthreads();
}
#endif

// Destructor
template <typename T>
__cuhostdev__ Vector<T>::~Vector(void) {
    if (this->data_ != nullptr) {
        delete[] this->data_;
    }
}

}  // namespace merlin

#endif  // MERLIN_VECTOR_TPP_
