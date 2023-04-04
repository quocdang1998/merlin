// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_TPP_
#define MERLIN_VECTOR_TPP_

#include <sstream>  // std::ostringstream
#include <type_traits>  // std::is_arithmetic_v, std::is_constructible_v, std::is_copy_assignable_v

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Constructor from initializer list
template <typename T>
__cuhostdev__ Vector<T>::Vector(std::initializer_list<T> data) noexcept : size_(data.size()) {
    this->data_ = new T[data.size()];
    for (std::uint64_t i = 0; i < data.size(); i++) {
        this->data_[i] = data.begin()[i];
    }
}

// Constructor from size and fill-in value
template <typename T>
__cuhostdev__ Vector<T>::Vector(std::uint64_t size, const T & value) : size_(size) {
    static_assert(std::is_copy_constructible_v<T> || std::is_copy_assignable_v<T>,
                  "Desired type is not copy constructible or copy assignable.\n");
    this->data_ = new T[size];
    for (std::uint64_t i = 0; i < size; i++) {
        if constexpr (std::is_copy_assignable_v<T>) {
            this->data_[i] = value;
        } else if constexpr (std::is_copy_constructible_v<T>) {
            new (&(this->data_[i])) T(value);
        }
    }
}

// Constructor from a pointer to first and last element
template <typename T>
template <typename Convertable>
__cuhostdev__ Vector<T>::Vector(const Convertable * ptr_first, const Convertable * ptr_last) {
    static_assert(std::is_constructible_v<T, Convertable>, "Object is not constructable from original type.\n");
    this->size_ = ptr_last - ptr_first;
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; i < this->size_; i++) {
        this->data_[i] = T(ptr_first[i]);
    }
}

// Convertable constructor
template <typename T>
template <typename Convertable>
__cuhostdev__ Vector<T>::Vector(const Convertable * ptr_src, std::uint64_t size) : size_(size) {
    static_assert(std::is_constructible_v<T, Convertable>, "Object is not constructable from original type.\n");
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; i < this->size_; i++) {
        this->data_[i] = T(ptr_src[i]);
    }
}

// Copy constructor
template <typename T>
__cuhostdev__ Vector<T>::Vector(const Vector<T> & src) : size_(src.size_) {
    this->data_ = new T[this->size_];
    for (std::uint64_t i = 0; i < src.size_; i++) {
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
    for (std::uint64_t i = 0; i < src.size_; i++) {
        this->data_[i] = src.data_[i];
    }
    return *this;
}

// Move constructor
template <typename T>
__cuhostdev__ Vector<T>::Vector(Vector<T> && src) : data_(src.data_), size_(src.size_), assigned_(src.assigned_) {
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
    this->assigned_ = src.assigned_;
    src.data_ = nullptr;
    return *this;
}

// Assign current vector as sub-vector
template <typename T>
__cuhostdev__ void Vector<T>::assign(T * ptr_src, std::uint64_t size) {
    this->data_ = ptr_src;
    this->size_ = size;
    this->assigned_ = true;
}

// Assign current vector as sub-vector
template <typename T>
__cuhostdev__ void Vector<T>::assign(T * ptr_first, T * ptr_last) {
    this->data_ = ptr_first;
    this->size_ = (ptr_last - ptr_first);
    this->assigned_ = true;
}

#ifndef __MERLIN_CUDA__

// Copy data from CPU to a global memory on GPU
template <typename T>
void * Vector<T>::copy_to_gpu(Vector<T> * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access this feature.\n");
    return nullptr;
}

// Copy data from GPU to CPU
template <typename T>
void Vector<T>::copy_from_gpu(const T * gpu_ptr, std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access this feature.\n");
}

#elif defined(__NVCC__)

// Copy data from CPU to a global memory on GPU
template <typename T>
void * Vector<T>::copy_to_gpu(Vector<T> * gpu_ptr, void * data_ptr, std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    Vector<T> copy_on_gpu;
    // copy data
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(data_ptr, this->data_, this->size_*sizeof(T), ::cudaMemcpyHostToDevice, stream);
    // copy metadata
    copy_on_gpu.data_ = reinterpret_cast<T *>(data_ptr);
    copy_on_gpu.size_ = this->size_;
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(Vector<T>), ::cudaMemcpyHostToDevice, stream);
    // nullify data on copy to avoid deallocate memory on CPU
    copy_on_gpu.data_ = nullptr;
    std::uintptr_t ptr_end = reinterpret_cast<std::uintptr_t>(data_ptr) + this->size_*sizeof(T);
    return reinterpret_cast<void *>(ptr_end);
}

// Copy data from GPU to CPU
template <typename T>
void Vector<T>::copy_from_gpu(const T * gpu_ptr, std::uintptr_t stream_ptr) {
    // copy data
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(this->data_, gpu_ptr, this->size_*sizeof(T),
                      cudaMemcpyDeviceToHost, stream);
}

#endif  // !__MERLIN_CUDA__

#ifdef __NVCC__

// Copy to shared memory
template <typename T>
__cudevice__ void * Vector<T>::copy_to_shared_mem(Vector<T> * share_ptr, void * data_ptr) const {
    bool check_zeroth_thread = (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        // copy size
        share_ptr->size_ = this->size_;
        share_ptr->data_ = reinterpret_cast<T *>(data_ptr);
        // copy data
        for (std::uint64_t i = 0; i < this->size_; i++) {
            share_ptr->data_[i] = this->data_[i];
        }
    }
    __syncthreads();
    std::uintptr_t ptr_end = reinterpret_cast<std::uint64_t>(data_ptr) + this->size_*sizeof(T);
    return reinterpret_cast<void *>(ptr_end);
}

// Copy to shared memory by current thread
template <typename T>
__cudevice__ void * Vector<T>::copy_to_shared_mem_single(Vector<T> * share_ptr, void * data_ptr) const {
    // copy size
    share_ptr->size_ = this->size_;
    share_ptr->data_ = reinterpret_cast<T *>(data_ptr);
    // copy data
    for (std::uint64_t i = 0; i < this->size_; i++) {
        share_ptr->data_[i] = this->data_[i];
    }
    std::uintptr_t ptr_end = reinterpret_cast<std::uint64_t>(data_ptr) + this->size_*sizeof(T);
    return reinterpret_cast<void *>(ptr_end);
}

#endif  // __NVCC__

// String representation for types printdable to std::ostream
template <typename T>
std::string Vector<T>::str(const char * sep) const {
    std::ostringstream os;
    os << "<";
    for (std::uint64_t i = 0; i < this->size_; i++) {
        if constexpr (std::is_arithmetic_v<T>) {
            os << this->data_[i];
        } else {
            os << this->data_[i].str();
        }
        if (i != this->size_-1) {
            os << sep;
        }
    }
    os << ">";
    return os.str();
}

// Identical comparison operator
template <typename T>
__cuhostdev__ bool operator==(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept {
    if (vec_1.size_ != vec_2.size_) {
        return false;
    }
    for (std::uint64_t i = 0; i < vec_1.size_; i++) {
        if (vec_1.data_[i] != vec_2.data_[i]) {
            return false;
        }
    }
    return true;
}

// Difference comparison operator
template <typename T>
__cuhostdev__ bool operator!=(const Vector<T> & vec_1, const Vector<T> & vec_2) noexcept {
    if (vec_1.size_ != vec_2.size_) {
        return true;
    }
    for (std::uint64_t i = 0; i < vec_1.size_; i++) {
        if (vec_1.data_[i] != vec_2.data_[i]) {
            return true;
        }
    }
    return false;
}

// Check all elemets are zeros
template <typename T>
__cuhostdev__ bool is_zeros(const Vector<T> & vec) noexcept {
    for (std::uint64_t i = 0; i < vec.size_; i++) {
        if (vec.data_[i] != 0) {
            return false;
        }
    }
    return true;
}

// Destructor
template <typename T>
__cuhostdev__ Vector<T>::~Vector(void) {
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        delete[] this->data_;
    }
}

// Create vector from constrcutor arguments
template <typename T, typename ... Args>
Vector<T> make_vector(std::uint64_t size, Args ... args) noexcept {
    static_assert(std::is_constructible_v<T, Args ...>, "Desired type is not constructible from provided arg.\n");
    Vector<T> result;
    result.data() = new T[size];
    result.size() = size;
    for (std::uint64_t i = 0; i < size; i++) {
        new (&(result[i])) T(args...);
    }
    return result;
}

}  // namespace merlin

#endif  // MERLIN_VECTOR_TPP_
