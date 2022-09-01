// Copyright 2022 quocdang1998
#include "merlin/parcel.hpp"

#include <cstdarg>  // std::va_list, va_start, va_arg, va_end
#include <functional>  // std::bind, std::placeholders

#include "merlin/array.hpp"  // merlin::Array
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::inner_prod, merlin::contiguous_strides,
                             // merlin::get_current_device, merlin::contiguous_to_ndim_idx
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

#if !defined(__LIBMERLINCUDA_STATIC__) || defined(__MERLIN_FORCE_STATIC__)

// Default constructor
Parcel::Parcel(void) {}

// Constructor from CPU array
Parcel::Parcel(const Array & cpu_array, uintptr_t stream) : NdData(cpu_array) {
    // get device id
    cudaGetDevice(&(this->device_id_));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // cast stream
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(stream);
    // reset strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // create copy function
    auto copy_func = std::bind(cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               cudaMemcpyHostToDevice, copy_stream);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&cpu_array), copy_func);
}

// Check if current device holds data pointed by object
int Parcel::check_device(void) const {
    return (this->device_id_ - get_current_device());
}

// Copy constructor
Parcel::Parcel(const Parcel & src) : NdData(src) {
    // get device id
    cudaGetDevice(&(this->device_id_));
    // reform strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_id_,
                               std::placeholders::_2, src.device_id_, std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), copy_func);
}

// Copy assignement
Parcel & Parcel::operator=(const Parcel & src) {
    // free old data
    this->free_current_data();
    // reform strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_id_,
                               std::placeholders::_2, src.device_id_, std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), copy_func);
    return *this;
}

// Move constructor
Parcel::Parcel(Parcel && src) : NdData(src) {
    // move device id
    this->device_id_ = src.device_id_;
    // take over pointer to source
    src.data_ = NULL;
}

// Move assignment
Parcel & Parcel::operator=(Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id
    this->device_id_ = src.device_id_;
    // take over pointer to source
    src.data_ = NULL;
    return *this;
}

// Copy data to a pre-allocated memory
void Parcel::copy_to_device_ptr(Parcel * gpu_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    Parcel copy_on_gpu;
    // copy shape and strides data
    uintptr_t shape_ptr = reinterpret_cast<uintptr_t>(gpu_ptr) + sizeof(Parcel);
    cudaMemcpy(reinterpret_cast<unsigned long int *>(shape_ptr), this->shape_.data(),
               this->ndim_*sizeof(unsigned long int), cudaMemcpyHostToDevice);
    uintptr_t strides_ptr = shape_ptr + this->ndim_*sizeof(unsigned long int);
    cudaMemcpy(reinterpret_cast<unsigned long int *>(strides_ptr), this->strides_.data(),
               this->ndim_*sizeof(unsigned long int), cudaMemcpyHostToDevice);
    // shallow copy of the current object
    copy_on_gpu.data_ = this->data_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.device_id_ = this->device_id_;
    copy_on_gpu.shape_.data() = reinterpret_cast<unsigned long int *>(shape_ptr);
    copy_on_gpu.shape_.size() = this->ndim_;
    copy_on_gpu.strides_.data() = reinterpret_cast<unsigned long int *>(strides_ptr);
    copy_on_gpu.strides_.size() = this->ndim_;
    // copy temporary object to GPU
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(Parcel), cudaMemcpyHostToDevice);
    // nullify data pointer to avoid free data
    copy_on_gpu.data_ = NULL;
    copy_on_gpu.shape_.data() = NULL;
    copy_on_gpu.strides_.data() = NULL;
}

// Free old data
void Parcel::free_current_data(void) {
    // save current device and set device to the corresponding GPU
    int current_device = get_current_device();
    cudaSetDevice(this->device_id_);
    // free data
    if (this->data_ != NULL) {
        cudaFree(this->data_);
        this->data_ = NULL;
    }
    // finalize: set back the original GPU and unlock the mutex
    cudaSetDevice(current_device);
}

// Destructor
Parcel::~Parcel(void) {
    this->free_current_data();
}

#endif  // __LIBMERLINCUDA_STATIC__ || __MERLIN_FORCE_STATIC__

#if defined(__LIBMERLINCUDA_STATIC__) || defined(__MERLIN_FORCE_STATIC__)

// Get element at a given C-contiguous index
__cudevice__ float & Parcel::operator[](unsigned long int index) {
    // calculate index vector
    intvec index_ = contiguous_to_ndim_idx(index, this->shape_);
    // calculate strides
    unsigned long int strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(this->data_) + strides);
    return *element_ptr;
}



__cudevice__  void Parcel::copy_to_shared_ptr(Parcel * share_ptr) {
    // copy meta data
    share_ptr->data_ = this->data_;
    share_ptr->ndim_ = this->ndim_;
    // assign shape and strides pointer to data
    share_ptr->shape_.data() = (unsigned long int *) &share_ptr[1];
    share_ptr->shape_.size() = this->ndim_;
    share_ptr->strides_.data() = share_ptr->shape_.data() + this->ndim_;
    share_ptr->strides_.size() = this->ndim_;
    // copy shape and strides
    bool check_zeroth_thread = (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)
                            && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        for (int i = 0; i < this->ndim_; i++) {
            share_ptr->shape_[i] = this->shape_[i];
            share_ptr->strides_[i] = this->strides_[i];
        }
    }
    __syncthreads();
}

#endif  // __LIBMERLINCUDA_STATIC__ || __MERLIN_FORCE_STATIC__

// Get element at a given multidimensional index
/*
 __cudevice__ float & Parcel::operator[](std::initializer_list<unsigned long int> index) {
    // initialize index vector
    intvec index_(index);
    // calculate strides
    unsigned long int strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(this->data_) + strides);
    return *element_ptr;
}
*/

}  // namespace merlin
