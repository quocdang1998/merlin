// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides, merlin::array::array_copy
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::inner_prod

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Parcel
// --------------------------------------------------------------------------------------------------------------------

// Constructor from CPU array
array::Parcel::Parcel(const array::Array & cpu_array, const cuda::Stream & stream) : array::NdData(cpu_array) {
    // get device id
    this->device_ = stream.get_gpu();
    // allocate data
    cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // cast stream
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(stream.stream());
    // reset strides vector
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               cudaMemcpyHostToDevice, copy_stream);
    // copy data to GPU
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&cpu_array), copy_func);
}

// Constructor from a slice
array::Parcel::Parcel(const array::Parcel & whole, const Vector<array::Slice> & slices) :
array::NdData(whole, slices) {
    this->force_free = false;
}

// Constructor from shape vector
array::Parcel::Parcel(const intvec & shape) : array::NdData(shape) {
    // allocate data
    cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
}

// Copy constructor
array::Parcel::Parcel(const array::Parcel & src) : array::NdData(src) {
    // get device id
    this->device_ = cuda::Device::get_current_gpu();
    // reform strides vector
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyPeer, std::placeholders::_1, this->device_.id(),
                               std::placeholders::_2, src.device_.id(), std::placeholders::_3);
    // copy data to GPU
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
}

// Copy assignement
array::Parcel & array::Parcel::operator=(const array::Parcel & src) {
    // free old data
    this->free_current_data();
    // copy metadata and reform strides vector
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = ::cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyPeer, std::placeholders::_1, this->device_.id(),
                               std::placeholders::_2, src.device_.id(), std::placeholders::_3);
    // copy data to GPU
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
    return *this;
}

// Move constructor
array::Parcel::Parcel(array::Parcel && src) : array::NdData(src) {
    // move device id
    this->device_ = src.device_;
    // take over pointer to source
    src.data_ = nullptr;
}

// Move assignment
array::Parcel & array::Parcel::operator=(array::Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id
    this->device_ = src.device_;
    // copy metadata
    this->array::NdData::operator=(src);
    // take over pointer to source
    src.data_ = nullptr;
    return *this;
}

// Get value of element at a n-dim index
float array::Parcel::get(const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    float result;
    array::Parcel::mutex_.lock();
    cuda::Device current_device = cuda::Device::get_current_gpu();
    this->device_.set_as_current();
    ::cudaMemcpy(&result, reinterpret_cast<float *>(data_ptr), sizeof(float), ::cudaMemcpyDeviceToHost);
    current_device.set_as_current();
    array::Parcel::mutex_.unlock();
    return result;
}

// Get value of element at a C-contiguous index
float array::Parcel::get(std::uint64_t index) const {
    return this->get(contiguous_to_ndim_idx(index, this->shape()));
}

// Set value of element at a n-dim index
void array::Parcel::set(const intvec index, float value) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    array::Parcel::mutex_.lock();
    cuda::Device current_device = cuda::Device::get_current_gpu();
    this->device_.set_as_current();
    ::cudaMemcpy(reinterpret_cast<float *>(data_ptr), &value, sizeof(float), ::cudaMemcpyHostToDevice);
    current_device.set_as_current();
    array::Parcel::mutex_.unlock();
}

// Set value of element at a C-contiguous index
void array::Parcel::set(std::uint64_t index, float value) {
    this->set(contiguous_to_ndim_idx(index, this->shape()), value);
}

// Copy data to a pre-allocated memory
void array::Parcel::copy_to_gpu(array::Parcel * gpu_ptr, void * shape_strides_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    array::Parcel copy_on_gpu;
    // shallow copy of the current object
    copy_on_gpu.data_ = this->data_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.device_ = this->device_;
    // copy temporary object to GPU
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(array::Parcel), cudaMemcpyHostToDevice);
    // copy shape and strides data
    this->shape_.copy_to_gpu(&(gpu_ptr->shape_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr));
    this->strides_.copy_to_gpu(&(gpu_ptr->strides_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr)+this->ndim_);
    // nullify data pointer to avoid free data
    copy_on_gpu.data_ = nullptr;
    copy_on_gpu.shape_.data() = nullptr;
    copy_on_gpu.strides_.data() = nullptr;
}

// Free old data
void array::Parcel::free_current_data(void) {
    // lock mutex
    array::Parcel::mutex_.lock();
    // save current device and set device to the corresponding GPU
    cuda::Device current_device = cuda::Device::get_current_gpu();
    this->device_.set_as_current();
    // free data
    if ((this->data_ != nullptr) && this->force_free) {
        cudaFree(this->data_);
        this->data_ = nullptr;
    }
    // finalize: set back the original GPU and unlock the mutex
    current_device.set_as_current();
    array::Parcel::mutex_.unlock();
}

// Destructor
array::Parcel::~Parcel(void) {
    this->free_current_data();
}

}  // namespace merlin
