// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides, merlin::array::array_copy
#include "merlin/logger.hpp"  // FAILURE

namespace merlin::array {

// -------------------------------------------------------------------------------------------------------------------------
// Parcel
// -------------------------------------------------------------------------------------------------------------------------

// Default constructor
Parcel::Parcel(void) {}

// Constructor from CPU array
Parcel::Parcel(const Array & cpu_array, std::uintptr_t stream) : NdData(cpu_array) {
    // get device id
    this->device_ = device::Device();
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

// Constructor from a slice
Parcel::Parcel(const Parcel & whole, std::initializer_list<Slice> slices)  : NdData(whole, slices) {
    this->force_free = false;
}

// Copy constructor
Parcel::Parcel(const Parcel & src) : NdData(src) {
    // get device id
    this->device_ = device::Device();
    // reform strides vector
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_.id(),
                               std::placeholders::_2, src.device_.id(), std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), copy_func);
}

// Copy assignement
Parcel & Parcel::operator=(const Parcel & src) {
    // free old data
    this->free_current_data();
    // copy metadata and reform strides vector
    this->NdData::operator=(src);
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // allocate data
    cudaError_t err_ = cudaMalloc(&(this->data_), sizeof(float) * this->size());
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "Memory allocation failed with message \"%s\".\n", cudaGetErrorString(err_));
    }
    // create copy function
    auto copy_func = std::bind(cudaMemcpyPeer, std::placeholders::_1, this->device_.id(),
                               std::placeholders::_2, src.device_.id(), std::placeholders::_3);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), copy_func);
    return *this;
}

// Move constructor
Parcel::Parcel(Parcel && src) : NdData(src) {
    // move device id
    this->device_ = src.device_;
    // take over pointer to source
    src.data_ = NULL;
}

// Move assignment
Parcel & Parcel::operator=(Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id
    this->device_ = src.device_;
    // copy metadata
    this->NdData::operator=(src);
    // take over pointer to source
    src.data_ = NULL;
    return *this;
}

// Copy data to a pre-allocated memory
void Parcel::copy_to_gpu(Parcel * gpu_ptr, void * shape_strides_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    Parcel copy_on_gpu;
    // shallow copy of the current object
    copy_on_gpu.data_ = this->data_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.device_ = this->device_;
    // copy temporary object to GPU
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(Parcel), cudaMemcpyHostToDevice);
    // copy shape and strides data
    this->shape_.copy_to_gpu(&(gpu_ptr->shape_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr));
    this->strides_.copy_to_gpu(&(gpu_ptr->strides_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr)+this->ndim_);
    // nullify data pointer to avoid free data
    copy_on_gpu.data_ = NULL;
    copy_on_gpu.shape_.data() = NULL;
    copy_on_gpu.strides_.data() = NULL;
}

// Free old data
void Parcel::free_current_data(void) {
    // lock mutex
    Parcel::m_.lock();
    // save current device and set device to the corresponding GPU
    device::Device current_device = device::Device::get_current_gpu();
    this->device_.set_as_current();
    // free data
    if ((this->data_ != NULL) && this->force_free) {
        cudaFree(this->data_);
        this->data_ = NULL;
    }
    // finalize: set back the original GPU and unlock the mutex
    current_device.set_as_current();
    Parcel::m_.unlock();
}

// Destructor
Parcel::~Parcel(void) {
    this->free_current_data();
}

}  // namespace merlin::array
