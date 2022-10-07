// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/array.hpp"  // merlin::Array
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/array/utils.hpp"  // merlin::inner_prod, merlin::contiguous_strides,
                             // merlin::get_current_device, merlin::contiguous_to_ndim_idx
                             // merlin::array_copy
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

#ifdef __MERLIN_LIBCUDA__
// Get element at a given C-contiguous index
__cudevice__ float & array::Parcel::operator[](std::uint64_t index) {
    // calculate index vector
    intvec index_ = contiguous_to_ndim_idx(index, this->shape_);
    // calculate strides
    std::uint64_t strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->data_) + strides);
    return *element_ptr;
}

// Get element at a given Nd index
__cudevice__ float & array::Parcel::operator[](std::initializer_list<std::uint64_t> index) {
    // initialize index vector
    intvec index_(index);
    // calculate strides
    std::uint64_t strides = inner_prod(index_, this->strides_);
    float * element_ptr = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->data_) + strides);
    return *element_ptr;
}

// Copy to shared memory
__cudevice__ void array::Parcel::copy_to_shared_mem(array::Parcel * share_ptr, void * shape_strides_ptr) {
    // copy meta data
    bool check_zeroth_thread = (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (check_zeroth_thread) {
        share_ptr->data_ = this->data_;
        share_ptr->ndim_ = this->ndim_;
    }
    __syncthreads();
    // copy shape and strides
    this->shape_.copy_to_shared_mem(&(share_ptr->shape_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr));
    this->strides_.copy_to_shared_mem(&(share_ptr->strides_),
                                      reinterpret_cast<std::uint64_t *>(shape_strides_ptr)+this->ndim_);
}
#endif  // __MERLIN_LIBCUDA__

#ifdef __MERLIN_DLLCUDA__
// Default constructor
array::Parcel::Parcel(void) {}

// Constructor from CPU array
array::Parcel::Parcel(const array::Array & cpu_array, std::uintptr_t stream) : array::NdData(cpu_array) {
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
    array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&cpu_array), copy_func);
}

// Check if current device holds data pointed by object
int array::Parcel::check_device(void) const {
    return (this->device_id_ - get_current_device());
}

// Copy constructor
array::Parcel::Parcel(const array::Parcel & src) : array::NdData(src) {
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
    array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
}

// Copy assignement
array::Parcel & array::Parcel::operator=(const array::Parcel & src) {
    // free old data
    this->free_current_data();
    // copy metadata and reform strides vector
    this->array::NdData::operator=(src);
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
    array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
    return *this;
}

// Move constructor
array::Parcel::Parcel(array::Parcel && src) : array::NdData(src) {
    // move device id
    this->device_id_ = src.device_id_;
    // take over pointer to source
    src.data_ = NULL;
}

// Move assignment
array::Parcel & array::Parcel::operator=(array::Parcel && src) {
    // free old data
    this->free_current_data();
    // move device id
    this->device_id_ = src.device_id_;
    // copy metadata
    this->array::NdData::operator=(src);
    // take over pointer to source
    src.data_ = NULL;
    return *this;
}

// Copy data to a pre-allocated memory
void array::Parcel::copy_to_gpu(array::Parcel * gpu_ptr, void * shape_strides_ptr) {
    // initialize buffer to store data of the copy before cloning it to GPU
    array::Parcel copy_on_gpu;
    // shallow copy of the current object
    copy_on_gpu.data_ = this->data_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.device_id_ = this->device_id_;
    // copy temporary object to GPU
    cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(array::Parcel), cudaMemcpyHostToDevice);
    // copy shape and strides data
    this->shape_.copy_to_gpu(&(gpu_ptr->shape_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr));
    this->strides_.copy_to_gpu(&(gpu_ptr->strides_), reinterpret_cast<std::uint64_t *>(shape_strides_ptr)+this->ndim_);
    // nullify data pointer to avoid free data
    copy_on_gpu.data_ = NULL;
    copy_on_gpu.shape_.data() = NULL;
    copy_on_gpu.strides_.data() = NULL;
}

// Free old data
void array::Parcel::free_current_data(void) {
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
array::Parcel::~Parcel(void) {
    this->free_current_data();
}

#endif  // __MERLIN_DLLCUDA__

}  // namespace merlin
