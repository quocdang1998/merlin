// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

namespace merlin {

#ifdef __NVCC__

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

#endif

}