// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin::array {

// Member initialization for C++ interface
NdData::NdData(float * data, std::uint64_t ndim,
               const intvec & shape, const intvec & strides) : ndim_(ndim), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
NdData::NdData(float * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides) {
    this->ndim_ = ndim;
    this->data_ = data;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
}

// Number of elements
std::uint64_t NdData::size(void) {
    std::uint64_t size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->shape_[i];
    }
    return size;
}

}  // namespace merlin::array
