// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <cinttypes>  // PRIu64

#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// NdData
// --------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
array::NdData::NdData(float * data, std::uint64_t ndim,
                      const intvec & shape, const intvec & strides) : ndim_(ndim), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
array::NdData::NdData(float * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides) {
    this->ndim_ = ndim;
    this->data_ = data;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
}

// Constructor from shape vector
array::NdData::NdData(const intvec & shape) : ndim_(shape.size()), shape_(shape) {
    this->strides_ = contiguous_strides(shape, sizeof(float));
}

// Constructor from a slice
array::NdData::NdData(const array::NdData & whole, const Vector<array::Slice> & slices) {
    // check size
    if (slices.size() != whole.ndim_) {
        CUHDERR(std::invalid_argument, "Dimension of Slices and NdData not compatible (expected %u, got %u).\n",
                static_cast<unsigned int>(whole.ndim_), static_cast<unsigned int>(slices.size()));
    }
    // create result
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(whole.data_);
    this->ndim_ = whole.ndim_;
    this->shape_ = intvec(whole.ndim_);
    this->strides_ = intvec(whole.ndim_);
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim_; i_dim++) {
        auto [offset, shape, stride] = slices[i_dim].slice_on(whole.shape_[i_dim], whole.strides_[i_dim]);
        data_ptr += offset;
        this->shape_[i_dim] = shape;
        this->strides_[i_dim] = stride;
    }
    this->data_ = reinterpret_cast<float *>(data_ptr);
}

// Partite an array into multiple parts
Vector<Vector<array::Slice>> array::NdData::partite(std::uint64_t max_memory) {
    // if memory fit in, skip
    std::uint64_t data_size = this->size() * sizeof(float);
    if (data_size < max_memory) {
        return Vector<Vector<array::Slice>>();
    }
    // find dimension at which index = 1 -> memory just fit
    intvec size_per_dimension = array::contiguous_strides(this->shape_, sizeof(float));
    std::uint64_t divide_dimension = 0;
    while (size_per_dimension[divide_dimension] >= max_memory) {
        --divide_dimension;
    }
    // calculate number of partition
    std::uint64_t num_partition = 1;
    for (std::uint64_t i = 0; i <= divide_dimension; i++) {
        num_partition *= this->shape_[i];
    }
    // get slices for each partition
    intvec sub_shape(this->shape_.cbegin(), divide_dimension);
    Vector<Vector<array::Slice>> result(num_partition, Vector<array::Slice>(this->ndim_));
    for (std::uint64_t i_partition = 0; i_partition < num_partition; i_partition++) {
        // slices of dividing index
        intvec index = contiguous_to_ndim_idx(i_partition, sub_shape);
        for (int i_dim = 0; i_dim < sub_shape.size(); i_dim++) {
            result[i_partition][i_dim].start() = index[i_dim];
            result[i_partition][i_dim].step() = 0;
        }
        // slices of non diving index
        for (int i_dim = sub_shape.size(); i_dim < this->ndim_; i_dim++) {
            result[i_partition][i_dim].start() = 0;
            result[i_partition][i_dim].stop() = INT64_MAX;
            result[i_partition][i_dim].step() = 1;
        }
    }
    return result;
}

}  // namespace merlin
