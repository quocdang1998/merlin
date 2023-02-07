// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <cinttypes>  // PRIu64

#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// NdData
// --------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
array::NdData::NdData(double * data, std::uint64_t ndim,
                      const intvec & shape, const intvec & strides) :
ndim_(ndim), data_(data), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
array::NdData::NdData(double * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides) :
ndim_(ndim), data_(data), shape_(shape, shape + ndim), strides_(strides, strides + ndim) {}

// Constructor from shape vector
array::NdData::NdData(const intvec & shape) : ndim_(shape.size()), shape_(shape) {
    this->strides_ = array::contiguous_strides(shape, sizeof(double));
}

// Partite an array into multiple parts
Vector<Vector<array::Slice>> array::NdData::partite(std::uint64_t max_memory) {
    // if memory fit in, skip
    std::uint64_t data_size = this->size() * sizeof(double);
    if (data_size < max_memory) {
        return Vector<Vector<array::Slice>>(1, Vector<array::Slice>(this->ndim_));
    }
    // find dimension at which index = 1 -> memory just fit
    intvec size_per_dimension = array::contiguous_strides(this->shape_, sizeof(double));
    std::uint64_t divide_dimension = 0;
    while (size_per_dimension[divide_dimension] > max_memory) {
        --divide_dimension;
    }
    // calculate number of partition
    std::uint64_t num_partition = 1;
    for (std::uint64_t i = 0; i < divide_dimension; i++) {
        num_partition *= this->shape_[i];
    }
    // get slices for each partition
    intvec divident_shape(this->shape_.cbegin(), divide_dimension);  // shape of array of which elements are sub-arrays
    Vector<Vector<array::Slice>> result(num_partition, Vector<array::Slice>(this->ndim_));
    for (std::uint64_t i_partition = 0; i_partition < num_partition; i_partition++) {
        // slices of dividing index
        intvec index = contiguous_to_ndim_idx(i_partition, divident_shape);
        for (std::uint64_t i_dim = 0; i_dim < divident_shape.size(); i_dim++) {
            result[i_partition][i_dim] = array::Slice({index[i_dim]});
        }
    }
    return result;
}

}  // namespace merlin
