// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <cinttypes>  // PRIu64
#include <utility>    // std::move
#include <vector>     // std::vector

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/array/stock.hpp"      // merlin::array::Stock
#include "merlin/logger.hpp"           // FAILURE
#include "merlin/slice.hpp"            // merlin::Slice
#include "merlin/utils.hpp"            // merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// NdData
// ---------------------------------------------------------------------------------------------------------------------

// Calculate size of array
void array::NdData::calc_array_size(void) noexcept {
    this->size_ = 1;
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        this->size_ *= this->shape_[i];
    }
}

// Member initialization for C++ interface
array::NdData::NdData(double * data, const intvec & shape, const intvec & strides) :
data_(data), shape_(shape), strides_(strides) {
    this->calc_array_size();
    if (!is_same_size(shape, strides)) {
        FAILURE(std::invalid_argument,
                "Expected size of shape (%" PRIu64 ") equals to size of strides (%" PRIu64 ").\n", shape.size(),
                strides.size());
    }
}

// Constructor from shape vector
array::NdData::NdData(const intvec & shape) : shape_(shape) {
    this->strides_ = array::contiguous_strides(shape, sizeof(double));
    this->calc_array_size();
}

// Constructor from a slice
array::NdData::NdData(const array::NdData & whole, const slicevec & slices) {
    // check size
    if (slices.size() != whole.ndim()) {
        FAILURE(std::invalid_argument, "Dimension of Slices and NdData not compatible (expected %u, got %u).\n",
                static_cast<unsigned int>(whole.ndim()), static_cast<unsigned int>(slices.size()));
    }
    // create result
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(whole.data_);
    this->shape_ = intvec(whole.ndim());
    this->strides_ = intvec(whole.ndim());
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim(); i_dim++) {
        auto [offset, shape, stride] = slices[i_dim].slice_on(whole.shape_[i_dim], whole.strides_[i_dim]);
        data_ptr += offset;
        this->shape_[i_dim] = shape;
        this->strides_[i_dim] = stride;
    }
    this->calc_array_size();
    this->data_ = reinterpret_cast<double *>(data_ptr);
    this->release_ = false;
}

// Reshape
void array::NdData::reshape(const intvec & new_shape) {
    if (this->ndim() != 1) {
        FAILURE(std::invalid_argument, "Cannot reshape array of n-dim bigger than 1.\n");
    }
    std::uint64_t new_size = 1;
    for (std::uint64_t i_dim = 0; i_dim < new_shape.size(); i_dim++) {
        new_size *= new_shape[i_dim];
    }
    if (new_size != this->shape_[0]) {
        FAILURE(std::invalid_argument,
                "Cannot reshape to an array with different size (current size %" PRIu64 ", new size %" PRIu64 ").\n",
                this->shape_[0], new_size);
    }
    this->shape_ = new_shape;
    this->strides_ = array::contiguous_strides(new_shape, this->strides_[0]);
}

// Collapse dimension from felt (or right)
void array::NdData::remove_dim(std::uint64_t i_dim) {
    if (this->shape_[i_dim] != 1) {
        return;
    }
    intvec new_shape(this->ndim() - 1), new_strides(this->ndim() - 1);
    for (std::uint64_t i = 0; i < i_dim; i++) {
        new_shape[i] = this->shape_[i];
        new_strides[i] = this->strides_[i];
    }
    for (std::uint64_t i = i_dim; i < this->ndim() - 1; i++) {
        new_shape[i] = this->shape_[i + 1];
        new_strides[i] = this->strides_[i + 1];
    }
    this->shape_ = std::move(new_shape);
    this->strides_ = std::move(new_strides);
}

// String representation
std::string array::NdData::str(bool first_call) const {
    return array::print(this, "NdData", first_call);
}

// Destructor
array::NdData::~NdData(void) {}

}  // namespace merlin
