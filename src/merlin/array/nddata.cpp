// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <algorithm>  // std::copy, std::fill
#include <cinttypes>  // PRIu64
#include <utility>    // std::move, std::swap
#include <vector>     // std::vector

#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides
#include "merlin/logger.hpp"           // FAILURE
#include "merlin/slice.hpp"            // merlin::Slice

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

// Create sub-array
void array::NdData::create_sub_array(array::NdData & sub_array, const slicevec & slices) const noexcept {
    // check size
    if (slices.size() != this->ndim_) {
        FAILURE(std::invalid_argument, "Dimension of Slices and NdData not compatible.\n");
    }
    // create result
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        auto [offset, shape, stride] = slices[i_dim].slice_on(this->shape_[i_dim], this->strides_[i_dim]);
        data_ptr += offset;
        sub_array.shape_[i_dim] = shape;
        sub_array.strides_[i_dim] = stride;
    }
    sub_array.calc_array_size();
    sub_array.ndim_ = this->ndim_;
    sub_array.data_ = reinterpret_cast<double *>(data_ptr);
    sub_array.release = false;
}

// Member initialization for C++ interface
array::NdData::NdData(double * data, const intvec & shape, const intvec & strides) : data_(data) {
    if (shape.size() > max_dim) {
        FAILURE(std::invalid_argument, "Exceeding maximum ndim (%" PRIu64 ").\n", max_dim);
    }
    if (!is_same_size(shape, strides)) {
        FAILURE(std::invalid_argument, "Shape and strides vectors must have the same size.\n");
    }
    this->ndim_ = shape.size();
    std::copy(shape.begin(), shape.end(), this->shape_.begin());
    std::copy(strides.begin(), strides.end(), this->strides_.begin());
    this->calc_array_size();
}

// Constructor from shape vector
array::NdData::NdData(const intvec & shape) {
    std::copy(shape.begin(), shape.end(), this->shape_.begin());
    this->ndim_ = shape.size();
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    this->calc_array_size();
}

// Check if the array is C-contiguous
bool array::NdData::is_c_contiguous(void) const {
    std::uint64_t c_strides = sizeof(double);
    for (std::int64_t i = this->ndim() - 1; i >= 0; i--) {
        if (this->strides_[i] != c_strides) {
            return false;
        }
        c_strides *= this->shape_[i];
    }
    return true;
}

// Reshape
void array::NdData::reshape(const intvec & new_shape) {
    if (this->ndim_ != 1) {
        FAILURE(std::invalid_argument, "Cannot reshape array of n-dim bigger than 1.\n");
    }
    if (new_shape.size() > max_dim) {
        FAILURE(std::invalid_argument, "Exceeding maximum ndim (%" PRIu64 ").\n", max_dim);
    }
    std::uint64_t new_size = 1;
    for (std::uint64_t i_dim = 0; i_dim < new_shape.size(); i_dim++) {
        new_size *= new_shape[i_dim];
    }
    if (new_size != this->size_) {
        FAILURE(std::invalid_argument,
                "Cannot reshape to an array with different size (current size %" PRIu64 ", new size %" PRIu64 ").\n",
                this->shape_[0], new_size);
    }
    std::copy(new_shape.begin(), new_shape.end(), this->shape_.begin());
    this->ndim_ = new_shape.size();
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
}

// Collapse dimension from felt (or right)
void array::NdData::remove_dim(std::uint64_t i_dim) {
    if (this->shape_[i_dim] != 1) {
        WARNING("Cannot remove dimension with size differ than 1.\n");
        return;
    }
    std::copy(this->shape_.begin() + i_dim + 1, this->shape_.begin() + this->ndim_, this->shape_.begin() + i_dim);
    std::copy(this->strides_.begin() + i_dim + 1, this->strides_.begin() + this->ndim_, this->strides_.begin() + i_dim);
    this->ndim_ -= 1;
    this->shape_[this->ndim_] = 0;
    this->strides_[this->ndim_] = 0;
}

// Collapse all dimensions with size 1
void array::NdData::squeeze(void) {
    std::uint64_t new_ndim = 0;
    for (std::uint64_t i = 0; i < this->ndim_; i++) {
        if (this->shape_[i] != 1) {
            this->shape_[new_ndim] = this->shape_[i];
            this->strides_[new_ndim] = this->strides_[i];
            new_ndim++;
        }
    }
    std::fill(this->shape_.data() + new_ndim, this->shape_.data() + this->ndim_, 0);
    std::fill(this->strides_.data() + new_ndim, this->strides_.data() + this->ndim_, 0);
    this->ndim_ = new_ndim;
}

// String representation
std::string array::NdData::str(bool first_call) const {
    return array::print(this, "NdData", first_call);
}

// Destructor
array::NdData::~NdData(void) {}

}  // namespace merlin
