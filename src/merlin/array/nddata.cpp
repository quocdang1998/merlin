// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <algorithm>  // std::copy

#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides
#include "merlin/logger.hpp"           // merlin::Fatal, merlin::Warning
#include "merlin/utils.hpp"            // merlin::prod_elements

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
void array::NdData::create_sub_array(array::NdData & sub_array, const SliceArray & slices) const noexcept {
    // create result
    sub_array.shape_.resize(this->ndim());
    sub_array.strides_.resize(this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        auto [offset, shape, stride] = slices[i_dim].slice_on(this->shape_[i_dim], this->strides_[i_dim]);
        data_ptr += offset;
        sub_array.shape_[i_dim] = shape;
        sub_array.strides_[i_dim] = stride;
    }
    sub_array.data_ = reinterpret_cast<double *>(data_ptr);
    sub_array.calc_array_size();
    sub_array.release = false;
}

// Member initialization for C++ interface
array::NdData::NdData(double * data, const Index & shape, const Index & strides) :
data_(data), shape_(shape), strides_(strides) {
    if (shape.size() != strides.size()) {
        Fatal<std::invalid_argument>("Shape and strides vectors must have the same size.\n");
    }
    this->calc_array_size();
}

// Constructor from shape vector
array::NdData::NdData(const Index & shape) : shape_(shape) {
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
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
void array::NdData::reshape(const Index & new_shape) {
    if (this->ndim() != 1) {
        Fatal<std::invalid_argument>("Cannot reshape array of n-dim bigger than 1.\n");
    }
    std::uint64_t new_size = prod_elements(new_shape.data(), new_shape.size());
    if (new_size != this->size_) {
        Fatal<std::invalid_argument>("Cannot reshape to an array with different size (current size {}, new size {}).\n",
                                     this->shape_[0], new_size);
    }
    this->shape_ = new_shape;
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
}

// Collapse dimension from felt (or right)
void array::NdData::remove_dim(std::uint64_t i_dim) {
    if (this->shape_[i_dim] != 1) {
        Warning("Cannot remove dimension with size differ than 1.\n");
        return;
    }
    std::copy(this->shape_.begin() + i_dim + 1, this->shape_.end(), this->shape_.begin() + i_dim);
    std::copy(this->strides_.begin() + i_dim + 1, this->strides_.end(), this->strides_.begin() + i_dim);
    std::uint64_t new_ndim = this->ndim() - 1;
    this->shape_.resize(new_ndim);
    this->strides_.resize(new_ndim);
}

// Collapse all dimensions with size 1
void array::NdData::squeeze(void) {
    std::uint64_t new_ndim = 0;
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        if (this->shape_[i] != 1) {
            this->shape_[new_ndim] = this->shape_[i];
            this->strides_[new_ndim] = this->strides_[i];
            new_ndim++;
        }
    }
    this->shape_.resize(new_ndim);
    this->strides_.resize(new_ndim);
}

// String representation
std::string array::NdData::str(bool first_call) const { return array::print(this, "NdData", first_call); }

// Destructor
array::NdData::~NdData(void) {}

}  // namespace merlin
