// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <cinttypes>  // PRIu64
#include <sstream>  // std::ostringstream
#include <vector>  // std::vector
#include <utility>  // std::move

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/array/stock.hpp"  // merlin::array::Stock
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// NdData
// --------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
array::NdData::NdData(double * data, const intvec & shape, const intvec & strides) : data_(data), shape_(shape),
strides_(strides) {
    this->calc_array_size();
    if (!is_same_size(shape, strides)) {
        FAILURE(std::invalid_argument, "Expected size of shape (%" PRIu64 ") equals to size of strides (%" PRIu64
                ").\n", shape.size(), strides.size());
    }
    if (this->ndim() > array::max_allowed_dim) {
        FAILURE(std::invalid_argument, "Only allow array up to %" PRIu64 " dimension, got %" PRIu64 ".\n",
                this->ndim(), array::max_allowed_dim);
    }
}

// Constructor from shape vector
array::NdData::NdData(const intvec & shape) : shape_(shape) {
    this->strides_ = array::contiguous_strides(shape, sizeof(double));
    this->calc_array_size();
    if (this->ndim() > array::max_allowed_dim) {
        FAILURE(std::invalid_argument, "Only allow array up to %" PRIu64 " dimension, got %" PRIu64 ".\n",
                this->ndim(), array::max_allowed_dim);
    }
}

// Get value of element at a n-dim index
double array::NdData::get(const intvec & index) const {return 0.0;}

// Get value of element at a C-contiguous index.
double array::NdData::get(std::uint64_t index) const {return 0.0;}

// Set value of element at a n-dim index.
void array::NdData::set(const intvec index, double value) {}

// Set value of element at a C-contiguous index.
void array::NdData::set(std::uint64_t index, double value) {}

// Partite an array into multiple parts
Vector<Vector<array::Slice>> array::NdData::partite(std::uint64_t max_memory) {
    // if memory fit in, skip
    std::uint64_t data_size = this->size() * sizeof(double);
    if (data_size < max_memory) {
        return Vector<Vector<array::Slice>>(1, Vector<array::Slice>(this->ndim()));
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
    Vector<Vector<array::Slice>> result(num_partition, Vector<array::Slice>(this->ndim()));
    for (std::uint64_t i_partition = 0; i_partition < num_partition; i_partition++) {
        // slices of dividing index
        intvec index = contiguous_to_ndim_idx(i_partition, divident_shape);
        for (std::uint64_t i_dim = 0; i_dim < divident_shape.size(); i_dim++) {
            result[i_partition][i_dim] = array::Slice({index[i_dim]});
        }
    }
    return result;
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
        FAILURE(std::invalid_argument, "Cannot reshape to an array with different size (current size %" PRIu64
                ", new size %" PRIu64 ").\n", this->shape_[0], new_size);
    }
    this->shape_ = new_shape;
    this->strides_ = array::contiguous_strides(new_shape, this->strides_[0]);
}

// Collapse dimension from felt (or right)
void array::NdData::remove_dim(std::uint64_t i_dim) {
    if (this->shape_[i_dim] != 1) {
        return;
    }
    intvec new_shape(this->ndim()-1), new_strides(this->ndim()-1);
    for (std::uint64_t i = 0; i < i_dim; i++) {
        new_shape[i] = this->shape_[i];
        new_strides[i] = this->strides_[i];
    }
    for (std::uint64_t i = i_dim; i < this->ndim()-1; i++) {
        new_shape[i] = this->shape_[i+1];
        new_strides[i] = this->strides_[i+1];
    }
    this->shape_ = std::move(new_shape);
    this->strides_ = std::move(new_strides);
}

// String representation
std::string array::NdData::str(bool first_call) const {
    std::ostringstream os;
    // trivial case
    if (this->ndim() == 1) {
        os << "<";
        for (std::uint64_t i = 0; i < this->shape_[0]; i++) {
            if (i > 0) {
                os << " ";
            }
            os << this->get({i});
        }
        os << ">";
        return os.str();
    }
    // recursively call str function of sub_array
    os << "<";
    if (first_call) {
        if (dynamic_cast<const array::Array *>(this) != nullptr) {
            os << "Array(";
        } else if (dynamic_cast<const array::Parcel *>(this) != nullptr) {
            os << "Parcel(";
        } else if (dynamic_cast<const array::Stock *>(this) != nullptr) {
            os << "Stock(";
        } else {
            os << "NdData(";
        }
    }
    for (std::uint64_t i = 0; i < this->shape_[0]; i++) {
        if (i > 0) {
            os << " ";
        }
        Vector<array::Slice> slice_i(this->ndim());
        slice_i[0] = array::Slice({i});
        array::NdData * p_sliced = array::slice_on(*this, slice_i);
        p_sliced->remove_dim(0);
        os << p_sliced->str(false);
        delete p_sliced;
    }
    if (first_call) {
        os << ")";
    }
    os << ">";
    return os.str();
}

// Destructor
array::NdData::~NdData(void) {}

// Slice an array and get a new instance of the same polymorphic type
array::NdData * array::slice_on(const array::NdData & original, const Vector<array::Slice> & slices) {
    array::NdData * result;
    if (const array::Array * p_ori = dynamic_cast<const array::Array *>(&original); p_ori != nullptr) {
        result = new array::Array(*p_ori, slices);
        return result;
    } else if (const array::Parcel * p_ori = dynamic_cast<const array::Parcel *>(&original); p_ori != nullptr) {
        result = new array::Parcel(*p_ori, slices);
        return result;
    } else if (const array::Stock * p_ori = dynamic_cast<const array::Stock *>(&original); p_ori != nullptr) {
        result = new array::Stock(*p_ori, slices);
        return result;
    }
    result = new array::NdData(original, slices);
    return result;
}

}  // namespace merlin
