// Copyright 2022 quocdang1998
#include "merlin/nddata.hpp"

#include <cstdlib>  // std::abs, div_t, div
#include <cstdint>  // uintptr_t

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::inner_prod, merlin::contiguous_to_ndim_idx

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// NdData
// -------------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
NdData::NdData(float * data, unsigned long int ndim,
               const intvec & shape, const intvec & strides) : ndim_(ndim), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
NdData::NdData(float * data, unsigned long int ndim, const unsigned long int * shape, const unsigned long int * strides) {
    this->ndim_ = ndim;
    this->data_ = data;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
}

// Number of elements
unsigned long int NdData::size(void) {
    unsigned long int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->shape_[i];
    }
    return size;
}

// -------------------------------------------------------------------------------------------------------------------------
// Iterator
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from multi-dimensional index and container
Iterator::Iterator(const intvec & index, NdData & container)  : index_(index), container_(&container) {
    unsigned long int leap = inner_prod(index, container.strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(container.data()) + leap);
}

// Constructor from C-contiguous index
Iterator::Iterator(unsigned long int index, NdData & container) : container_(&container) {
    this->index_ = contiguous_to_ndim_idx(index, this->container_->shape());
    unsigned long int leap = inner_prod(this->index_, container.strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(container.data()) + leap);
}

// Pre-increment operator
Iterator & Iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    unsigned long int current_dim = this->index_.size() - 1;
    intvec & shape = this->container_->shape();
    while (this->index_[current_dim] >= shape[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == shape[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot add more.\n");
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    unsigned long int leap = inner_prod(this->index_, this->container_->strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(this->container_->data()) + leap);
    return *this;
}

// Update index vector to be consistent with the shape
void Iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned long int current_dim = this->index_.size();
    intvec & shape = this->container_->shape();
    for (int i = this->index_.size() - 1; i >= 0; i--) {
        if (this->index_[i] >= shape[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->index_.size()) {  // no update needed
        return;
    }
    // carry the surplus to the dimensions with bigger strides
    while (this->index_[current_dim] >= shape[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == shape[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot add more.\n");
            }
        }
        ldiv_t carry = ldiv(static_cast<long int>(this->index_[current_dim]), static_cast<long int>(shape[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
    // calculate leap and update pointer
    unsigned long int leap = inner_prod(this->index_, this->container_->strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(this->container_->data()) + leap);
}

// -------------------------------------------------------------------------------------------------------------------------
// Slice
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from initializer list
Slice::Slice(std::initializer_list<int> list) {
    const int * list_data = std::data(list);
    switch (list.size()) {
    case 1:  // 1 element = stop
        this->stop_ = list_data[0];
        break;
    case 2:  // 2 element = {start, stop}
        if (list_data[0] < 0) {
            FAILURE(std::invalid_argument, "First element (%d) must be a positive number.\n", list_data[0]);
        }
        this->start_ = list_data[0];
        this->stop_ = list_data[1];
        break;
    case 3:
        if (list_data[0] < 0) {
                FAILURE(std::invalid_argument, "First element (%d) must be a positive number.\n", list_data[0]);
            }
        this->start_ = list_data[0];
        this->stop_ = list_data[1];
        this->step_ = list_data[2];
        break;
    default:
        FAILURE(std::invalid_argument, "Expected intializer list with size at most 3, got %d.\n", list.size());
        break;
    }
}

// Conver Slice to vector of corresponding indices
intvec Slice::range(unsigned long int length) {
    // check validity of start and stop
    if (this->start_ > length) {
        FAILURE(std::length_error, "Start index (%u) bigger than length (%u).\n", this->start_, length);
    } else if (std::abs(this->stop_) > length) {
        FAILURE(std::length_error, "Stop index (%d) must be in range [%d, %u].\n", this->stop_, -length, length);
    }
    // intialize range to be returned
    std::vector<unsigned long int> range;
    unsigned long int stop_index = this->stop_ % length;
    unsigned long int start_index = this->start_;
    if (this->step_ > 0) {
        if (start_index >= stop_index) {
            WARNING("Start (%u) >= stop (%u), return empty slice.\n", start_index, stop_index);
        }
        for (unsigned int i = start_index; i < stop_index; i += this->step_) {
            range.push_back(i);
        }
    } else {
        if (start_index <= stop_index) {
            WARNING("Start (%u) <= stop (%u), return empty slice.\n");
        }
        for (unsigned int i = start_index; i > stop_index; i += this->step_) {
            range.push_back(i);
        }
    }
    return intvec(range.data(), range.size());
}

}  // namespace merlin
