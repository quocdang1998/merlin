// Copyright 2022 quocdang1998
#include "merlin/iterator.hpp"

#include <cstdlib>  // std::abs, div_t, div

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::inner_prod, merlin::contiguous_to_ndim_idx, merlin::ndim_to_contiguous_idx

namespace merlin {

// Constructor from multi-dimensional index and container
Iterator::Iterator(const intvec & index, const intvec & shape) : index_(index), shape_(shape) {
    this->item_ptr_ = ndim_to_contiguous_idx(index, shape);
}

// Constructor from C-contiguous index
Iterator::Iterator(std::uint64_t index, const intvec & shape) : item_ptr_(index), shape_(shape) {
    this->index_ = contiguous_to_ndim_idx(index, shape);
}

// Pre-increment operator
Iterator & Iterator::operator++(void) {
    this->index_[this->index_.size()-1]++;
    std::uint64_t current_dim = this->index_.size() - 1;
    while (this->index_[current_dim] >= this->shape_[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == this->shape_[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot increase more.\n");
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    this->item_ptr_ = ndim_to_contiguous_idx(this->index_, this->shape_);
    return *this;
}

// Update index vector to be consistent with the shape
void Iterator::update(void) {
    // detect dimensions having index bigger than dim
    std::uint64_t current_dim = this->index_.size();
    for (int i = this->index_.size() - 1; i >= 0; i--) {
        if (this->index_[i] >= this->shape_[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->index_.size()) {  // no update needed
        return;
    }
    // carry the surplus to the dimensions with bigger strides
    while (this->index_[current_dim] >= this->shape_[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == this->shape_[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot add more.\n");
            }
        }
        ldiv_t carry = ldiv(static_cast<std::int64_t>(this->index_[current_dim]),
                            static_cast<std::int64_t>(this->shape_[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
    // calculate leap and update pointer
    this->item_ptr_ = ndim_to_contiguous_idx(this->index_, this->shape_);
}

}  // namespace merlin
