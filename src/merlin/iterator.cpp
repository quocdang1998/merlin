// Copyright 2022 quocdang1998
#include "merlin/iterator.hpp"

#include <cstdlib>  // std::abs, div_t, div

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::inner_prod, merlin::contiguous_to_ndim_idx

namespace merlin {

// Constructor from multi-dimensional index and container
Iterator::Iterator(const intvec & index, array::NdData & container)  : index_(index), container_(&container) {
    std::uint64_t leap = inner_prod(index, container.strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(container.data()) + leap);
}

// Constructor from C-contiguous index
Iterator::Iterator(std::uint64_t index, array::NdData & container) : container_(&container) {
    this->index_ = contiguous_to_ndim_idx(index, this->container_->shape());
    std::uint64_t leap = inner_prod(this->index_, container.strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(container.data()) + leap);
}

// Pre-increment operator
Iterator & Iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    std::uint64_t current_dim = this->index_.size() - 1;
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
    std::uint64_t leap = inner_prod(this->index_, this->container_->strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->container_->data()) + leap);
    return *this;
}

// Update index vector to be consistent with the shape
void Iterator::update(void) {
    // detect dimensions having index bigger than dim
    std::uint64_t current_dim = this->index_.size();
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
        ldiv_t carry = ldiv(static_cast<std::int64_t>(this->index_[current_dim]),
                            static_cast<std::int64_t>(shape[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
    // calculate leap and update pointer
    std::uint64_t leap = inner_prod(this->index_, this->container_->strides());
    this->item_ptr_ = reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(this->container_->data()) + leap);
}

}  // namespace merlin
