// Copyright 2024 quocdang1998
#include "merlin/linalg/vector.hpp"

#include <cstring>  // std::memset
#include <utility>  // std::exchange, std::swap

#include "merlin/linalg/allocator.hpp"  // merlin::linalg::aligned_alloc, merlin::linalg::aligned_free
#include "merlin/linalg/avx.hpp"        // merlin::linalg::pack_size, merlin::linalg::vector_size

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Aligned vector
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from a size, and fill the vector with zeros
linalg::Vector::Vector(std::uint64_t size) : size_(size) {
    this->compute_capacity();
    this->data_ = linalg::aligned_alloc(linalg::vector_size, linalg::vector_size * this->capacity_);
    std::memset(this->data_, 0, linalg::vector_size * this->capacity_);
}

// Copy constructor
linalg::Vector::Vector(const linalg::Vector & src) : size_(src.size_), capacity_(src.capacity_) {
    this->data_ = linalg::aligned_alloc(linalg::vector_size, linalg::vector_size * this->capacity_);
    std::memset(this->data_, 0, linalg::vector_size * this->capacity_);
    std::copy_n(src.data_, this->size_, this->data_);
}

// Copy assignment
linalg::Vector & linalg::Vector::operator=(const linalg::Vector & src) {
    if (this == &src) {
        return *this;
    }
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        linalg::aligned_free(this->data_);
    }
    this->size_ = src.size_;
    this->capacity_ = src.capacity_;
    this->data_ = linalg::aligned_alloc(linalg::vector_size, linalg::vector_size * this->capacity_);
    std::memset(this->data_, 0, linalg::vector_size * this->capacity_);
    std::copy_n(src.data_, this->size_, this->data_);
    return *this;
}

// Move constructor
linalg::Vector::Vector(linalg::Vector && src) {
    this->data_ = std::exchange(src.data_, nullptr);
    this->size_ = std::exchange(src.size_, 0);
    this->capacity_ = std::exchange(src.capacity_, 0);
    this->assigned_ = std::exchange(src.assigned_, false);
}

// Move assignment
linalg::Vector & linalg::Vector::operator=(linalg::Vector && src) {
    if (this == &src) {
        return *this;
    }
    std::swap(this->data_, src.data_);
    std::swap(this->size_, src.size_);
    std::swap(this->capacity_, src.capacity_);
    std::swap(this->assigned_, src.assigned_);
    return *this;
}

// Default destructor
linalg::Vector::~Vector(void) {
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        linalg::aligned_free(this->data_);
    }
}

// Compute capacity
void linalg::Vector::compute_capacity(void) {
    this->capacity_ = (this->size_ + (linalg::pack_size - 1)) / linalg::pack_size;
}

}  // namespace merlin
