// Copyright 2024 quocdang1998
#include "merlin/linalg/aligned_vector.hpp"

#include <cstring>  // std::memset
#include <utility>  // std::exchange, std::swap

#include "merlin/simd/aligned_allocator.hpp"  // merlin::simd::aligned_alloc, merlin::simd::aligned_free
#include "merlin/simd/simd.hpp"               // merlin::simd::size, merlin::simd::alignment

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Aligned vector
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from a size, and fill the vector with zeros
linalg::AlignedVector::AlignedVector(std::uint64_t size) : size_(size) {
    this->compute_capacity();
    this->data_ = simd::aligned_alloc(simd::alignment, simd::size * this->capacity_);
    std::memset(this->data_, 0, simd::alignment * this->capacity_);
}

// Copy constructor
linalg::AlignedVector::AlignedVector(const linalg::AlignedVector & src) : size_(src.size_), capacity_(src.capacity_) {
    this->data_ = simd::aligned_alloc(simd::alignment, simd::size * this->capacity_);
    std::memset(this->data_, 0, simd::alignment * this->capacity_);
    std::copy_n(src.data_, this->size_, this->data_);
}

// Copy assignment
linalg::AlignedVector & linalg::AlignedVector::operator=(const linalg::AlignedVector & src) {
    if (this == &src) {
        return *this;
    }
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        simd::aligned_free(this->data_);
    }
    this->size_ = src.size_;
    this->capacity_ = src.capacity_;
    this->data_ = simd::aligned_alloc(simd::alignment, simd::size * this->capacity_);
    std::memset(this->data_, 0, simd::alignment * this->capacity_);
    std::copy_n(src.data_, this->size_, this->data_);
    return *this;
}

// Move constructor
linalg::AlignedVector::AlignedVector(linalg::AlignedVector && src) {
    this->data_ = std::exchange(src.data_, nullptr);
    this->size_ = std::exchange(src.size_, 0);
    this->capacity_ = std::exchange(src.capacity_, 0);
    this->assigned_ = std::exchange(src.assigned_, false);
}

// Move assignment
linalg::AlignedVector & linalg::AlignedVector::operator=(linalg::AlignedVector && src) {
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
linalg::AlignedVector::~AlignedVector(void) {
    if ((!this->assigned_) && (this->data_ != nullptr)) {
        simd::aligned_free(this->data_);
    }
}

// Compute capacity
void linalg::AlignedVector::compute_capacity(void) { this->capacity_ = (this->size_ + (simd::size - 1)) / simd::size; }

}  // namespace merlin
