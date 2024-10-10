// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_ITERATOR_HELPERS_HPP_
#define MERLIN_VECTOR_ITERATOR_HELPERS_HPP_

#include <cstddef>           // std::ptrdiff_t
#include <iterator>          // std::random_access_iterator_tag

#include "merlin/config.hpp"              // __cuhostdev__, __cudevice__
#include "merlin/vector/declaration.hpp"  // merlin::vector::ForwardIterator, merlin::vector::ReverseIterator

namespace merlin {

// Forward iterator
// ----------------

/** @brief Helper class for forward iterator.*/
template <typename T>
class vector::ForwardIterator {
  public:
    /// @name Type traits
    /// @{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;
    /// @}

    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ constexpr ForwardIterator(void) {}
    /** @brief Constructor from pointer value.*/
    __cuhostdev__ constexpr ForwardIterator(T * ptr) : ptr_(ptr) {}
    /// @}

    /// @name Dereference
    /// @{
    __cuhostdev__ constexpr T & operator*() const { return *(this->ptr_); }
    __cuhostdev__ constexpr T * operator->() { return this->ptr_; }
    /// @}

    /// @name Increment
    /// @{
    __cuhostdev__ constexpr vector::ForwardIterator<T> & operator++() {
        ++(this->ptr_);
        return *this;
    }
    __cuhostdev__ constexpr vector::ForwardIterator<T> operator++(int) {
        vector::ForwardIterator<T> temp = *this;
        ++(*this);
        return temp;
    }
    /// @}

    /// @name Decrement
    /// @{
    __cuhostdev__ constexpr vector::ForwardIterator<T> & operator--() {
        --(this->ptr_);
        return *this;
    }
    __cuhostdev__ constexpr vector::ForwardIterator<T> operator--(int) {
        vector::ForwardIterator<T> temp = *this;
        --(*this);
        return temp;
    }
    /// @}

    /// @name Arithmetic
    /// @{
    friend __cuhostdev__ constexpr vector::ForwardIterator<T> operator+(const vector::ForwardIterator<T> & it,
                                                                        std::ptrdiff_t offset) {
        return vector::ForwardIterator<T>(it.ptr_ + offset);
    }
    friend __cuhostdev__ constexpr vector::ForwardIterator<T> operator-(const vector::ForwardIterator<T> & it,
                                                                        std::ptrdiff_t offset) {
        return vector::ForwardIterator<T>(it.ptr_ - offset);
    }
    friend __cuhostdev__ constexpr std::ptrdiff_t operator-(const vector::ForwardIterator<T> & lhs,
                                                            const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ - rhs.ptr_;
    }
    /// @}

    /// @name Arithmetic assignment
    /// @{
    __cuhostdev__ constexpr vector::ForwardIterator<T> & operator+=(std::ptrdiff_t offset) {
        this->ptr_ += offset;
        return *this;
    }
    __cuhostdev__ constexpr vector::ForwardIterator<T> & operator-=(std::ptrdiff_t offset) {
        this->ptr_ -= offset;
        return *this;
    }
    __cuhostdev__ constexpr T & operator[](std::ptrdiff_t index) const { return *(this->ptr_ + index); }
    /// @}

    /// @name Comparison
    /// @{
    friend __cuhostdev__ constexpr bool operator==(const vector::ForwardIterator<T> & lhs,
                                                   const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ == rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator!=(const vector::ForwardIterator<T> & lhs,
                                                   const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ != rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator<(const vector::ForwardIterator<T> & lhs,
                                                  const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ < rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator<=(const vector::ForwardIterator<T> & lhs,
                                                   const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ <= rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator>(const vector::ForwardIterator<T> & lhs,
                                                  const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ > rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator>=(const vector::ForwardIterator<T> & lhs,
                                                   const vector::ForwardIterator<T> & rhs) {
        return lhs.ptr_ >= rhs.ptr_;
    }
    /// @}

  private:
    /** @brief Underlying pointer.*/
    T * ptr_ = nullptr;
};

// Reverse iterator
// ----------------

/** @brief Helper class for reverse iterator.*/
template <typename T>
class vector::ReverseIterator {
  public:
    /// @name Type traits
    /// @{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;
    /// @}

    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ constexpr ReverseIterator(void) {}
    /** @brief Constructor from pointer value.*/
    __cuhostdev__ constexpr ReverseIterator(T * ptr) : ptr_(ptr - 1) {}
    /// @}

    /// @name Dereference
    /// @{
    __cuhostdev__ constexpr T & operator*() const { return *ptr_; }
    __cuhostdev__ constexpr T * operator->() { return ptr_; }
    /// @}

    /// @name Increment
    /// @{
    __cuhostdev__ constexpr vector::ReverseIterator<T> & operator++() {
        --(this->ptr_);
        return *this;
    }
    __cuhostdev__ constexpr vector::ReverseIterator<T> operator++(int) {
        vector::ReverseIterator<T> temp = *this;
        --(*this);
        return temp;
    }
    /// @}

    /// @name Decrement
    /// @{
    __cuhostdev__ constexpr vector::ReverseIterator<T> & operator--() {
        ++(this->ptr_);
        return *this;
    }
    __cuhostdev__ constexpr vector::ReverseIterator<T> operator--(int) {
        vector::ReverseIterator<T> temp = *this;
        ++(*this);
        return temp;
    }
    /// @}

    /// @name Arithmetic
    /// @{
    friend __cuhostdev__ constexpr vector::ReverseIterator<T> operator+(const vector::ReverseIterator<T> & it,
                                                                        std::ptrdiff_t offset) {
        return vector::ReverseIterator<T>(it.ptr_ - offset);
    }
    friend __cuhostdev__ constexpr vector::ReverseIterator<T> operator-(const vector::ReverseIterator<T> & it,
                                                                        std::ptrdiff_t offset) {
        return vector::ReverseIterator<T>(it.ptr_ + offset);
    }
    friend __cuhostdev__ constexpr std::ptrdiff_t operator-(const vector::ReverseIterator<T> & lhs,
                                                            const vector::ReverseIterator<T> & rhs) {
        return rhs.ptr_ - lhs.ptr_;
    }
    /// @}

    /// @name Arithmetic assignment
    /// @{
    __cuhostdev__ constexpr vector::ReverseIterator<T> & operator+=(std::ptrdiff_t offset) {
        this->ptr_ -= offset;
        return *this;
    }
    __cuhostdev__ constexpr vector::ReverseIterator<T> & operator-=(std::ptrdiff_t offset) {
        this->ptr_ += offset;
        return *this;
    }
    __cuhostdev__ constexpr T & operator[](std::ptrdiff_t index) const { return *(this->ptr_ - index); }
    /// @}

    /// @name Comparison
    /// @{
    friend __cuhostdev__ constexpr bool operator==(const vector::ReverseIterator<T> & lhs,
                                                   const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ == rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator!=(const vector::ReverseIterator<T> & lhs,
                                                   const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ != rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator<(const vector::ReverseIterator<T> & lhs,
                                                  const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ > rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator<=(const vector::ReverseIterator<T> & lhs,
                                                   const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ >= rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator>(const vector::ReverseIterator<T> & lhs,
                                                  const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ < rhs.ptr_;
    }
    friend __cuhostdev__ constexpr bool operator>=(const vector::ReverseIterator<T> & lhs,
                                                   const vector::ReverseIterator<T> & rhs) {
        return lhs.ptr_ <= rhs.ptr_;
    }
    /// @}

  private:
    /** @brief Underlying pointer.*/
    T * ptr_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_VECTOR_ITERATOR_HELPERS_HPP_
