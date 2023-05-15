// Copyright 2022 quocdang1998
#ifndef MERLIN_ITERATOR_HPP_
#define MERLIN_ITERATOR_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Iterator of multi-dimensional objects.*/
class Iterator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Iterator(void) = default;
    /** @brief Constructor from multi-dimensional index and shape of the container.*/
    MERLIN_EXPORTS Iterator(const intvec & index, const intvec & shape);
    /** @brief Constructor from contiguous index and shape of the container.*/
    MERLIN_EXPORTS Iterator(std::uint64_t index, const intvec & shape);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Iterator(const Iterator & src) = default;
    /** @brief Copy assignment.*/
    Iterator & operator=(const Iterator & src) = default;
    /** @brief Move constructor.*/
    Iterator(Iterator && src) = default;
    /** @brief Move assignment.*/
    Iterator & operator=(Iterator && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get constant reference to multi-dimensional index.*/
    constexpr const intvec & index(void) const noexcept {return this->index_;}
    /** @brief Get constant reference to contiguous index.*/
    constexpr const std::uint64_t & contiguous_index(void) const noexcept {return this->item_ptr_;}
    /// @}

    /// @name Operators
    /// @{
    /** @brief Comparison operator.*/
    MERLIN_EXPORTS friend bool operator!=(const Iterator & left, const Iterator & right) noexcept {
        return left.item_ptr_ != right.item_ptr_;
    }
    /** @brief Pre-increment operator.*/
    MERLIN_EXPORTS Iterator & operator++(void);
    /** @brief Post-increment operator.*/
    Iterator operator++(int) {return ++(*this);}
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Iterator(void) = default;
    /// @}

  protected:
    /** @brief Pointer to item.*/
    std::uint64_t item_ptr_ = 0;
    /** @brief Index vector.*/
    intvec index_;
    /** @brief Shape of the object.*/
    intvec shape_;

  private:
    /** @brief Update index vector to be consistent with the shape (deprecated).*/
    void MERLIN_DEPRECATED update(void);
};

}  // namespace merlin

#endif  // MERLIN_ITERATOR_HPP_
