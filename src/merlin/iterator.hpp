// Copyright 2022 quocdang1998
#ifndef MERLIN_ITERATOR_HPP_
#define MERLIN_ITERATOR_HPP_

#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Iterator of multi-dimensional array and grid.
 *  @details Callable only on CPU.
 */
class MERLIN_EXPORTS Iterator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Iterator(void) = default;
    /** @brief Constructor from multi-dimensional index and container.*/
    Iterator(const intvec & index, array::NdData & container);
    /** @brief Constructor from C-contiguous index.*/
    Iterator(std::uint64_t index, array::NdData & container);
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
    /** @brief Get multi-dimensional index of an iterator.*/
    intvec & index(void) {return this->index_;}
    /** @brief Get constant multi-dimensional index of an iterator.*/
    const intvec & index(void) const {return this->index_;}
    /// @}

    /// @name Operators
    /// @{
    /** @brief Dereference operator of an iterator.*/
    float & operator*(void) {return *(this->item_ptr_);}
    /** @brief Comparison operator.*/
    MERLIN_EXPORTS friend bool operator!=(const Iterator & left, const Iterator & right) {
        return left.item_ptr_ != right.item_ptr_;
    }
    /** @brief Pre-increment operator.*/
    Iterator & operator++(void);
    /** @brief Post-increment operator.*/
    Iterator operator++(int) {return ++(*this);}
    /** @brief Update index vector to be consistent with the shape.*/
    void MERLIN_DEPRECATED update(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Iterator(void) = default;
    /// @}

  protected:
    /** @brief Pointer to item.*/
    float * item_ptr_ = nullptr;
    /** @brief Index vector.*/
    intvec index_;
    /** @brief Pointer to NdData object possessing the item.*/
    array::NdData * container_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_ITERATOR_HPP_
