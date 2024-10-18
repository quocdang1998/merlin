// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_ALIGNED_VECTOR_HPP_
#define MERLIN_LINALG_ALIGNED_VECTOR_HPP_

#include <algorithm>         // std::copy, std::copy_n
#include <concepts>          // std::convertible_to
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list, std::data
#include <iterator>          // std::distance, std::forward_iterator, std::iter_reference_t

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::AlignedVector
#include "merlin/vector.hpp"              // merlin::DoubleView

namespace merlin {

// Aligned vector
// --------------

/** @brief Aligned allocated vector
 *  @details The memory allocated for the vector is guaranteed to be aligned to the requirement of the longest SIMD
 *  intrinsic instructions.
 */
class linalg::AlignedVector {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    AlignedVector(void) = default;
    /** @brief Constructor from a size, and fill the vector with zeros.*/
    MERLIN_EXPORTS AlignedVector(std::uint64_t size);
    /** @brief Constructor by copying data from a range.*/
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const double &>
    AlignedVector(Pointer first, Pointer last) : linalg::AlignedVector::AlignedVector(std::distance(first, last)) {
        std::copy(first, last, this->data_);
    }
    /** @brief Constructor by copying data from a pointer and its size.*/
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const double &>
    AlignedVector(Pointer data, std::uint64_t size) : linalg::AlignedVector::AlignedVector(size) {
        std::copy_n(data, size, this->data_);
    }
    /** @brief Constructor from initializer list.*/
    AlignedVector(std::initializer_list<double> data) :
    linalg::AlignedVector::AlignedVector(std::data(data), data.size()) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS AlignedVector(const linalg::AlignedVector & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS linalg::AlignedVector & operator=(const linalg::AlignedVector & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS AlignedVector(linalg::AlignedVector && src);
    /** @brief Move assignment.*/
    MERLIN_EXPORTS linalg::AlignedVector & operator=(linalg::AlignedVector && src);
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get pointer of data.*/
    constexpr double * data(void) noexcept { return this->data_; }
    /** @brief Get constant pointer of data.*/
    constexpr const double * data(void) const noexcept { return this->data_; }
    /** @brief Get constant reference to size.*/
    constexpr const std::uint64_t & size(void) const noexcept { return this->size_; }
    /** @brief Get capacity divided by the pack size.*/
    constexpr std::uint64_t capacity(void) const noexcept { return this->capacity_; }
    /// @}

    /// @name Get view
    /// @{
    /** @brief Get a view corresponding to the vector.*/
    constexpr DoubleView get_view(void) const { return DoubleView(this->data_, this->size_); }
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const { return this->get_view().str(); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~AlignedVector(void);
    /// @}

  protected:
    /** @brief Pointer to data.*/
    double * data_ = nullptr;
    /** @brief Size of the vector.*/
    std::uint64_t size_ = 0;
    /** @brief Capacity divided by the pack size (i.e. the number of register passes to fully traverse the vector).*/
    std::uint64_t capacity_ = 0;
    /** @brief Flag indicating whether to free the memory in the destructor.*/
    bool assigned_ = false;

  private:
    /** @brief Compute capacity.*/
    void compute_capacity(void);
};

}  // namespace merlin

#endif  // MERLIN_LINALG_ALIGNED_VECTOR_HPP_
