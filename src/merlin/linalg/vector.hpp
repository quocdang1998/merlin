// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_VECTOR_HPP_
#define MERLIN_LINALG_VECTOR_HPP_

#include <algorithm>         // std::copy, std::copy_n
#include <concepts>          // std::convertible_to
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list, std::data
#include <iterator>          // std::distance, std::forward_iterator, std::iter_reference_t

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linlag::Vector
#include "merlin/vector.hpp"              // merlin::DoubleView

namespace merlin {

// Aligned vector
// --------------

/** @brief Aligned allocated vector
 *  @details The memory associated to the vector is guaranteed to be aligned to the requirement of the longest SIMD
 *  intrinsic instructions.
 */
class linalg::Vector {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Vector(void) = default;
    /** @brief Constructor from a size, and fill the vector with zeros.*/
    MERLIN_EXPORTS Vector(std::uint64_t size);
    /** @brief Constructor by copying data from a range.*/
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const double &>
    Vector(Pointer first, Pointer last) : linalg::Vector::Vector(std::distance(first, last)) {
        std::copy(first, last, this->data_);
    }
    /** @brief Constructor by copying data from a pointer and its size.*/
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const double &>
    Vector(Pointer data, std::uint64_t size) : linalg::Vector::Vector(size) {
        std::copy_n(data, size, this->data_);
    }
    /** @brief Constructor from initializer list.*/
    Vector(std::initializer_list<double> data) : linalg::Vector::Vector(std::data(data), data.size()) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS Vector(const linalg::Vector & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS linalg::Vector & operator=(const linalg::Vector & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Vector(linalg::Vector && src);
    /** @brief Move assignment.*/
    MERLIN_EXPORTS linalg::Vector & operator=(linalg::Vector && src);
    /// @}

    /// @name Get view
    /// @{
    /** @brief Get a view corresponding to the vector.*/
    constexpr DoubleView get_view(void) const {
        return DoubleView(this->data_, this->size_);
    }
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const { return this->get_view().str(); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~Vector(void);
    /// @}

  protected:
    /** @brief Pointer to data.*/
    double * data_ = nullptr;
    /** @brief Size of the vector.*/
    std::uint64_t size_ = 0;
    /** @brief Capacity divided by 4.*/
    std::uint64_t capacity_ = 0;
    /** @brief Flag indicating whether to free the memory in the destructor.*/
    bool assigned_ = false;

  private:
    /** @brief Compute capacity.*/
    void compute_capacity(void);
};

}  // namespace merlin

#endif  // MERLIN_LINALG_VECTOR_HPP_
