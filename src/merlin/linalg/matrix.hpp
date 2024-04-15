// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_MATRIX_HPP_
#define MERLIN_LINALG_MATRIX_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix

namespace merlin {

// Dynamically allocated matrix
// ----------------------------

/** @brief Matrix object for linear algebra calculation.*/
class linalg::Matrix {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    MERLIN_EXPORTS Matrix(void) = default;
    /** @brief Create an empty matrix from shape.
     *  @param nrow Number of rows.
     *  @param ncol Number of columns.
     */
    MERLIN_EXPORTS Matrix(std::uint64_t nrow, std::uint64_t ncol);
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get pointer to data.*/
    constexpr double * data(void) noexcept { return this->data_; }
    /** @brief Get constant pointer to data.*/
    constexpr const double * data(void) const noexcept { return this->data_; }
    /** @brief Get number of rows.*/
    constexpr const std::uint64_t & nrow(void) const noexcept { return this->shape_[0]; }
    /** @brief Get number of columns.*/
    constexpr const std::uint64_t & ncol(void) const noexcept { return this->shape_[1]; }
    /** @brief Get leading dimension.*/
    constexpr const std::uint64_t & lead_dim(void) const noexcept { return this->ld_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to element at a given row and column index.*/
    constexpr double & get(std::uint64_t irow, std::uint64_t icol) noexcept {
        return *(this->data_ + irow + this->ld_ * icol);
    }
    /** @brief Get constant reference to element at a given row and column index.*/
    constexpr const double & cget(std::uint64_t irow, std::uint64_t icol) const noexcept {
        return *(this->data_ + irow + this->ld_ * icol);
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~Matrix(void);
    /// @}

  protected:
    /** @brief Pointer to data of the matrix.*/
    double * data_ = nullptr;
    /** @brief Shape vector.*/
    std::array<std::uint64_t, 2> shape_;
    /** @brief Leading dimension.*/
    std::uint64_t ld_;
};

}  // namespace merlin

#endif  // MERLIN_LINALG_MATRIX_HPP_
