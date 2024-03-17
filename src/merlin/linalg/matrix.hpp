// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_MATRIX_HPP_
#define MERLIN_LINALG_MATRIX_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/cuda_interface.hpp"      // __cuhostdev__
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix

namespace merlin {

/** @brief Matrix object for linear algebra calculation.
 *  @details Memory must be shaped as column-contiguous matrix.
 *  @note This object will not handle memory allocation. Memory must be explicitly handled outside of this class.
 */
class linalg::Matrix {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Matrix(void) {}
    /** @brief Create a matrix from pre-allocated pointer and shape.
     *  @note The matrix assigned is assumed to be Fortran-contiguous.
     *  @param data Pointer to data.
     *  @param shape Shape vector ``[nrow, ncol]``.
     */
    __cuhostdev__ Matrix(double * data, const std::array<std::uint64_t, 2> & shape) : data_(data), shape_(shape) {
        this->ld_ = shape[0];
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get pointer to data.*/
    __cuhostdev__ constexpr double * data(void) noexcept { return this->data_; }
    /** @brief Get constant pointer to data.*/
    __cuhostdev__ constexpr const double * data(void) const noexcept { return this->data_; }
    /** @brief Get number of rows.*/
    __cuhostdev__ constexpr const std::uint64_t & nrow(void) const noexcept { return this->shape_[0]; }
    /** @brief Get number of columns.*/
    __cuhostdev__ constexpr const std::uint64_t & ncol(void) const noexcept { return this->shape_[1]; }
    /** @brief Get leading dimension.*/
    __cuhostdev__ constexpr const std::uint64_t & lead_dim(void) const noexcept { return this->ld_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to element at a given row and column index.*/
    __cuhostdev__ double & get(std::uint64_t irow, std::uint64_t icol) noexcept {
        return *(this->data_ + irow * this->ld_ + icol);
    }
    /** @brief Get constant reference to element at a given row and column index.*/
    __cuhostdev__ const double & cget(std::uint64_t irow, std::uint64_t icol) const noexcept {
        return *(this->data_ + irow * this->ld_ + icol);
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
    __cuhostdev__ inline ~Matrix(void) {}
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
