// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_MATRIX_HPP_
#define MERLIN_LINALG_MATRIX_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <string>   // std::string

#include "merlin/cuda_interface.hpp"      // merlin::linalg::Matrix
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // __cuhostdev__

namespace merlin {

/** @brief Matrix object for linear algebra calculation.*/
class linalg::Matrix {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Matrix(void) {}
    /** @brief Create a matrix from pre-allocated pointer, shape and strides.
     *  @param data Pointer to data.
     *  @param shape Shape vector.
     *  @param strides Strides vector.
     *  @param free_in_destructor De-allocate memory when the matrix is destroyed.
     */
    __cuhostdev__ Matrix(double * data, const std::array<std::uint64_t, 2> & shape,
                         const std::array<std::uint64_t, 2> & strides, bool free_in_destructor = false);
    /** @brief Create an empty matrix of a given shape.
     *  @details The created matrix is C-contiguous.
     *  @param nrow Number of row.
     *  @param ncol Number of column.
     */
    __cuhostdev__ Matrix(std::uint64_t nrow, std::uint64_t ncol);
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
    /** @brief Get strides.*/
    __cuhostdev__ constexpr const std::array<std::uint64_t, 2> & strides(void) const noexcept { return this->strides_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to element at a given row and column index.*/
    __cuhostdev__ double & get(std::uint64_t irow, std::uint64_t icol) noexcept {
        std::uintptr_t destination = reinterpret_cast<std::uintptr_t>(this->data_);
        destination += irow * this->strides_[0] + icol * this->strides_[1];
        return *(reinterpret_cast<double *>(destination));
    }
    /** @brief Get constant reference to element at a given row and column index.*/
    __cuhostdev__ const double & cget(std::uint64_t irow, std::uint64_t icol) const noexcept {
        std::uintptr_t destination = reinterpret_cast<std::uintptr_t>(this->data_);
        destination += irow * this->strides_[0] + icol * this->strides_[1];
        return *(reinterpret_cast<double *>(destination));
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
    __cuhostdev__ ~Matrix(void);
    /// @}

  protected:
    /** @brief Pointer to data of the matrix.*/
    double * data_ = nullptr;
    /** @brief Shape vector.*/
    std::array<std::uint64_t, 2> shape_;
    /** @brief Stride vector.*/
    std::array<std::uint64_t, 2> strides_;
    /** @brief Free data at the end.*/
    bool force_free_ = false;
};

}  // namespace merlin

#endif  // MERLIN_LINALG_MATRIX_HPP_
