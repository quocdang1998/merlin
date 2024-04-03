// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_REGULAR_GRID_HPP_
#define MERLIN_GRID_REGULAR_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"   // merlin::grid::RegularGrid
#include "merlin/vector.hpp"             // merlin::DoubleVec

namespace merlin {

/** @brief Multi-dimensional grid of points.*/
class grid::RegularGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) = default;
    /** @brief Constructor number of points.
     *  @details Allocate an empty grid of points.
     */
    MERLIN_EXPORTS RegularGrid(std::uint64_t num_points, std::uint64_t ndim);
    /** @brief Constructor from an array of point coordinates.
     *  @param point_coordinates 2D array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points and
     *  ``ndim`` is the number of dimension of each point.
     */
    MERLIN_EXPORTS RegularGrid(const array::Array & point_coordinates);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const grid::RegularGrid & src) = default;
    /** @brief Copy assignment.*/
    grid::RegularGrid & operator=(const grid::RegularGrid & src) = default;
    /** @brief Move constructor.*/
    RegularGrid(grid::RegularGrid && src) = default;
    /** @brief Move assignment.*/
    grid::RegularGrid & operator=(grid::RegularGrid && src) = default;
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get pointer to grid data.*/
    __cuhostdev__ constexpr double * grid_data(void) noexcept { return this->grid_data_.data(); }
    /** @brief Get constant pointer to grid data.*/
    __cuhostdev__ constexpr const double * grid_data(void) const noexcept { return this->grid_data_.data(); }
    /** @brief Get dimensions of the grid.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept { return this->ndim_; }
    /** @brief Get total number of points in the grid.*/
    __cuhostdev__ constexpr std::uint64_t size(void) const noexcept { return this->num_points_; }
    /** @brief Get available memory for the number of points in the grid.*/
    __cuhostdev__ constexpr std::uint64_t capacity(void) const noexcept { return this->grid_data_.size(); }
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Write coordinate of point to a pre-allocated memory given flatten index.
     *  @param index Flatten index of point in the grid (in C order).
     *  @param point_data Pointer to memory recording point coordinate.
     */
    __cuhostdev__ void get(std::uint64_t index, double * point_data) const noexcept;
    /** @brief Get element at a given flatten index.
     *  @param index Flatten index of point in the grid (in C order).
     */
    DoubleVec operator[](std::uint64_t index) const noexcept {
        DoubleVec point(this->ndim_);
        this->get(index, point.data());
        return point;
    }
    /// @}

    /// @name Get points
    /// @{
    /** @brief Get all points in the grid.*/
    MERLIN_EXPORTS array::Array get_points(void) const;
    /// @}

    /// @name Destructor
    /// @{
    MERLIN_EXPORTS ~RegularGrid(void);
    /// @}

  protected:
    /** @brief Grid data.*/
    DoubleVec grid_data_;
    /** @brief Number of dimension.*/
    std::uint64_t ndim_;

  private:
    /** @brief Number of points in the grid.*/
    std::uint64_t num_points_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_REGULAR_GRID_HPP_
