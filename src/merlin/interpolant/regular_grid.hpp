// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_REGULAR_GRID_HPP_
#define MERLIN_INTERPOLANT_REGULAR_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/interpolant/grid.hpp"  //  merlin::interpolant::Grid
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec

namespace merlin {

/** @brief A set of multi-dimensional points.*/
class MERLIN_EXPORTS interpolant::RegularGrid : interpolant::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) {}
    /** @brief Construct an empty grid from a given number of n-dim points.
     *  @param npoint Number of points in the grid.
     *  @param ndim Number of dimension of points in the grid.
     */
    RegularGrid(std::uint64_t npoint, std::uint64_t ndim);
    /** @brief Construct a grid and copy data from an array.
     *  @param points 2D merlin::Array of points, dimension ``(npoints, ndim)``.
     */
    RegularGrid(const array::Array & points);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const interpolant::RegularGrid & src);
    /** @brief Copy assignment.*/
    interpolant::RegularGrid & operator=(const interpolant::RegularGrid & src);
    /** @brief Move constructor.*/
    RegularGrid(interpolant::RegularGrid && src);
    /** @brief Move assignment.*/
    interpolant::RegularGrid & operator=(interpolant::RegularGrid && src);
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get reference to array of grid points.*/
    array::Array & grid_points(void) noexcept {return *(dynamic_cast<array::Array *>(this->points_));}
    /** @brief Get constant reference to array of grid points.*/
    const array::Array & grid_points(void) const noexcept {return *(dynamic_cast<array::Array *>(this->points_));}
    /** @brief Number of dimension of each point in the grid.*/
    constexpr std::uint64_t ndim(void) const noexcept {return this->points_->shape()[1];}
    /** @brief Number of points in the grid.*/
    constexpr std::uint64_t size(void) const noexcept {return this->npoint_;}
    /** @brief Maximum number of point which the RegularGrid can hold without reallocating memory.*/
    constexpr std::uint64_t capacity(void) const noexcept {return this->points_->shape()[0];}
    /// @}

    /// @name Iterator
    /// @{
    /** @brief RegularGrid iterator.*/
    using iterator = Iterator;
    /** @brief Begin iterator.*/
    RegularGrid::iterator begin(void);
    /** @brief End iterator.*/
    RegularGrid::iterator end(void);
    /// @}

    /// @name Modify points
    /// @{
    /** @brief Get reference Array to a point.
     *  @param index Index of point to get in the grid.
     */
    Vector<double> operator[](std::uint64_t index);
    /** @brief Append a point at the end of the grid.*/
    void push_back(Vector<double> && point);
    /** @brief Remove a point at the end of the grid.*/
    void pop_back(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~RegularGrid(void);
    /// @}

  protected:
    /** @brief Number of points in the grid.*/
    std::uint64_t npoint_;
    /** @brief Begin iterator.*/
    interpolant::RegularGrid::iterator begin_;
    /** @brief End iterator.*/
    interpolant::RegularGrid::iterator end_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_REGULAR_GRID_HPP_
