// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_REGULAR_GRID_HPP_
#define MERLIN_GRID_REGULAR_GRID_HPP_

#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <string>            // std::string

#include "merlin/cuda_interface.hpp"    // __cuhostdev__
#include "merlin/exports.hpp"           // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"  // merlin::grid::RegularGrid
#include "merlin/vector.hpp"            // merlin::floatvec, merlin::Vector

namespace merlin {

/** @brief Multi-dimensional grid of points.*/
class grid::RegularGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) = default;
    /** @brief Constructor number of points.*/
    RegularGrid(std::uint64_t num_points, std::uint64_t n_dim) :
    n_dim_(n_dim), num_points_(num_points), grid_data_(num_points * n_dim) {}
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS RegularGrid(const grid::RegularGrid & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS grid::RegularGrid & operator=(const grid::RegularGrid & src);
    /** @brief Move constructor.*/
    RegularGrid(grid::RegularGrid && src) = default;
    /** @brief Move assignment.*/
    grid::RegularGrid & operator=(grid::RegularGrid && src) = default;
    /// @}

  protected:
    /** @brief Grid data.*/
    floatvec grid_data_;
    /** @brief Number of dimension.*/
    std::uint64_t n_dim_;

  private:
    /** @brief Number of points in the grid.*/
    std::uint64_t num_points_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_REGULAR_GRID_HPP_
