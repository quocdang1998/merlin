// Copyright 2023 quocdang1998
#ifndef SPGRID_SPARSE_GRID_HPP_
#define SPGRID_SPARSE_GRID_HPP_

#include <type_traits>  // std::add_pointer
#include <functional>   // std::function
#include <string>       // std::string

#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/vector.hpp"               // merlin::Vector, merlin::floatvec, merlin::intvec

#include "spgrid/declaration.hpp"  // spgrid::SparseGrid

namespace spgrid {

/** @brief Sparse grid.
 *  @details A set of point in multi-dimensional space based on hierarchical basis. Here, the sparse grid is composed
 *  of disjointed union of many multi-dimensional Cartesian grids, each associated with a level index vector (an array
 *  of level on each dimension). Each point in the grid belongs to a sub-grid, and associated to an index in the grid.
 */
class SparseGrid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    SparseGrid(void) = default;
    /** @brief Constructor a full sparse grid from vectors of components.*/
    SparseGrid(const merlin::Vector<merlin::floatvec> & full_grid_vectors,
               const std::function<bool(const merlin::intvec &)> accept_condition);
    /** @brief Constructor sparse grid from vector of components and level index vectors.*/
    SparseGrid(const merlin::Vector<merlin::floatvec> & full_grid_vectors, const merlin::intvec & level_vectors);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    SparseGrid(const SparseGrid & src) = default;
    /** @brief Copy assignment.*/
    SparseGrid & operator=(const SparseGrid & src) = default;
    /** @brief Move constructor.*/
    SparseGrid(SparseGrid && src) = default;
    /** @brief Move assignment.*/
    SparseGrid & operator=(SparseGrid && src) = default;
    /// @}

    /// @name Members and attributes
    /// @{
    /** @brief Number of dimension of the grid.*/
    constexpr std::uint64_t ndim(void) const noexcept { return this->full_grid_.ndim(); }
    /** @brief Number of level of the hierarchical grid.*/
    constexpr std::uint64_t nlevel(void) const noexcept { return this->nd_level_.size() / this->full_grid_.ndim(); }
    /** @brief Shape of full Cartesian grid.*/
    constexpr const merlin::intvec & shape(void) const noexcept { return this->full_grid_.shape(); }
    /** @brief Get reference to the full Cartesian grid.*/
    constexpr const merlin::grid::CartesianGrid & fullgrid(void) const noexcept { return this->full_grid_; }
    /// @}

    /// @name Hierarchical levels
    /// @{
    /** @brief Get a view to the level vector at a given index.
     *  @param index Index of level.
     */
    merlin::intvec get_ndlevel_at_index(std::uint64_t index) const noexcept {
        merlin::intvec ndlevel;
        ndlevel.assign(const_cast<std::uint64_t *>(&(this->nd_level_[index * this->ndim()])), this->ndim());
        return ndlevel;
    }
    /** @brief Get Cartesian grid at a given level index.
     *  @param index Index of level.
     */
    merlin::grid::CartesianGrid get_grid_at_level(std::uint64_t index) const noexcept;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~SparseGrid(void) = default;
    /// @}

  private:
    /** @brief Full Cartesian grid.*/
    merlin::grid::CartesianGrid full_grid_;
    /** @brief Vector of multi-dimensional level.*/
    merlin::intvec nd_level_;
};

}  // namespace spgrid

#endif  // SPGRID_SPARSE_GRID_HPP_
