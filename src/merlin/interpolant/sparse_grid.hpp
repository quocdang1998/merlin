// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_SPARSE_GRID_HPP_
#define MERLIN_INTERPOLANT_SPARSE_GRID_HPP_

#include <initializer_list>  // std::initializer_list
#include <string>  // std::string

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::Grid, merlin::interpolant::CartesianGrid
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec

namespace merlin {

/** @brief Sparse grid.
 *  @details A set of point in multi-dimensional space based on hierarchical basis. Here, the sparse grid is composed
 *  of disjointed union of many multi-dimensional Cartesian grids, each associated with a level index vector (an array
 *  of level on each dimension). Each point in the grid belongs to a sub-grid, and associated to an index in the grid.
 */
class MERLIN_EXPORTS interpolant::SparseGrid : interpolant::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    SparseGrid(void) = default;
    /** @brief Constructor a full sparse grid from vector of components.*/
    SparseGrid(std::initializer_list<Vector<double>> grid_vectors);
    /** @brief Constructor anisotropic grid from vector of components.*/
    SparseGrid(std::initializer_list<Vector<double>> grid_vectors, std::uint64_t max, const intvec & weight);
    /** @brief Constructor sparse grid from vector of components and level index vectors.*/
    SparseGrid(std::initializer_list<Vector<double>> grid_vectors, const Vector<intvec> & level_vectors);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ SparseGrid(const interpolant::SparseGrid & src);
    /** @brief Copy assignment.*/
    __cuhostdev__ interpolant::SparseGrid & operator=(const interpolant::SparseGrid & src);
    /** @brief Move constructor.*/
    __cuhostdev__ SparseGrid(interpolant::SparseGrid && src);
    /** @brief Move assignment.*/
    __cuhostdev__ interpolant::SparseGrid & operator=(interpolant::SparseGrid && src);
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vectors.*/
    __cuhostdev__ constexpr const Vector<Vector<double>> & grid_vectors(void) const noexcept {
        return this->grid_vectors_;
    }
    /** @brief Number of dimension of the SparseGrid.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept {return this->grid_vectors_.size();}
    /** @brief Get shape of the grid.*/
    __cuhostdev__ intvec get_grid_shape(void) const noexcept;
    /** @brief Get start index of each sub-grid.*/
    __cuhostdev__ constexpr const intvec & sub_grid_start_index(void) const noexcept {
        return this->sub_grid_start_index_;
    }
    /** @brief Number of points in the SparseGrid.*/
    __cuhostdev__ std::uint64_t size(void) const;
    /// @}

    /// @name Grid levels
    /// @{
    /** @brief Get number of Cartesian sub-grid.*/
    __cuhostdev__ constexpr std::uint64_t num_level(void) const noexcept {
        return this->level_index_.size() / this->ndim();
    }
    /** @brief Get level vector at a given index.
     *  @param index Index of level.
     */
    /** @brief List of index of first point of each Cartesian sub-grid.*/
    __cuhostdev__ intvec level_index(std::uint64_t index) noexcept {
        intvec result;
        result.assign(&(this->level_index_[index*this->ndim()]), this->ndim());
        return result;
    }
    /** @brief Get constant level vector at a given index.
     *  @param index Index of level.
     */
    __cuhostdev__ const intvec level_index(std::uint64_t index) const noexcept {
        intvec result;
        result.assign(const_cast<std::uint64_t *>(&(this->level_index_[index*this->ndim()])), this->ndim());
        return result;
    }
    /** @brief Max level on each dimension.*/
    __cuhostdev__ intvec max_levels(void) const;
    /** @brief Get Cartesian Grid corresponding to a given level vector.*/
    // interpolant::CartesianGrid get_cartesian_grid(const intvec & level_vector) const;
    /// @}

    /// @name Iterator
    /// @{
    /** @brief Index of point in grid given its contiguous order.*/
    __cuhostdev__ intvec index_from_contiguous(std::uint64_t contiguous_index) const;
    /** @brief Point at a given multi-dimensional index.*/
    __cuhostdev__ Vector<double> point_at_index(const intvec & index) const;
    /// @}

    /// @name Representation
    /// @{
    std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~SparseGrid(void);
    /// @}

  protected:
    /** @brief List of vector of grid values.*/
    Vector<Vector<double>> grid_vectors_;
    /** @brief Level vectors.*/
    intvec level_index_;
    /** @brief Index relative to the full sparse grid of the first point of the Cartesian sub-grid corresponding to
     *  each level index vector.
     */
    intvec sub_grid_start_index_;
};

namespace interpolant {

/** @brief Copy value from multi dimensional value array to compacted array for sparse grid interpolation.*/
void copy_value_from_cartesian_array(array::NdData & dest, const array::NdData & src,
                                     const interpolant::SparseGrid & grid);

/** @brief Get Cartesian grid corresponding to a given level vector.*/
interpolant::CartesianGrid get_cartesian_grid(const SparseGrid & grid, std::uint64_t subgrid_index);

}  // namespace interpolant


}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_SPARSE_GRID_HPP_
