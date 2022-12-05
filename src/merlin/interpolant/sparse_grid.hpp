// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_SPARSE_GRID_HPP_
#define MERLIN_INTERPOLANT_SPARSE_GRID_HPP_

#include <initializer_list>  // std::initializer_list

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::Grid, merlin::interpolant::CartesianGrid
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief Get the level from a given valid size.*/
__cuhostdev__ std::uint64_t get_level_from_size(std::uint64_t size) noexcept;

/** @brief Index at a given level of a 1D grid.
 *  @param level Level to get index.
 *  @param size Size of 1D grid level.
 */
__cuhostdev__ intvec hiearchical_index(std::uint64_t level, std::uint64_t size);

/** @brief A set of multi-dimensional points.*/
class MERLIN_EXPORTS interpolant::SparseGrid : interpolant::Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    SparseGrid(void) = default;
    /** @brief Constructor isotropic sparse grid from vector of components.*/
    SparseGrid(std::initializer_list<floatvec> grid_vectors);
    /** @brief Constructor anisotropic grid from vector of components.*/
    SparseGrid(std::initializer_list<floatvec> grid_vectors, std::uint64_t max, const intvec & weight);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ SparseGrid(const interpolant::SparseGrid & src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Copy assignment.*/
    __cuhostdev__ interpolant::SparseGrid & operator=(const interpolant::SparseGrid & src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /** @brief Move constructor.*/
    __cuhostdev__ SparseGrid(interpolant::SparseGrid && src) : grid_vectors_(src.grid_vectors_) {}
    /** @brief Move assignment.*/
    __cuhostdev__ interpolant::SparseGrid & operator=(interpolant::SparseGrid && src) {
        this->grid_vectors_ = src.grid_vectors_;
        return *this;
    }
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get grid vectors.*/
    __cuhostdev__ const Vector<floatvec> & grid_vectors(void) const {return this->grid_vectors_;}
    /** @brief Number of dimension of the SparseGrid.*/
    __cuhostdev__ std::uint64_t ndim(void) {return this->grid_vectors_.size();}
    /** @brief List of valid level vectors.*/
    __cuhostdev__ const Vector<intvec> & level_vectors(void) {return this->level_vectors_;}
    /** @brief Number of points in the SparseGrid.*/
    // __cuhostdev__ std::uint64_t size(void);
    /// @}

    /// @name Grid levels
    /// @{
    /** @brief Max level on each dimension.*/
    __cuhostdev__ intvec max_levels(void);
    /** @brief Calculate and save a list of valid level vectors taken in the grid.*/
    void calc_level_vectors(void);
    /** @brief Get Cartesian Grid corresponding to a given level vector.*/
    interpolant::CartesianGrid get_cartesian_grid(const intvec & level_vector);
    /// @}

    /// @name Iterator
    /// @{
    /** @brief RegularGrid iterator.*/
    // using iterator = Iterator;
    /** @brief Begin iterator.*/
    // SparseGrid::iterator begin(void);
    /** @brief End iterator.*/
    // SparseGrid::iterator end(void);
    /// @}

    /// @name Modify points
    /// @{

    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~SparseGrid(void);
    /// @}

  protected:
    /** @brief List of vector of values.*/
    Vector<floatvec> grid_vectors_;
    /** @brief Max level.*/
    std::uint64_t max_;
    /** @brief Weight vector.*/
    intvec weight_;

    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
    /** @brief Valid level vectors.*/
    Vector<intvec> level_vectors_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_SPARSE_GRID_HPP_
