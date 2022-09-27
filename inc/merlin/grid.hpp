// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/nddata.hpp"  // merlin::NdData, merlin::Iterator
#include "merlin/array.hpp"  // merlin::Array
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief A base class for all kinds of Grid.*/
class MERLIN_EXPORTS Grid {
  public:
    /** @brief Default constructor.*/
    __cuhostdev__ Grid(void) {}
    /** @brief Destructor.*/
    __cuhostdev__ ~Grid(void) {}

  protected:
    /** @brief Array holding coordinates of points in the Grid.*/
    NdData * points_ = NULL;
};

/** @brief A set of multi-dimensional points.*/
class MERLIN_EXPORTS RegularGrid : Grid {
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
    RegularGrid(const Array & points);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const RegularGrid & src);
    /** @brief Copy assignment.*/
    RegularGrid & operator=(const RegularGrid & src);
    /** @brief Move constructor.*/
    RegularGrid(RegularGrid && src);
    /** @brief Move assignment.*/
    RegularGrid & operator=(RegularGrid && src);
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get reference to array of grid points.*/
    Array grid_points(void) const {return *(dynamic_cast<Array *>(this->points_));}
    /** @brief Number of dimension of each point in the grid.*/
    std::uint64_t ndim(void) const {return this->points_->shape()[1];}
    /** @brief Number of points in the grid.*/
    std::uint64_t size(void) const {return this->npoint_;}
    /** @brief Maximum number of point which the RegularGrid can hold without reallocating memory.*/
    std::uint64_t capacity(void) const {return this->points_->shape()[0];}
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
    Array operator[](unsigned int index);
    /** @brief Append a point at the end of the grid.*/
    void push_back(Vector<float> && point);
    /** @brief Remove a point at the end of the grid.*/
    void pop_back(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~RegularGrid(void) {
        if (this->points_ != NULL) {
            delete this->points_;
        }
    }
    /// @}

  protected:
    /** @brief Number of points in the grid.*/
    std::uint64_t npoint_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
};

/** @brief Multi-dimensional Cartesian grid.*/
class MERLIN_EXPORTS CartesianGrid : public Grid {
  public:
      /** @brief Constructor from a list of vector of values.*/
    CartesianGrid(std::initializer_list<floatvec> grid_vectors);
    /** @brief Default destructor.*/
    ~CartesianGrid(void) = default;

    /** @brief Get grid vectors.*/
    Vector<floatvec> & grid_vectors(void) {return this->grid_vectors_;}
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    Array grid_points(void);
    /** @brief Number of dimension of the CartesianGrid.*/
    std::uint64_t ndim(void) {return this->grid_vectors_.size();}
    /** @brief Number of points in the CartesianGrid.*/
    std::uint64_t size(void);
    /** @brief Shape of the grid.*/
    intvec grid_shape(void);

    using iterator = Iterator;
    /** @brief Begin iterator.*/
    CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    CartesianGrid::iterator end(void);

    /** @brief Get element at a given index.
     *  @param index Index of point in the CartesianGrid::grid_points table.
     */
    floatvec operator[](std::uint64_t index);
    /** @brief Get element at a given index vector.
     *  @param index Vector of index on each dimension.
     */
    floatvec operator[](const intvec & index);

  protected:
    /** @brief List of vector of values.*/
    Vector<floatvec> grid_vectors_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_
