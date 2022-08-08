// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

#include <initializer_list>

#include "merlin/nddata.hpp"  // merlin::NdData, merlin::Iterator
#include "merlin/array.hpp"  // merlin::Array
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin {

/** @brief A base class for all kinds of Grid.*/
class Grid {
  public:
    /** @brief Default constructor.*/
    Grid(void) = default;
    /** @brief Destructor.*/
    ~Grid(void) = default;
};

/** @brief A set of multi-dimensional points.*/
class RegularGrid : Grid {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RegularGrid(void) = default;
    /** @brief Construct an empty grid from a given number of n-dim points.
     *  @param npoint Number of points in the grid.
     *  @param ndim Number of dimension of points in the grid.
     */
    RegularGrid(unsigned long int npoint, unsigned long int ndim);
    /** @brief Construct a grid and copy data.
     *  @param points 2D merlin::Array of points, dimension ``(npoints, ndim)``.
     */
    RegularGrid(const Array & points);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    RegularGrid(const RegularGrid & src) = default;
    /** @brief Copy assignment.*/
    RegularGrid & operator=(const RegularGrid & src) = default;
    /** @brief Move constructor.*/
    RegularGrid(RegularGrid && src) = default;
    /** @brief Move assignment.*/
    RegularGrid & operator=(RegularGrid && src) = default;
    /// @}

    /// @name Get members and attributes
    /// @{
    /** @brief Get reference to array of grid points.*/
    Array grid_points(void) const;
    /** @brief Number of dimension of each point in the grid.*/
    unsigned int ndim(void) {return this->points_.shape()[1];}
    /** @brief Number of points in the grid.*/
    unsigned int npoint(void) {return this->npoint_;}
    /** @brief Maximum number of point which the RegularGrid can hold without reallocating memory.*/
    unsigned int capacity(void) {return this->points_.shape()[0];}
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

    /** @brief Get reference Array to a point.
     *  @param index Index of point to get in the grid.
     */
    Array operator[](unsigned int index);
    /** @brief Append a point at the end of the grid.*/
    void push_back(Vector<float> && point);
    /** @brief Remove a point at the end of the grid.*/
    void pop_back(void);

    /** @brief Default destructor.*/
    ~RegularGrid(void) = default;

  protected:
    /** @brief Number of points in the grid.*/
    unsigned long int npoint_;
    /** @brief Tensor to a 2D C-contiguous tensor of size (capacity, ndim).
     *  @details This 2D table store the value of each n-dimensional point as a row vector.
     *
     *  Capacity is the smallest \f$2^n\f$ so that \f$n_{point} \le 2^n\f$
     */
    Array points_;
    /** @brief Begin iterator.*/
    intvec begin_;
    /** @brief End iterator.*/
    intvec end_;
};


/** @brief Multi-dimensional Cartesian grid.*/
class CartesianGrid {
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
    unsigned long int ndim(void) {return this->grid_vectors_.size();}
    /** @brief Number of points in the CartesianGrid.*/
    unsigned long int npoint(void);
    /** @brief Shape of the grid.*/
    intvec grid_shape(void);

    // using iterator = Array::iterator;
    /** @brief Begin iterator.*/
    // CartesianGrid::iterator begin(void);
    /** @brief End iterator.*/
    // CartesianGrid::iterator end(void);

    /** @brief Get element at a given index.

    @param index Index of point in the CartesianGrid::grid_points table.*/
    // Tensor operator[] (unsigned int index);
    /** @brief Get element at a given index vector.

    @param index Vector of index on each dimension.*/
    // Tensor operator[] (const std::vector<unsigned int> & index);


  protected:
    /** @brief List of vector of values.*/
    Vector<floatvec> grid_vectors_;
    /** @brief Begin iterator.*/
    // intvec begin_;
    /** @brief End iterator.*/
    // intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_
