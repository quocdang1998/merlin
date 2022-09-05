// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

#include <vector>

#include "merlin/array.hpp"  // merlin::Array
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief A set of multi-dimensional points.*/
class Grid {
  public:
    /** @brief Default constructor.*/
    Grid(void) = default;
    /** @brief Construct an empty grid from a given number of n-dim points.
     *  @param npoint Number of points in the grid.
     *  @param ndim Number of dimension of points in the grid.
     */
    Grid(unsigned long int npoint, unsigned long int ndim);
    /** @brief Construct a grid and copy data.
     *  @param points 2D merlin::Array of points, dimension ``(npoints, ndim)``.
     */
    Grid(const Array & points);
    /** @brief Default destructor.*/
    virtual ~Grid(void) = default;

    /** @brief Get reference to array of grid points.*/
    virtual Array grid_points(void) const;
    /** @brief Number of dimension of each point in the grid.*/
    virtual unsigned int ndim(void) {return this->points_.shape()[1];}
    /** @brief Number of points in the grid.*/
    virtual unsigned int npoint(void) {return this->npoint_;}
    /** @brief Maximum number of point which the Grid can hold without reallocating memory.*/
    virtual unsigned int capacity(void) {return this->points_.shape()[0];}


    /** @brief Grid iterator.*/
    using iterator = Array::iterator;
    /** @brief Begin iterator.*/
    virtual Grid::iterator begin(void);
    /** @brief End iterator.*/
    virtual Grid::iterator end(void);
    /** @brief Slicing operator.


    /** @brief Get reference Array to a point.
     *  @param index Index of point to get in the grid.
     */
    virtual Tensor operator[](unsigned int index);
    /** @brief Append a point at the end of the grid.*/
    // virtual void push_back(std::vector<float> && point);
    /** @brief Remove a point at the end of the grid.*/
    // virtual void pop_back(void);

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

#ifdef __COMMENT__
/** @brief Multi-dimensional cartesian grid.*/
class CartesianGrid : public Grid {
  public:
      /** @brief Constructor from a list of vector of values.*/
    CartesianGrid(const std::vector<std::vector<float>> & grid_vectors);
    /** @brief Default destructor.*/
    ~CartesianGrid(void) = default;

    /** @brief Get grid vectors.*/
    std::vector<std::vector<float>> & grid_vectors(void) {return this->grid_vectors_;}
    /** @brief Full tensor of each point in the CartesianGrid in form of 2D table.*/
    Tensor grid_points(void);
    /** @brief Number of dimension of the CartesianGrid.*/
    unsigned int ndim(void);
    /** @brief Number of points in the CartesianGrid.*/
    unsigned int npoint(void);
    /** @brief Yield error.*/
    unsigned int capacity(void);

    /** @brief Begin iterator.*/
    Grid::iterator begin(void);
    /** @brief End iterator.*/
    Grid::iterator end(void);

    /** @brief Get element at a given index.

    @param index Index of point in the CartesianGrid::grid_points table.*/
    Tensor operator[] (unsigned int index);
    /** @brief Get element at a given index vector.

    @param index Vector of index on each dimension.*/
    Tensor operator[] (const std::vector<unsigned int> & index);


  protected:
      /** @brief List of vector of values.*/
    std::vector<std::vector<float>> grid_vectors_;
    /** @brief Vector of dimensions.*/
    std::vector<unsigned int> dims_;
};
#endif

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_
