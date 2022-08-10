// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

#include <vector>

#include "merlin/array.hpp"

namespace merlin {

/** @brief A set of multi-dimensional points.*/
class Grid {
  public:
    Grid(void) = default;
    /** @brief Construct an empty grid from a given number of n-dim points.

    @param ndim Number of dimension of points in the grid.
    @param npoint Number of points in the grid.*/
    Grid(unsigned int ndim, unsigned int npoint);
    /** @brief Default destructor.*/
    virtual ~Grid(void) = default;


    /** @brief Reference to array of grid points.*/
    virtual Array grid_points(void);
    /** @brief Number of dimension of each point in the grid.*/
    virtual unsigned int ndim(void) {return this->capacity_points_.dims()[1];}
    /** @brief Number of points in the grid.*/
    virtual unsigned int npoint(void) {return this->npoint_;}
    /** @brief Maximum number of point which the Grid can hold without reallocating memory.*/
    virtual unsigned int capacity(void) {return this->capacity_points_.dims()[0];}


    /** @brief Grid iterator.*/
    using iterator = Array::iterator;
    /** @brief Begin iterator.*/
    virtual Grid::iterator begin(void);
    /** @brief End iterator.*/
    virtual Grid::iterator end(void);
    /** @brief Slicing operator.


    @param index Index of point to get in the grid.*/
    virtual Array operator[] (unsigned int index);
    /** @brief Append a point at the end of the grid.*/
    virtual void push_back(std::vector<float> && point);
    /** @brief Remove a point at the end of the grid.*/
    virtual void pop_back(void);

  protected:
    /** @brief Number of points in the grid.*/
    unsigned int npoint_;
    /** @brief Array to a 2D C-contiguous array of size (capacity, ndim).

    This 2D table store the value of each n-dimensional point as a row vector.

    Capacity is the smallest \f$2^n\f$ so that \f$n_{point} \le 2^n\f$*/
    Array capacity_points_;
    /** @brief Begin iterator.*/
    std::vector<unsigned int> begin_;
    /** @brief End iterator.*/
    std::vector<unsigned int> end_;
};


/** @brief Multi-dimensional cartesian grid.*/
class CartesianGrid : public Grid {
  public:
      /** @brief Constructor from a list of vector of values.*/
    CartesianGrid(const std::vector<std::vector<float>> & grid_vectors);
    /** @brief Default destructor.*/
    ~CartesianGrid(void) = default;

    /** @brief Get grid vectors.*/
    std::vector<std::vector<float>> & grid_vectors(void) {return this->grid_vectors_;}
    /** @brief Full array of each point in the CartesianGrid in form of 2D table.*/
    Array grid_points(void);
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
    Array operator[] (unsigned int index);
    /** @brief Get element at a given index vector.

    @param index Vector of index on each dimension.*/
    Array operator[] (const std::vector<unsigned int> & index);


  protected:
      /** @brief List of vector of values.*/
    std::vector<std::vector<float>> grid_vectors_;
    /** @brief Vector of dimensions.*/
    std::vector<unsigned int> dims_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_
