// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

#include <initializer_list>
#include <vector>
#include <cstdio>

#include "merlin/array.hpp"

namespace merlin {

/** @brief A set of multi-dimensional points.*/ 
class Grid {
  public:
    /** @brief Construct an empty grid from a given number of n-dim points.
    
    @param ndim Number of dimension of points in the grid.
    @param npoint Number of points in the grid.*/
    Grid(unsigned int ndim, unsigned int npoint);
    /** @brief Default destructor.*/
    virtual ~Grid(void) {
        std::printf("Freeing grid object.\n");
    }

    /** @brief Reference to array of grid points.*/
    Array & grid_points(void) {return this->grid_points_;}
    /** @brief %Constant reference to array of grid points.*/
    const Array & grid_points(void) const {return this->grid_points_;}
    /** @brief Number of dimension of each point in the grid.*/
    unsigned int ndim(void) {return this->ndim_;}
    /** @brief Number of points in the grid.*/
    unsigned int npoint(void) {return this->npoint_;}

    /** @brief Grid iterator.*/
    using iterator = Array::iterator;
    /** @brief Begin iterator.*/
    virtual Grid::iterator begin(void) {return this->grid_points_.begin();}
    /** @brief End iterator.*/
    virtual Grid::iterator end(void) {return this->grid_points_.end();}
    /** @brief Slicing operator.
    
    @param index Index of point to get in the grid.*/
    virtual std::vector<float> operator[] (unsigned int index);

  protected:
    /** @brief Number of dimensions of the grid.*/
    unsigned int ndim_;
    /*! \brief Number of points in the grid.*/
    unsigned int npoint_;
    /*! \brief Maximum number of points  can be add to the grid without reallocate a brand new
    memory zone.*/
    unsigned int capacity_;
    /*! \brief Array to a 2D C-contiguous array of size (npoint, ndim).
    
    This 2D table store the value of each n-dimensional point as a row vector.*/
    Array grid_points_;
};

class CartesianGrid : public Grid {
public:
    CartesianGrid(std::initializer_list<Array> grid_vectors);
    CartesianGrid(std::initializer_list<std::vector<float>> grid_vectors);



protected:
    std::vector<Array> grid_vectors_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_