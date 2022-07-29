// Copyright 2022 quocdang1998
#ifndef MERLIN_GRID_HPP_
#define MERLIN_GRID_HPP_

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
    virtual ~Grid(void);

    Array & grid_points(void) {return this->grid_points_;}
    const Array & grid_points(void) const {return this->grid_points_;}

    using iterator = Array::iterator;
    virtual Grid::iterator begin(void) {return this->grid_points_.begin();}
    virtual Grid::iterator end(void) {return this->grid_points_.end();}
    virtual std::vector<float> operator[] (unsigned int index) {
        std::vector<unsigned int> index_grid = {index, 0};
        float * target = &(this->grid_points_[index_grid]);
        return std::vector<float>(target, target+this->ndim_);
    }

  protected:
    /** @brief Number of dimensions of the grid.*/
    unsigned int ndim_;
    /*! \brief Number of points in the grid.*/
    unsigned int npoint_;
    /*! \brief Array to a 2D C-contiguous array of size (npoint, ndim).
    
    This 2D table store the value of each n-dimensional point as a row vector.*/.
    Array grid_points_;
};

}  // namespace merlin

#endif  // MERLIN_GRID_HPP_