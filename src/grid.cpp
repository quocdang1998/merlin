// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <vector>

namespace merlin {

Grid::Grid(unsigned int ndim, unsigned int npoint) {
    float * data = new float[ndim*npoint];
    unsigned int dims[2] = {npoint, ndim};
    unsigned int strides[2] = {ndim*sizeof(float), sizeof(float)};
    this->grid_points_ = Array(data, 2, dims, strides, false);
    this->grid_points_.is_copy = true;
}

}  // namespace merlin