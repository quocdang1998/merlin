// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <utility>  // std::move

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::array_copy, merlin::contiguous_to_ndim_idx

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// RegularGrid
// -------------------------------------------------------------------------------------------------------------------------

// Construct an empty grid from a given number of n-dim points
RegularGrid::RegularGrid(unsigned long int npoint, unsigned long int ndim) : npoint_(npoint) {
    // calculate capacity
    unsigned long int capacity = 1;
    while (capacity < npoint) {
        capacity <<= 1;
    }
    // allocate data
    this->points_ = Array({capacity, ndim});
}

// Construct a grid and copy data
RegularGrid::RegularGrid(const Array & points) {
    // check if original data is 2D
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected an Array of dimension 2.\n");
    }
    // calculate capacity
    this->npoint_ = points.shape()[0];
    unsigned long int capacity = 1;
    while (capacity < this->npoint_) {
        capacity <<= 1;
    }
    // copy data from old array to new array
    this->points_ = Array({capacity, points.ndim()});
    array_copy(&(this->points_), &points, std::memcpy);
}

// Get reference to array of grid points
Array RegularGrid::grid_points(void) const {
    const unsigned long int * shape_ptr = &(this->points_.shape()[0]);
    const unsigned long int * strides_ptr = &(this->points_.strides()[0]);
    return Array(this->points_.data(), 2, shape_ptr, strides_ptr, false);
}

// Get reference Array to a point
Array RegularGrid::operator[](unsigned int index) {
    float * target_ptr = &(this->points_[{index, 0}]);
    unsigned long int shape = this->points_.ndim();
    unsigned long int strides = sizeof(float);
    return Array(target_ptr, 1, &shape, &strides, false);
}

// Begin iterator
RegularGrid::iterator RegularGrid::begin(void) {
    this->begin_ = intvec(2, 0);
    this->end_ = intvec(2, 0);
    this->end_[0] = this->npoint_;
    return RegularGrid::iterator(this->begin_, this->points_);
}

// End iterator
RegularGrid::iterator RegularGrid::end(void) {
    return RegularGrid::iterator(this->end_, this->points_);
}

// Append a point at the end of the grid
void RegularGrid::push_back(Vector<float> && point) {
    // check size of point
    if (point.size() != this->ndim()) {
        FAILURE(std::invalid_argument, "Cannot add point of dimension %d to a grid of dimension %d.",
                point.size(), this->ndim());
    }
    // add point to grid
    this->npoint_ += 1;
    if (this->npoint_ > this->capacity()) {
        // reallocate data
        Array new_location = Array({2*this->capacity(), this->ndim()});
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->points_.data(), sizeof(float)*this->npoint_*this->ndim());
        this->points_ = std::move(new_location);
    }
    std::memcpy(&(this->points_[{this->npoint_-1, 0}]), &(point[0]),
                sizeof(float)*this->ndim());
}

// Remove a point at the end of the grid
void RegularGrid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity()/2) {
        // reallocate data
        Array new_location = Array({this->capacity()/2, this->ndim()});
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->points_.data(), sizeof(float)*this->npoint_*this->ndim());
        this->points_ = std::move(new_location);
    } else {
        std::memset(&(this->points_[{this->npoint_, 0}]), 0,
                    sizeof(float)*this->ndim());
    }
}


// -------------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// -------------------------------------------------------------------------------------------------------------------------

// Construct from a list of vector of values
CartesianGrid::CartesianGrid(std::initializer_list<floatvec> grid_vectors) : grid_vectors_(grid_vectors) {
    this->grid_vectors_ = grid_vectors;
}

// Get total number of points
unsigned long int CartesianGrid::npoint(void) {
    unsigned long int result = 1;
    for (int i = 0; i < this->grid_vectors_.size(); i++) {
        result *= this->grid_vectors_[i].size();
    }
    return result;
}

// Get shape of the grid
intvec CartesianGrid::grid_shape(void) {
    intvec result(this->ndim());
    for (int i = 0; i < this->ndim(); i++) {
        result[i] = this->grid_vectors_[i].size();
    }
    return result;
}

// Construct 2D table of points in a Cartesian Grid
Array CartesianGrid::grid_points(void) {
    // initialize table of grid points
    unsigned long int npoint_ = this->npoint();
    Array result({npoint_, this->ndim()});

    // assign value to each point
    intvec shape_ = this->grid_shape();
    for (int i = 0; i < npoint_; i++) {
        intvec index_ = contiguous_to_ndim_idx(i, shape_);
        floatvec value_(this->ndim());
        for (int j = 0; j < this->ndim(); j++) {
            value_[j] = this->grid_vectors_[j][index_[j]];
        }
        std::memcpy(&(result[{static_cast<unsigned long int>(i), 0}]), value_.data(), sizeof(float)*this->ndim());
    }

    return result;
}
#ifdef __COMMENT__
// Begin iterator
Grid::iterator CartesianGrid::begin(void) {
    this->begin_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_[0] = this->grid_vectors_[0].size();
    return Grid::iterator(this->begin_, this->dims_);
}

// End iterator
Grid::iterator CartesianGrid::end(void) {
    return Tensor::iterator(this->end_, this->dims_);
}

// Append/Remove/Get point
// -----------------------

Tensor CartesianGrid::operator[] (unsigned int index) {
    // convert C-contiguous index to ndim index
    std::vector<unsigned int> index_ = contiguous_to_ndim_idx({index}, this->dims_)[0];
    // get value
    Tensor value_(std::vector<unsigned int>({this->ndim()}));
    for (unsigned int j = 0; j < index_.size(); j++) {
        value_[std::vector<unsigned int>({j})] = this->grid_vectors_[j][index_[j]];
    }
    return value_;
}


Tensor CartesianGrid::operator[] (const std::vector<unsigned int> & index) {
    // check size of index
    if (index.size() != this->ndim()) {
        FAILURE("Size of index (%d) is different from ndim of CartesianGrid (%d).",
                index.size(), this->ndim());
    }
    // assign to result tensor
    Tensor result(std::vector<unsigned int>({this->ndim()}));
    for (unsigned int i = 0; i < index.size(); i++) {
        if (index[i] >= this->dims_[i]) {
            FAILURE("Size of dimension %d of index (%d) must be less than %d.",
                    i, index[i], this->dims_[i]);
        }
        result.data()[i] = this->grid_vectors_[i][index[i]];
    }
    return result;
}
#endif
}  // namespace merlin
