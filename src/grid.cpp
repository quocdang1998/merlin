// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <cstring>  // std::memcpy
#include <vector>
#include <numeric>
#include <utility>

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::array_copy

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// Grid
// -------------------------------------------------------------------------------------------------------------------------

// Construct an empty grid from a given number of n-dim points
Grid::Grid(unsigned long int npoint, unsigned long int ndim) : npoint_(npoint) {
    // calculate capacity
    unsigned long int capacity = 1;
    while (capacity < npoint) {
        capacity <<= 1;
    }
    // allocate data
    this->points_ = Array({capacity, ndim});
}

// Construct a grid and copy data
Grid::Grid(const Array & points) {
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
Array Grid::grid_points (void) const {
    const unsigned long int * shape_ptr = &(this->points_.shape()[0]);
    const unsigned long int * strides_ptr = &(this->points_.strides()[0]);
    return Array(this->points_.data(), this->points_.ndim(), shape_ptr, strides_ptr, false);
}

// Get reference Array to a point


// Begin iterator
Grid::iterator Grid::begin(void) {
    this->begin_ = intvec(2, 0);
    this->end_ = intvec(2, 0);
    this->end_[0] = this->npoint_;
    return Grid::iterator(this->begin_, this->points_.shape());
}

// End iterator
Grid::iterator Grid::end(void) {
    return Grid::iterator(this->end_, this->points_.shape());
}

// Append/Remove/Get point
// -----------------------
#ifdef __COMMENT__
Tensor Grid::operator[] (unsigned int index) {
    std::vector<unsigned int> index_grid = {index, 0};
    float * target = &(this->capacity_points_[index_grid]);
    unsigned int dims_[1] = {this->ndim()};
    unsigned int strides_[1] = {sizeof(float)};
    return Tensor(target, 1, dims_, strides_, false);
}


void Grid::push_back(std::vector<float> && point) {
    // check size of point
    if (point.size() != this->ndim()) {
        FAILURE("Cannot add point of dimension %d to a grid of dimension %d.", point.size(), this->ndim());
    }
    // add point to grid
    this->npoint_ += 1;
    if (this->npoint_ > this->capacity()) {
        // reallocate data
        Tensor new_location = Tensor(std::vector<unsigned int>({2*this->capacity(), this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->capacity_points_.data(), sizeof(float)*this->npoint_*this->ndim());
        this->capacity_points_ = std::move(new_location);
    }
    std::memcpy(&(this->capacity_points_[std::vector<unsigned int>({this->npoint_-1, 0})]), &(point[0]),
                sizeof(float)*this->ndim());
}


void Grid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity()/2) {
        // reallocate data
        Tensor new_location = Tensor(std::vector<unsigned int>({this->capacity()/2, this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->capacity_points_.data(), sizeof(float)*this->npoint_*this->ndim());
        this->capacity_points_ = std::move(new_location);
    } else {
        std::memset(&(this->capacity_points_[std::vector<unsigned int>({this->npoint_, 0})]), 0,
                    sizeof(float)*this->ndim());
    }
}

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

// Constructors
// ------------

CartesianGrid::CartesianGrid(const std::vector<std::vector<float>> & grid_vectors) {
    this->grid_vectors_ = grid_vectors;
    this->dims_ = std::vector<unsigned int>(grid_vectors.size());
    for (unsigned int i = 0; i < grid_vectors.size(); i++) {
        this->dims_[i] = grid_vectors[i].size();
    }
}

// Get members
// -----------

Tensor CartesianGrid::grid_points(void) {
    // create index vector
    unsigned int npoint_ = this->npoint();
    std::vector<unsigned int> indexes_(npoint_);
    std::iota(indexes_.begin(), indexes_.end(), 0);
    // get value
    Tensor grid_points(std::vector<unsigned int>({npoint_, this->ndim()}));
    std::vector<std::vector<unsigned int>> n_indexes = contiguous_to_ndim_idx(indexes_, this->dims_);
    std::vector<float> value_(this->ndim());
    for (unsigned int i = 0; i < npoint_; i++) {
        for (unsigned int j = 0; j < this->ndim(); j++) {
            value_[j] = this->grid_vectors_[j][n_indexes[i][j]];
        }
        // copy value to data
        std::memcpy(&(grid_points.data()[i*this->ndim()]), &(value_[0]), sizeof(float)*this->ndim());
    }

    return grid_points;
}

unsigned int CartesianGrid::npoint(void) {
    unsigned int npoint_ = 1;
    for (unsigned int i = 0; i < this->grid_vectors_.size(); i++) {
        npoint_ *= this->grid_vectors_[i].size();
    }
    return npoint_;
}


unsigned int CartesianGrid::ndim(void) {
    return this->grid_vectors_.size();
}


unsigned int CartesianGrid::capacity(void) {
    FAILURE("CartesianGrid doesn't have capacity.");
    return 0;
}

// Iterator
// --------

Grid::iterator CartesianGrid::begin(void) {
    // initialize index tensor
    this->begin_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_[0] = this->grid_vectors_[0].size();
    return Grid::iterator(this->begin_, this->dims_);
}


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
