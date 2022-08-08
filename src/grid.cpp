// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <cstring>
#include <vector>
#include <utility>

#include "merlin/logger.hpp"

namespace merlin {

// ------------------------------------------------------------------------------------------------
// Grid
// ------------------------------------------------------------------------------------------------

// Constructors
// ------------

Grid::Grid(unsigned int ndim, unsigned int npoint) : npoint_(npoint) {
    // calculate capacity
    unsigned int capacity_ = 1;
    while (capacity_ < npoint) {
        capacity_ <<= 1;
    }
    // allocate data
    this->capacity_points_ = Array(std::vector<unsigned int>({capacity_, ndim}));
}

// Get members
// -----------

Array Grid::grid_points(void) {
    return Array(this->capacity_points_.data(), this->capacity_points_.ndim(),
                 &(this->capacity_points_.dims()[0]), &(this->capacity_points_.strides()[0]),
                 false);
}

// Iterator
// --------

Grid::iterator Grid::begin(void) {
    this->begin_ = std::vector<unsigned int>(2, 0);
    this->end_ = std::vector<unsigned int>(2, 0);
    this->end_[0] = this->npoint_;
    return Grid::iterator(this->begin_, this->capacity_points_.dims());
}


Grid::iterator Grid::end(void) {
    return Array::iterator(this->end_, this->capacity_points_.dims());
}

// Append/Remove/Get point
// -----------------------

Array Grid::operator[] (unsigned int index) {
    std::vector<unsigned int> index_grid = {index, 0};
    float * target = &(this->capacity_points_[index_grid]);
    unsigned int dims_[1] = {this->ndim()};
    unsigned int strides_[1] = {sizeof(float)};
    return Array(target, 1, dims_, strides_, false);
}


void Grid::push_back(std::vector<float> && point) {
    // check size of point
    if (point.size() != this->ndim()) {
        FAILURE("Cannot add point of dimension %d to a grid of dimension %d.",
                point.size(), this->ndim());
    }
    // add point to grid
    this->npoint_ += 1;
    if (this->npoint_ > this->capacity()) {
        // reallocate data
        Array new_location = Array(std::vector<unsigned int>({2*this->capacity(), this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->capacity_points_.data(),
                    sizeof(float)*this->npoint_*this->ndim());
        this->capacity_points_ = std::move(new_location);
    }
    std::memcpy(&(this->capacity_points_[std::vector<unsigned int>({this->npoint_-1,0})]),
                &(point[0]), sizeof(float)*this->ndim());
}


void Grid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity()/2) {
        // reallocate data
        Array new_location = Array(std::vector<unsigned int>({this->capacity()/2, this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location.data(), this->capacity_points_.data(),
                    sizeof(float)*this->npoint_*this->ndim());
        this->capacity_points_ = std::move(new_location);
    } else {
        std::memset(&(this->capacity_points_[std::vector<unsigned int>({this->npoint_,0})]),
                    0, sizeof(float)*this->ndim());
    }
}

// ------------------------------------------------------------------------------------------------
// CartesianGrid
// ------------------------------------------------------------------------------------------------

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

Array CartesianGrid::grid_points(void) {
    // initialize metadata
    unsigned int npoint_ = this->npoint();
    unsigned int ndim_ = this->ndim();
    std::vector<unsigned int> prod_(ndim_);
    for (int i = ndim_-1; i >= 0; i--) {
        if (i == ndim_-1) {
            prod_[i] = 1;
        } else {
            prod_[i] = prod_[i+1] * this->dims_[i+1];
        }
    }
    // create index vector and get value
    std::vector<unsigned int> index_(ndim_);
    std::vector<float> value_(ndim_);
    Array grid_points(std::vector<unsigned int>({npoint_, ndim_}));
    for(unsigned int i = 0; i < npoint_; i++) {
        for(unsigned int j = 0; j < ndim_; j++) {
            index_[j] = (i / prod_[j]) % this->dims_[j];
            value_[j] = this->grid_vectors_[j][index_[j]];
        }
        // copy value to data
        std::memcpy(&(grid_points.data()[i*ndim_]), &(value_[0]), sizeof(float)*ndim_);
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
    // initialize index array
    this->begin_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_ = std::vector<unsigned int>(this->ndim(), 0);
    this->end_[0] = this->grid_vectors_[0].size();
    return Grid::iterator(this->begin_, this->dims_);
}


Grid::iterator CartesianGrid::end(void) {
    return Array::iterator(this->end_, this->dims_);
}

// Append/Remove/Get point
// -----------------------

Array CartesianGrid::operator[] (unsigned int index) {
    // initialize vector of value
    unsigned int npoint_ = this->npoint();
    unsigned int ndim_ = this->ndim();
    std::vector<unsigned int> dims_(ndim_);
    std::vector<unsigned int> prod_(ndim_);
    for (int i = ndim_-1; i >= 0; i--) {
        dims_[i] = this->grid_vectors_[i].size();
        if (i == ndim_-1) {
            prod_[i] = 1;
        } else {
            prod_[i] = prod_[i+1]*dims_[i+1];
        }
    }
    // get value
    std::vector<unsigned int> index_(ndim_);
    Array value_(std::vector<unsigned int>({ndim_}));
    for(unsigned int j = 0; j < ndim_; j++) {
        index_[j] = (index / prod_[j]) % dims_[j];
        value_[std::vector<unsigned int>({j})] = this->grid_vectors_[j][index_[j]];
    }

    return value_;
}


Array CartesianGrid::operator[] (const std::vector<unsigned int> & index) {
    // check size of index
    if (index.size() != this->ndim()) {
        FAILURE("Size of index (%d) is different from ndim of CartesianGrid (%d).",
                index.size(), this->ndim());
    }
    // assign to result array
    Array result(std::vector<unsigned int>({this->ndim()}));
    for (unsigned int i = 0; i < index.size(); i++) {
        if (index[i] >= this->dims_[i]) {
            FAILURE("Size of dimension %d of index (%d) must be less than %d.",
                    i, index[i], this->dims_[i]);
        }
        result.data()[i] = this->grid_vectors_[i][index[i]];
    }
    return result;
}

}  // namespace merlin
