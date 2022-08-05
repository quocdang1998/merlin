// Copyright 2022 quocdang1998
#include "merlin/grid.hpp"

#include <cstring>
#include <vector>
#include <utility>

#include "merlin/logger.hpp"

namespace merlin {

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
                    sizeof(float)*this->capacity_points_.size());
        this->capacity_points_ = std::move(new_location);
    }
    std::memcpy(&(this->capacity_points_[std::vector<unsigned int>({this->npoint_-1,0})]),
                &(point[0]), sizeof(float)*this->ndim());
}

}  // namespace merlin
