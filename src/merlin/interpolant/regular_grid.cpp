// Copyright 2022 quocdang1998
#include "merlin/interpolant/regular_grid.hpp"

#include <cstring>  // std::memcpy
#include <cinttypes>
#include <numeric>  // std::iota
#include <utility>  // std::move

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array:Slice
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// RegularGrid
// --------------------------------------------------------------------------------------------------------------------

// Construct an empty grid from a given number of n-dim points
interpolant::RegularGrid::RegularGrid(std::uint64_t npoint, std::uint64_t ndim) : npoint_(npoint) {
    // calculate capacity
    std::uint64_t capacity = 1;
    while (capacity < npoint) {
        capacity <<= 1;
    }
    // allocate data
    this->points_ = new array::Array(intvec({capacity, ndim}));
}

// Construct a grid and copy data
interpolant::RegularGrid::RegularGrid(const array::Array & points) {
    // check if original data is 2D
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected an Array of dimension 2.\n");
    }
    // calculate capacity
    this->npoint_ = points.shape()[0];
    std::uint64_t capacity = 1;
    while (capacity < this->npoint_) {
        capacity <<= 1;
    }
    // copy data from old array to new array
    array::Array * points_array_ptr;
    this->points_ = points_array_ptr = new array::Array(intvec({capacity, points.ndim()}));
    array::Array temporary(*points_array_ptr, {array::Slice(0, this->npoint_), array::Slice()});
    array_copy(this->points_, &temporary, std::memcpy);
}

// Copy constructor
interpolant::RegularGrid::RegularGrid(const interpolant::RegularGrid & src) : npoint_(src.npoint_) {
    // copy data from old array to new array
    this->points_ = new array::Array(intvec({src.capacity(), src.ndim()}));
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(double)*src.capacity()*src.ndim());
}

// Copy assignment
interpolant::RegularGrid & interpolant::RegularGrid::operator=(const interpolant::RegularGrid & src) {
    this->npoint_ = src.npoint_;
    // copy data from old array to new array
    this->points_ = new array::Array(intvec({src.capacity(), src.ndim()}));
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(double)*src.capacity()*src.ndim());
    return *this;
}

// Move constructor
interpolant::RegularGrid::RegularGrid(interpolant::RegularGrid && src) : npoint_(src.npoint_) {
    this->points_ = src.points_;
    src.points_ = nullptr;
}

// Move assignment
interpolant::RegularGrid & interpolant::RegularGrid::operator=(interpolant::RegularGrid && src) {
    this->npoint_ = src.npoint_;
    this->points_ = src.points_;
    src.points_ = nullptr;
    return *this;
}

// Get reference Array to a point
Vector<double> interpolant::RegularGrid::operator[](std::uint64_t index) {
    array::Array & points = *(dynamic_cast<array::Array *>(this->points_));
    Vector<double> result;
    result.assign(&(points[{index, 0}]), this->ndim());
    return result;
}

// Begin iterator
interpolant::RegularGrid::iterator interpolant::RegularGrid::begin(void) {
    intvec index(2, 0);
    this->begin_ = interpolant::RegularGrid::iterator(index, this->points_->shape());
    index[0] = this->npoint_;
    this->end_ = interpolant::RegularGrid::iterator(index, this->points_->shape());
    return this->begin_;
}

// End iterator
interpolant::RegularGrid::iterator interpolant::RegularGrid::end(void) {
    return this->end_;
}

// Append a point at the end of the grid
void interpolant::RegularGrid::push_back(Vector<double> && point) {
    // check size of point
    if (point.size() != this->ndim()) {
        FAILURE(std::invalid_argument, "Cannot add point of dimension %" PRIu64 " to a grid of dimension %" PRIu64 ".",
                point.size(), this->ndim());
    }
    // add point to grid
    this->npoint_ += 1;
    if (this->npoint_ > this->capacity()) {
        // reallocate data
        array::Array * new_location = new array::Array(intvec({2*this->capacity(), this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(), sizeof(double)*this->capacity()*this->ndim());
        delete this->points_;
        this->points_ = new_location;
    }
    array::Array & points = *(static_cast<array::Array *>(this->points_));
    double * lastitem_ptr = &(points[{this->npoint_-1, 0}]);
    std::memcpy(lastitem_ptr, point.data(), sizeof(double)*this->ndim());
}

// Remove a point at the end of the grid
void interpolant::RegularGrid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity()/2) {
        // reallocate data
        array::Array * new_location = new array::Array(intvec({this->capacity()/2, this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(), sizeof(double)*(this->capacity()/2)*this->ndim());
        delete this->points_;
        this->points_ = new_location;
    } else {
        array::Array & points = *(static_cast<array::Array *>(this->points_));
        double * lastitem_ptr = &(points[{this->npoint_, 0}]);
        std::memset(lastitem_ptr, 0, sizeof(double)*this->ndim());
    }
}

// Destructor
interpolant::RegularGrid::~RegularGrid(void) {}

}  // namespace merlin
