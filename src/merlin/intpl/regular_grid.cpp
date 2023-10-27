// Copyright 2022 quocdang1998
#include "merlin/intpl/regular_grid.hpp"

#include <cinttypes>  // PRIu64
#include <cstring>    // std::memcpy
#include <numeric>    // std::iota
#include <utility>    // std::move

#include "merlin/array/operation.hpp"  // merlin::array::copy
#include "merlin/logger.hpp"           // FAILURE
#include "merlin/slice.hpp"            // merlin::Slice

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RegularGrid
// ---------------------------------------------------------------------------------------------------------------------

// Construct an empty grid from a given number of n-dim points
intpl::RegularGrid::RegularGrid(std::uint64_t npoint, std::uint64_t ndim) : npoint_(npoint) {
    // calculate capacity
    std::uint64_t capacity = 1;
    while (capacity < npoint) {
        capacity <<= 1;
    }
    // allocate data
    this->points_ = new array::Array(intvec({capacity, ndim}));
}

// Construct a grid and copy data
intpl::RegularGrid::RegularGrid(const array::Array & points) {
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
    array::Array temporary(*points_array_ptr, {Slice(0, this->npoint_), Slice()});
    copy(this->points_, &temporary, std::memcpy);
}

// Copy constructor
intpl::RegularGrid::RegularGrid(const intpl::RegularGrid & src) : npoint_(src.npoint_) {
    // copy data from old array to new array
    this->points_ = new array::Array(intvec({src.capacity(), src.ndim()}));
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(double) * src.capacity() * src.ndim());
}

// Copy assignment
intpl::RegularGrid & intpl::RegularGrid::operator=(const intpl::RegularGrid & src) {
    this->npoint_ = src.npoint_;
    // copy data from old array to new array
    this->points_ = new array::Array(intvec({src.capacity(), src.ndim()}));
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(double) * src.capacity() * src.ndim());
    return *this;
}

// Move constructor
intpl::RegularGrid::RegularGrid(intpl::RegularGrid && src) : npoint_(src.npoint_) {
    this->points_ = src.points_;
    src.points_ = nullptr;
}

// Move assignment
intpl::RegularGrid & intpl::RegularGrid::operator=(intpl::RegularGrid && src) {
    this->npoint_ = src.npoint_;
    this->points_ = src.points_;
    src.points_ = nullptr;
    return *this;
}

// Get reference Array to a point
Vector<double> intpl::RegularGrid::operator[](std::uint64_t index) {
    array::Array & points = *(dynamic_cast<array::Array *>(this->points_));
    Vector<double> result;
    result.assign(&(points[{index, 0}]), this->ndim());
    return result;
}

// Begin iterator
intpl::RegularGrid::iterator intpl::RegularGrid::begin(void) {
    intvec index(2, 0);
    this->begin_ = intpl::RegularGrid::iterator(index, this->points_->shape());
    index[0] = this->npoint_;
    this->end_ = intpl::RegularGrid::iterator(index, this->points_->shape());
    return this->begin_;
}

// End iterator
intpl::RegularGrid::iterator intpl::RegularGrid::end(void) { return this->end_; }

// Append a point at the end of the grid
void intpl::RegularGrid::push_back(Vector<double> && point) {
    // check size of point
    if (point.size() != this->ndim()) {
        FAILURE(std::invalid_argument, "Cannot add point of dimension %" PRIu64 " to a grid of dimension %" PRIu64 ".",
                point.size(), this->ndim());
    }
    // add point to grid
    this->npoint_ += 1;
    if (this->npoint_ > this->capacity()) {
        // reallocate data
        array::Array * new_location = new array::Array(intvec({2 * this->capacity(), this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(), sizeof(double) * this->capacity() * this->ndim());
        delete this->points_;
        this->points_ = new_location;
    }
    array::Array & points = *(static_cast<array::Array *>(this->points_));
    double * lastitem_ptr = &(points[{this->npoint_ - 1, 0}]);
    std::memcpy(lastitem_ptr, point.data(), sizeof(double) * this->ndim());
}

// Remove a point at the end of the grid
void intpl::RegularGrid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity() / 2) {
        // reallocate data
        array::Array * new_location = new array::Array(intvec({this->capacity() / 2, this->ndim()}));
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(),
                    sizeof(double) * (this->capacity() / 2) * this->ndim());
        delete this->points_;
        this->points_ = new_location;
    } else {
        array::Array & points = *(static_cast<array::Array *>(this->points_));
        double * lastitem_ptr = &(points[{this->npoint_, 0}]);
        std::memset(lastitem_ptr, 0, sizeof(double) * this->ndim());
    }
}

// Destructor
intpl::RegularGrid::~RegularGrid(void) {}

}  // namespace merlin
