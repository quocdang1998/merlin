// Copyright 2022 quocdang1998
#include "merlin/interpolant/grid.hpp"

#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <utility>  // std::move

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/array/utils.hpp"  // merlin::contiguous_strides, merlin::array_copy, merlin::contiguous_to_ndim_idx

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// RegularGrid
// -------------------------------------------------------------------------------------------------------------------------

// Construct an empty grid from a given number of n-dim points
RegularGrid::RegularGrid(std::uint64_t npoint, std::uint64_t ndim) : npoint_(npoint) {
    // calculate capacity
    std::uint64_t capacity = 1;
    while (capacity < npoint) {
        capacity <<= 1;
    }
    // allocate data
    this->points_ = new Array({capacity, ndim});
}

// Construct a grid and copy data
RegularGrid::RegularGrid(const Array & points) {
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
    this->points_ = new Array({capacity, points.ndim()});
    array_copy(this->points_, &points, std::memcpy);
}

// Copy constructor
RegularGrid::RegularGrid(const RegularGrid & src) : npoint_(src.npoint_) {
    // copy data from old array to new array
    this->points_ = new Array({src.capacity(), src.ndim()});
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(float)*src.capacity()*src.ndim());
}

// Copy assignment
RegularGrid & RegularGrid::operator=(const RegularGrid & src) {
    this->npoint_ = src.npoint_;
    // copy data from old array to new array
    this->points_ = new Array({src.capacity(), src.ndim()});
    std::memcpy(this->points_->data(), src.points_->data(), sizeof(float)*src.capacity()*src.ndim());
    return *this;
}

// Move constructor
RegularGrid::RegularGrid(RegularGrid && src) : npoint_(src.npoint_) {
    this->points_ = src.points_;
    src.points_ = NULL;
}

// Move assignment
RegularGrid & RegularGrid::operator=(RegularGrid && src) {
    this->npoint_ = src.npoint_;
    this->points_ = src.points_;
    src.points_ = NULL;
    return *this;
}

// Get reference Array to a point
Array RegularGrid::operator[](unsigned int index) {
    Array & points = *(dynamic_cast<Array *>(this->points_));
    float * target_ptr = &(points[{index, 0}]);
    std::uint64_t shape = this->points_->ndim();
    std::uint64_t strides = sizeof(float);
    return Array(target_ptr, 1, &shape, &strides, false);
}

// Begin iterator
RegularGrid::iterator RegularGrid::begin(void) {
    this->begin_ = intvec(2, 0);
    this->end_ = intvec(2, 0);
    this->end_[0] = this->npoint_;
    return RegularGrid::iterator(this->begin_, *(this->points_));
}

// End iterator
RegularGrid::iterator RegularGrid::end(void) {
    return RegularGrid::iterator(this->end_, *(this->points_));
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
        Array * new_location = new Array({2*this->capacity(), this->ndim()});
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(), sizeof(float)*this->capacity()*this->ndim());
        delete this->points_;
        this->points_ = new_location;
    }
    Array & points = *(dynamic_cast<Array *>(this->points_));
    float * lastitem_ptr = &(points[{this->npoint_-1, 0}]);
    std::memcpy(lastitem_ptr, point.data(), sizeof(float)*this->ndim());
}

// Remove a point at the end of the grid
void RegularGrid::pop_back(void) {
    this->npoint_ -= 1;
    if (this->npoint_ <= this->capacity()/2) {
        // reallocate data
        Array * new_location = new Array({this->capacity()/2, this->ndim()});
        // copy data from old location to new location
        std::memcpy(new_location->data(), this->points_->data(), sizeof(float)*(this->capacity()/2)*this->ndim());
        delete this->points_;
        this->points_ = new_location;
    } else {
        Array & points = *(dynamic_cast<Array *>(this->points_));
        float * lastitem_ptr = &(points[{this->npoint_, 0}]);
        std::memset(lastitem_ptr, 0, sizeof(float)*this->ndim());
    }
}

// Destructor
RegularGrid::~RegularGrid(void) {
    if (this->points_ != NULL) {
        delete this->points_;
    }
}


// -------------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// -------------------------------------------------------------------------------------------------------------------------

// Construct from a list of vector of values
CartesianGrid::CartesianGrid(std::initializer_list<floatvec> grid_vectors) : grid_vectors_(grid_vectors) {
    // check grid vector
    this->grid_vectors_ = grid_vectors;
    for (int i = 0; i < this->ndim(); i++) {
        for (int j = 1; j < this->grid_vectors_[i].size(); j++) {
            if (this->grid_vectors_[i][j-1] >= this->grid_vectors_[i][j]) {
                FAILURE(std::invalid_argument, "Expected vector entries in increasing order, got vector at index %d.\n", i);
            }
        }
    }
    intvec shape = this->grid_shape();
    intvec strides = contiguous_strides(shape, sizeof(float));
    this->points_ = new NdData(NULL, this->ndim(), shape, strides);
}

// Get total number of points
std::uint64_t CartesianGrid::size(void) {
    std::uint64_t result = 1;
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
    std::uint64_t npoint = this->size();
    Array result({npoint, this->ndim()});

    // assign value to each point
    intvec shape_ = this->grid_shape();
    for (int i = 0; i < npoint; i++) {
        intvec index_ = contiguous_to_ndim_idx(i, shape_);
        floatvec value_(this->ndim());
        for (int j = 0; j < this->ndim(); j++) {
            value_[j] = this->grid_vectors_[j][index_[j]];
        }
        std::memcpy(&(result[{static_cast<std::uint64_t>(i), 0}]), value_.data(), sizeof(float)*this->ndim());
    }

    return result;
}

// Begin iterator
CartesianGrid::iterator CartesianGrid::begin(void) {
    this->begin_ = intvec(this->ndim(), 0);
    this->end_ = intvec(this->ndim(), 0);
    this->end_[0] = this->grid_vectors_[0].size();
    return CartesianGrid::iterator(this->begin_, *(this->points_));
}

// End iterator
CartesianGrid::iterator CartesianGrid::end(void) {
    return CartesianGrid::iterator(this->end_, *(this->points_));
}

// Get element at a C-contiguous index
floatvec CartesianGrid::operator[](std::uint64_t index) {
    intvec nd_index = contiguous_to_ndim_idx(index, this->grid_shape());
    floatvec result(this->ndim(), 0);
    for (int i = 0; i < result.size(); i++) {
        result[i] = this->grid_vectors_[i][nd_index[i]];
    }
    return result;
}

// Get element at a multi-dimensional index
floatvec CartesianGrid::operator[](const intvec & index) {
    floatvec result(this->ndim(), 0);
    for (int i = 0; i < result.size(); i++) {
        result[i] = this->grid_vectors_[i][index[i]];
    }
    return result;
}

// Calculate minimum size to allocate to store the object
std::uint64_t CartesianGrid::malloc_size(void) {
    std::uint64_t size = sizeof(CartesianGrid) + this->ndim()*sizeof(floatvec);
    for (int i = 0; i < this->ndim(); i++) {
        size += this->grid_vectors_[i].size() * sizeof(float);
    }
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void copy_to_gpu(CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
}

#endif  // __MERLIN_CUDA__

// Destructor
CartesianGrid::~CartesianGrid(void) {
    if (this->points_ != NULL) {
        delete this->points_;
        this->points_ = NULL;
    }
}

}  // namespace merlin
