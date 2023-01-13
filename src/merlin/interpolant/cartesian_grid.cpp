// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <algorithm>  // std::is_sorted
#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <utility>  // std::move

#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

// Construct from a list of vector of values
interpolant::CartesianGrid::CartesianGrid(std::initializer_list<floatvec> grid_vectors) : grid_vectors_(grid_vectors) {
    for (int i = 0; i < this->ndim(); i++) {
        if (!std::is_sorted(this->grid_vectors_[i].begin(), this->grid_vectors_[i].end())) {
            FAILURE(std::invalid_argument, "Expected vector entries in increasing order, fail at index %d.\n", i);
        }
    }
    this->calc_grid_shape();
}

interpolant::CartesianGrid::CartesianGrid(const Vector<floatvec> & grid_vectors) : grid_vectors_(grid_vectors) {
    for (int i = 0; i < this->ndim(); i++) {
        if (!std::is_sorted(this->grid_vectors_[i].begin(), this->grid_vectors_[i].end())) {
            FAILURE(std::invalid_argument, "Expected vector entries in increasing order, fail at index %d.\n", i);
        }
    }
    this->calc_grid_shape();
}

// Construct 2D table of points in a Cartesian Grid
array::Array interpolant::CartesianGrid::grid_points(void) {
    // initialize table of grid points
    std::uint64_t npoint = this->size();
    array::Array result(intvec({npoint, this->ndim()}));
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
interpolant::CartesianGrid::iterator interpolant::CartesianGrid::begin(void) {
    this->begin_ = intvec(this->ndim(), 0);
    this->end_ = intvec(this->ndim(), 0);
    this->end_[0] = this->grid_vectors_[0].size();
    return interpolant::CartesianGrid::iterator(this->begin_, this->grid_shape_);
}

// End iterator
interpolant::CartesianGrid::iterator interpolant::CartesianGrid::end(void) {
    return interpolant::CartesianGrid::iterator(this->end_, this->grid_shape_);
}

// Get element at a C-contiguous index
floatvec interpolant::CartesianGrid::operator[](std::uint64_t index) {
    intvec nd_index = contiguous_to_ndim_idx(index, this->grid_shape());
    floatvec result(this->ndim(), 0);
    for (int i = 0; i < result.size(); i++) {
        result[i] = this->grid_vectors_[i][nd_index[i]];
    }
    return result;
}

// Get element at a multi-dimensional index
floatvec interpolant::CartesianGrid::operator[](const intvec & index) {
    floatvec result(this->ndim(), 0);
    for (int i = 0; i < result.size(); i++) {
        result[i] = this->grid_vectors_[i][index[i]];
    }
    return result;
}

// Calculate minimum size to allocate to store the object
std::uint64_t interpolant::CartesianGrid::malloc_size(void) const {
    std::uint64_t size = sizeof(CartesianGrid) + this->ndim()*sizeof(floatvec);
    for (int i = 0; i < this->ndim(); i++) {
        size += this->grid_vectors_[i].size() * sizeof(float);
    }
    size += this->ndim()*sizeof(std::uint64_t);
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
}

#endif  // __MERLIN_CUDA__

// Calculate grid shape
void interpolant::CartesianGrid::calc_grid_shape(void) {
    this->grid_shape_ = intvec(this->ndim());
    for (int i = 0; i < this->grid_shape_.size(); i++) {
        this->grid_shape_[i] = this->grid_vectors_[i].size();
    }
}

// Destructor
__cuhostdev__ interpolant::CartesianGrid::~CartesianGrid(void) {}

}  // namespace merlin
