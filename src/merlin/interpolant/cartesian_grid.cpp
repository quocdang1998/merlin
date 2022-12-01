// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

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
    // check grid vector
    this->grid_vectors_ = grid_vectors;
    for (int i = 0; i < this->ndim(); i++) {
        for (int j = 1; j < this->grid_vectors_[i].size(); j++) {
            if (this->grid_vectors_[i][j-1] >= this->grid_vectors_[i][j]) {
                FAILURE(std::invalid_argument, "Expected vector entries in increasing order, fail at index %d.\n", i);
            }
        }
    }
    intvec shape = this->grid_shape();
    intvec strides = array::contiguous_strides(shape, sizeof(float));
    this->points_ = new array::NdData(nullptr, this->ndim(), shape, strides);
}

// Construct 2D table of points in a Cartesian Grid
array::Array interpolant::CartesianGrid::grid_points(void) {
    // initialize table of grid points
    std::uint64_t npoint = this->size();
    array::Array result({npoint, this->ndim()});

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
    return interpolant::CartesianGrid::iterator(this->begin_, *(this->points_));
}

// End iterator
interpolant::CartesianGrid::iterator interpolant::CartesianGrid::end(void) {
    return interpolant::CartesianGrid::iterator(this->end_, *(this->points_));
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
std::uint64_t interpolant::CartesianGrid::malloc_size(void) {
    std::uint64_t size = sizeof(CartesianGrid) + this->ndim()*sizeof(floatvec);
    for (int i = 0; i < this->ndim(); i++) {
        size += this->grid_vectors_[i].size() * sizeof(float);
    }
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
}

#endif  // __MERLIN_CUDA__

// Destructor
__cuhostdev__ interpolant::CartesianGrid::~CartesianGrid(void) {}

}  // namespace merlin
