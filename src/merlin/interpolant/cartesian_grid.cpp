// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <algorithm>  // std::is_sorted, std::inplace_merge
#include <cinttypes>  // PRIu64
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

// Construct an empty Cartesian Grid of ndim dimension
interpolant::CartesianGrid::CartesianGrid(std::uint64_t ndim) : grid_vectors_(ndim) {
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
void interpolant::CartesianGrid::copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) const {
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

namespace interpolant {

// Union of 2 Cartesian grid
interpolant::CartesianGrid operator+(const interpolant::CartesianGrid & grid_1,
                                     const interpolant::CartesianGrid & grid_2) {
    // check n-dim of 2 grids
    if (grid_1.ndim() != grid_2.ndim()) {
        FAILURE(std::invalid_argument, "Cannot unify 2 grids with different n-dim.\n");
    }
    std::uint64_t ndim = grid_1.ndim();
    // unify 2 grid
    interpolant::CartesianGrid result;
    result.grid_vectors_ = Vector<floatvec>(ndim);
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        std::uint64_t dim_1 = grid_1.grid_vectors_[i_dim].size(), dim_2 = grid_2.grid_vectors_[i_dim].size();
        // sort 2 sorted array
        floatvec buffer(dim_1 + dim_2);
        std::memcpy(buffer.data(), grid_1.grid_vectors_[i_dim].data(), dim_1*sizeof(float));
        std::memcpy(buffer.data() + dim_1, grid_2.grid_vectors_[i_dim].data(), dim_2*sizeof(float));
        std::inplace_merge(buffer.data(), buffer.data() + dim_1, buffer.data() + dim_1 + dim_2);
        // squeeze the buffer
        std::uint64_t dim_size = buffer.size();
        for (std::uint64_t i_node = 1; i_node < buffer.size(); i_node++) {
            if (buffer[i_node] == buffer[i_node-1]) {
                --dim_size;
            }
        }
        result.grid_vectors_[i_dim] = floatvec(dim_size);
        result.grid_vectors_[i_dim][0] = buffer[0];
        for (std::uint64_t i_buffer = 1, i_squeezed = 1; i_buffer < buffer.size(); i_buffer++) {
            if (buffer[i_buffer] == buffer[i_buffer-1]) {
                continue;
            }
            result.grid_vectors_[i_dim][i_squeezed] = buffer[i_buffer];
            i_squeezed++;
        }
    }
    result.calc_grid_shape();
    return result;
}

}  // namespace interpolant

}  // namespace merlin
