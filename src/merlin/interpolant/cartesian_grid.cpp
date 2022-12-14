// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <algorithm>  // std::is_sorted, std::inplace_merge
#include <cinttypes>  // PRIu64
#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream
#include <utility>  // std::move

#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

// Construct from a list of vector of values
interpolant::CartesianGrid::CartesianGrid(std::initializer_list<Vector<double>> grid_vectors) :
grid_vectors_(grid_vectors) {
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        if (!std::is_sorted(this->grid_vectors_[i].begin(), this->grid_vectors_[i].end())) {
            FAILURE(std::invalid_argument, "Expected vector entries in increasing order, fail at index %" PRIu64 ".\n",
                    i);
        }
    }
}

// Construct from a vector of values
interpolant::CartesianGrid::CartesianGrid(const Vector<Vector<double>> & grid_vectors) : grid_vectors_(grid_vectors) {
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        if (!std::is_sorted(this->grid_vectors_[i].begin(), this->grid_vectors_[i].end())) {
            FAILURE(std::invalid_argument, "Expected vector entries in increasing order, fail at index %" PRIu64 ".\n",
                    i);
        }
    }
}

// Construct 2D table of points in a Cartesian Grid
array::Array interpolant::CartesianGrid::grid_points(void) const {
    // initialize table of grid points
    std::uint64_t npoint = this->size();
    array::Array result(intvec({npoint, this->ndim()}));
    // assign value to each point
    intvec shape_ = this->get_grid_shape();
    for (std::uint64_t i_point = 0; i_point < npoint; i_point++) {
        intvec index_ = contiguous_to_ndim_idx(i_point, shape_);
        Vector<double> value_(this->ndim());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            value_[i_dim] = this->grid_vectors_[i_dim][index_[i_dim]];
        }
        std::memcpy(&(result[{static_cast<std::uint64_t>(i_point), 0}]), value_.data(), sizeof(double)*this->ndim());
    }
    return result;
}

// Begin iterator
interpolant::CartesianGrid::iterator interpolant::CartesianGrid::begin(void) {
    intvec index(this->ndim(), 0);
    intvec shape = this->get_grid_shape();
    this->begin_ = interpolant::CartesianGrid::iterator(index, shape);
    index[0] = this->grid_vectors_[0].size();
    this->end_ = interpolant::CartesianGrid::iterator(index, shape);
    return this->begin_;
}

// End iterator
interpolant::CartesianGrid::iterator interpolant::CartesianGrid::end(void) {
    return this->end_;
}

// Calculate minimum size to allocate to store the object
std::uint64_t interpolant::CartesianGrid::malloc_size(void) const {
    std::uint64_t size = sizeof(CartesianGrid) + this->ndim()*sizeof(Vector<double>);
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        size += this->grid_vectors_[i].size() * sizeof(double);
    }
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * interpolant::CartesianGrid::copy_to_gpu(interpolant::CartesianGrid * gpu_ptr,
                                               void * grid_vector_data_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

interpolant::CartesianGrid & interpolant::CartesianGrid::operator+=(const interpolant::CartesianGrid & grid) {
    // check n-dim of 2 grids
    if (this->ndim() != grid.ndim()) {
        FAILURE(std::invalid_argument, "Cannot unify 2 grids with different n-dim.\n");
    }
    // unify 2 grid on each dimension
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        std::uint64_t dim_1 = this->grid_vectors_[i_dim].size(), dim_2 = grid.grid_vectors_[i_dim].size();
        // sort 2 sorted array
        Vector<double> buffer(dim_1 + dim_2);
        std::memcpy(buffer.data(), this->grid_vectors_[i_dim].data(), dim_1*sizeof(double));
        std::memcpy(buffer.data() + dim_1, grid.grid_vectors_[i_dim].data(), dim_2*sizeof(double));
        std::inplace_merge(buffer.data(), buffer.data() + dim_1, buffer.data() + dim_1 + dim_2);
        // squeeze the buffer
        std::uint64_t squeezed_size = buffer.size();
        for (std::uint64_t i_node = 1; i_node < buffer.size(); i_node++) {
            if (buffer[i_node] == buffer[i_node-1]) {
                --squeezed_size;
            }
        }
        // record to this instance
        this->grid_vectors_[i_dim] = Vector<double>(squeezed_size);
        this->grid_vectors_[i_dim][0] = buffer[0];
        for (std::uint64_t i_buffer = 1, i_squeezed = 1; i_buffer < buffer.size(); i_buffer++) {
            if (buffer[i_buffer] == buffer[i_buffer-1]) {
                continue;
            }
            this->grid_vectors_[i_dim][i_squeezed] = buffer[i_buffer];
            i_squeezed++;
        }
    }
    return *this;
}

// String representation
std::string interpolant::CartesianGrid::str(void) const {
    std::ostringstream os;
    os << "Cartesian grid:\n";
    os << "  Grid vectors:\n";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (i_dim != 0) {
            os << "\n";
        }
        os << "    Dimension " << i_dim << ": " << this->grid_vectors_[i_dim].str();
    }
    return os.str();
}

// Destructor
__cuhostdev__ interpolant::CartesianGrid::~CartesianGrid(void) {}

// Union of 2 Cartesian grid
interpolant::CartesianGrid interpolant::operator+(const interpolant::CartesianGrid & grid_1,
                                     const interpolant::CartesianGrid & grid_2) {
    interpolant::CartesianGrid result(grid_1);
    return result += grid_2;
}

}  // namespace merlin
