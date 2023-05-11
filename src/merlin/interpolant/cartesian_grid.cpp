// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <algorithm>  // std::find
#include <cinttypes>  // PRIu64
#include <cstring>  // std::memcpy
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream
#include <utility>  // std::move
#include <vector>  // std::vector

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Check utils
// --------------------------------------------------------------------------------------------------------------------

// Get sorted index
static std::vector<std::uint64_t> sorted_index(const Vector<double> & input) {
    std::vector<std::uint64_t> result(input.size());
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(result.begin(), result.end(),
                     [&input] (std::uint64_t i1, std::uint64_t i2) {return input[i1] < input[i2];});
    return result;
}

// Check for duplicated element in a vector
static bool has_duplicated_element(const Vector<double> & input) {
    std::vector<std::uint64_t> sorted_idx = sorted_index(input);
    for (std::uint64_t i = 1; i < sorted_idx.size(); i++) {
        if (input[sorted_idx[i]] == input[sorted_idx[i-1]]) {
            return true;
        }
    }
    return false;
}

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

// Construct from a vector of values
interpolant::CartesianGrid::CartesianGrid(const Vector<Vector<double>> & grid_vectors) : grid_vectors_(grid_vectors) {
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        if (has_duplicated_element(this->grid_vectors_[i])) {
            FAILURE(std::invalid_argument, "Found duplicated element at dimension %" PRIu64 ".\n", i);
        }
    }
}

// Constructor from an r-value reference to a vector of values
interpolant::CartesianGrid::CartesianGrid(Vector<Vector<double>> && grid_vectors) : grid_vectors_(grid_vectors) {
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        if (has_duplicated_element(this->grid_vectors_[i])) {
            FAILURE(std::invalid_argument, "Found duplicated element at dimension %" PRIu64 ".\n", i);
        }
    }
}

// Constructor as subgrid of another grid
interpolant::CartesianGrid::CartesianGrid(const interpolant::CartesianGrid & whole,
                                          const Vector<array::Slice> & slices) {
    // check size
    if (slices.size() != whole.ndim()) {
        FAILURE(std::invalid_argument, "Dimension of Slices and CartesianGrid not compatible (expected %u, got %u).\n",
                static_cast<unsigned int>(whole.ndim()), static_cast<unsigned int>(slices.size()));
    }
    // create result
    this->grid_vectors_ = Vector<Vector<double>>(whole.ndim());
    intvec original_shape = whole.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim(); i_dim++) {
        auto [_, new_shape, __] = slices[i_dim].slice_on(original_shape[i_dim], {sizeof(double)});
        this->grid_vectors_[i_dim] = Vector<double>(new_shape);
        for (std::uint64_t i_shape = 0; i_shape < new_shape; i_shape++) {
            std::uint64_t in_wrt_original = slices[i_dim].start() + i_shape * slices[i_dim].step();
            this->grid_vectors_[i_dim][i_shape] = whole.grid_vectors()[i_dim][in_wrt_original];
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
        Vector<double> value_;
        value_.assign(&(result[{i_point, 0}]), this->ndim());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            value_[i_dim] = this->grid_vectors_[i_dim][index_[i_dim]];
        }
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
    std::uint64_t size = sizeof(interpolant::CartesianGrid) + this->ndim()*sizeof(Vector<double>);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        size += this->grid_vectors_[i_dim].size() * sizeof(double);
    }
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * interpolant::CartesianGrid::copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                               std::uintptr_t stream_ptr) const {
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
        // push_back if not duplicated
        const Vector<double> & this_grid_vector = this->grid_vectors_[i_dim];
        const Vector<double> & other_grid_vector = grid.grid_vectors_[i_dim];
        std::vector<double> buffer(this_grid_vector.begin(), this_grid_vector.end());
        for (const double & node : other_grid_vector) {
            if (std::find(this_grid_vector.begin(), this_grid_vector.end(), node) == this_grid_vector.end()) {
                buffer.push_back(node);
            }
        }
        // record to this instance
        this->grid_vectors_[i_dim] = Vector<double>(buffer.data(), buffer.size());
    }
    return *this;
}

// Exclusion on each dimension of 2 Cartesian grids
double interpolant::exclusion_grid(const interpolant::CartesianGrid & grid_parent,
                                   const interpolant::CartesianGrid & grid_child, const Vector<double> & x) {
    // check n-dim of 2 grids
    if (grid_parent.ndim() != grid_child.ndim()) {
        FAILURE(std::invalid_argument, "Cannot get exclusion of 2 grids with different n-dim.\n");
    }
    // get exclusion of 2 grid on each dimension
    double result = 1.0;
    for (std::uint64_t i_dim = 0; i_dim < grid_parent.ndim(); i_dim++) {
        const Vector<double> & parent_vector = grid_parent.grid_vectors_[i_dim];
        const Vector<double> & child_vector = grid_child.grid_vectors_[i_dim];
        for (const double & node : parent_vector) {
            if (std::find(child_vector.cbegin(), child_vector.cend(), node) == child_vector.cend()) {
                result *= (x[i_dim] - node);
            }
        }
    }
    return result;
}

// Check if point in the grid
bool interpolant::CartesianGrid::contains(const Vector<double> & point) const {
    if (point.size() != this->ndim()) {
        return false;
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        const Vector<double> & grid_vector = this->grid_vectors_[i_dim];
        if (std::find(grid_vector.cbegin(), grid_vector.cend(), point[i_dim]) == grid_vector.cend()) {
            return false;
        }
    }
    return true;
}

// String representation
std::string interpolant::CartesianGrid::str(void) const {
    std::ostringstream os;
    os << "<CartesianGrid(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        os << "(" << this->grid_vectors_[i_dim].str() << ")";
    }
    os << ")>";
    return os.str();
}

// Destructor
interpolant::CartesianGrid::~CartesianGrid(void) {}

// Union of 2 Cartesian grid
interpolant::CartesianGrid interpolant::operator+(const interpolant::CartesianGrid & grid_1,
                                                  const interpolant::CartesianGrid & grid_2) {
    interpolant::CartesianGrid result(grid_1);
    return result += grid_2;
}

}  // namespace merlin
