// Copyright 2022 quocdang1998
#include "merlin/splint/cartesian_grid.hpp"

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <cstddef>    // std::size_t
#include <iterator>   // std::distance
#include <numeric>    // std::iota
#include <sstream>    // std::ostringstream
#include <vector>     // std::vector

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"   // merlin::ptr_to_subsequence

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Check duplication
// ---------------------------------------------------------------------------------------------------------------------

// Get sorted index
static std::vector<std::size_t> sorted_index(const std::vector<double> & input) {
    std::vector<std::size_t> index(input.size());
    std::iota(index.begin(), index.end(), 0);
    std::stable_sort(index.begin(), index.end(),
                     [&input](std::size_t i1, std::size_t i2) { return input[i1] < input[i2]; });
    return index;
}

// Check for duplicated element in a vector
static bool has_duplicated_element(const std::vector<double> & grid_vector) {
    std::vector<std::size_t> sorted_idx = sorted_index(grid_vector);
    for (std::size_t i = 1; i < sorted_idx.size(); i++) {
        if (grid_vector[sorted_idx[i]] == grid_vector[sorted_idx[i - 1]]) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// ---------------------------------------------------------------------------------------------------------------------

// Construct from initializer list
splint::CartesianGrid::CartesianGrid(const Vector<floatvec> & grid_vectors) :
grid_shape_(grid_vectors.size()), grid_vectors_(grid_vectors.size()) {
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < grid_vectors.size(); i_dim++) {
        // check for duplicate element
        const floatvec & grid_vector = grid_vectors[i_dim];
        if (has_duplicated_element(std::vector<double>(grid_vector.begin(), grid_vector.end()))) {
            FAILURE(std::invalid_argument, "Found duplicated elements.\n");
        }
        // get total number of grid nodes
        num_nodes += grid_vector.size();
        // save grid shape
        this->grid_shape_[i_dim] = grid_vector.size();
    }
    // reserve vector of grid node
    this->grid_nodes_ = floatvec(num_nodes);
    // re-arrange each node into grid node vector
    std::uint64_t node_idx = 0;
    for (const floatvec & grid_vector : grid_vectors) {
        for (const double & node : grid_vector) {
            this->grid_nodes_[node_idx++] = node;
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_, this->grid_vectors_.data());
}

// Constructor from list of nodes and shape
splint::CartesianGrid::CartesianGrid(floatvec && grid_nodes, intvec && shape) :
grid_nodes_(grid_nodes), grid_shape_(shape), grid_vectors_(shape.size()) {
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_, this->grid_vectors_.data());
    // check for duplicate element
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        std::vector<double> grid_vector(this->grid_vectors_[i_dim],
                                        this->grid_vectors_[i_dim]+this->grid_shape_[i_dim]);
        if (has_duplicated_element(std::vector<double>(grid_vector))) {
            FAILURE(std::invalid_argument, "Found duplicated elements.\n");
        }
    }
}

// Constructor as a sub-grid from a larger grid
splint::CartesianGrid::CartesianGrid(const splint::CartesianGrid & whole, const slicevec & slices) :
grid_shape_(whole.ndim()), grid_vectors_(whole.ndim()) {
    // check size
    if (slices.size() != whole.ndim()) {
        FAILURE(std::invalid_argument,
                "Dimension of Slices and CartesianGrid not compatible (expected %" PRIu64 ", got %" PRIu64 ").\n",
                whole.ndim(), slices.size());
    }
    // get new shape for each dimension and total number of nodes
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim(); i_dim++) {
        auto [_, dim_shape, __] = slices[i_dim].slice_on(whole.grid_shape_[i_dim], sizeof(double));
        this->grid_shape_[i_dim] = dim_shape;
        num_nodes += dim_shape;
    }
    // copy value to grid node array
    std::uint64_t count_node = 0;
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim(); i_dim++) {
        const floatvec grid_vector = whole.grid_vector(i_dim);
        for (std::uint64_t i_node = 0; i_node < this->grid_shape_[i_dim]; i_node++) {
            std::uint64_t idx_in_original = slices[i_dim].get_index_in_whole_array(i_node);
            this->grid_nodes_[count_node++] = grid_vector[idx_in_original];
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_, this->grid_vectors_.data());
}

// Copy constructor
splint::CartesianGrid::CartesianGrid(const splint::CartesianGrid & src) :
grid_nodes_(src.grid_nodes_), grid_shape_(src.grid_shape_), grid_vectors_(src.ndim()) {
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_, this->grid_vectors_.data());
}

// Copy assignment
splint::CartesianGrid & splint::CartesianGrid::operator=(const splint::CartesianGrid & src) {
    this->grid_nodes_ = src.grid_nodes_;
    this->grid_shape_ = src.grid_shape_;
    this->grid_vectors_ = Vector<double *>(this->ndim());
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_, this->grid_vectors_.data());
    return *this;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * splint::CartesianGrid::copy_to_gpu(splint::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                         std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// String representation
std::string splint::CartesianGrid::str(void) const {
    std::ostringstream os;
    os << "<CartesianGrid(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (i_dim != 0) {
            os << " ";
        }
        const floatvec grid_vector = this->grid_vector(i_dim);
        os << grid_vector.str();
    }
    os << ")>";
    return os.str();
}

// Destructor
splint::CartesianGrid::~CartesianGrid(void) {}

}  // namespace merlin
