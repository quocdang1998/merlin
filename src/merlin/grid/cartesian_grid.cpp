// Copyright 2022 quocdang1998
#include "merlin/grid/cartesian_grid.hpp"

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <cstddef>    // std::size_t, nullptr
#include <iterator>   // std::distance
#include <numeric>    // std::iota
#include <sstream>    // std::ostringstream
#include <vector>     // std::vector

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // FAILURE
#include "merlin/utils.hpp"        // merlin::ptr_to_subsequence, merlin::prod_elements

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
grid::CartesianGrid::CartesianGrid(const Vector<DoubleVec> & grid_vectors) : ndim_(grid_vectors.size()) {
    // fill array
    this->grid_shape_.fill(0);
    this->grid_vectors_.fill(nullptr);
    // count number of nodes
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < grid_vectors.size(); i_dim++) {
        // check for duplicate element
        const DoubleVec & grid_vector = grid_vectors[i_dim];
        if (has_duplicated_element(std::vector<double>(grid_vector.begin(), grid_vector.end()))) {
            FAILURE(std::invalid_argument, "Found duplicated elements.\n");
        }
        // get total number of grid nodes
        num_nodes += grid_vector.size();
        // save grid shape
        this->grid_shape_[i_dim] = grid_vector.size();
    }
    // reserve vector of grid node
    this->grid_nodes_ = DoubleVec(num_nodes);
    // re-arrange each node into grid node vector
    std::uint64_t node_idx = 0;
    for (const DoubleVec & grid_vector : grid_vectors) {
        // if nodes are already sorted
        if (std::is_sorted(grid_vector.begin(), grid_vector.end())) {
            for (const double & node : grid_vector) {
                this->grid_nodes_[node_idx++] = node;
            }
        } else {
            WARNING("Input nodes are not sorted in increasing order, sorting the nodes...\n");
            std::vector<double> sorted_nodes(grid_vector.begin(), grid_vector.end());
            std::stable_sort(sorted_nodes.begin(), sorted_nodes.end());
            for (double & node : sorted_nodes) {
                this->grid_nodes_[node_idx++] = node;
            }
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim_, this->grid_vectors_.data());
    // calculate size
    this->size_ = prod_elements(this->grid_shape_.data(), this->ndim_);
}

// Constructor as a sub-grid from a larger grid
grid::CartesianGrid::CartesianGrid(const grid::CartesianGrid & whole, const SliceArray & slices) : ndim_(whole.ndim_) {
    // fill array
    this->grid_shape_.fill(0);
    this->grid_vectors_.fill(nullptr);
    // get new shape for each dimension and total number of nodes
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim_; i_dim++) {
        auto [_, dim_shape, __] = slices[i_dim].slice_on(whole.grid_shape_[i_dim], sizeof(double));
        this->grid_shape_[i_dim] = dim_shape;
        num_nodes += dim_shape;
    }
    // copy value to grid node array
    this->grid_nodes_ = DoubleVec(num_nodes);
    std::uint64_t count_node = 0;
    for (std::uint64_t i_dim = 0; i_dim < whole.ndim_; i_dim++) {
        const DoubleVec grid_vector = whole.grid_vector(i_dim);
        for (std::uint64_t i_node = 0; i_node < this->grid_shape_[i_dim]; i_node++) {
            std::uint64_t idx_in_original = slices[i_dim].get_index_in_whole_array(i_node);
            this->grid_nodes_[count_node++] = grid_vector[idx_in_original];
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim_, this->grid_vectors_.data());
    // calculate size
    this->size_ = prod_elements(this->grid_shape_.data(), this->ndim_);
}

// Copy constructor
grid::CartesianGrid::CartesianGrid(const grid::CartesianGrid & src) :
grid_nodes_(src.grid_nodes_), grid_shape_(src.grid_shape_), ndim_(src.ndim_), size_(src.size_) {
    this->grid_vectors_.fill(nullptr);
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim_, this->grid_vectors_.data());
}

// Copy assignment
grid::CartesianGrid & grid::CartesianGrid::operator=(const grid::CartesianGrid & src) {
    this->grid_nodes_ = src.grid_nodes_;
    this->grid_shape_ = src.grid_shape_;
    this->ndim_ = src.ndim_;
    this->size_ = src.size_;
    this->grid_vectors_.fill(nullptr);
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim_, this->grid_vectors_.data());
    return *this;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * grid::CartesianGrid::copy_to_gpu(grid::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                        std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Get all points in the grid
array::Array grid::CartesianGrid::get_points(void) const {
    // initialize result
    Index shape_data;
    shape_data.fill(0);
    shape_data[0] = this->size_;
    shape_data[1] = this->ndim_;
    array::Array points(shape_data);
    Index point_idx;
    point_idx.fill(0);
    for (std::uint64_t i_point = 0; i_point < this->size_; i_point++) {
        double * point_data = points.data() + i_point * this->ndim_;
        contiguous_to_ndim_idx(i_point, this->grid_shape_.data(), this->ndim_, point_idx.data());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
            point_data[i_dim] = this->grid_vectors_[i_dim][point_idx[i_dim]];
        }
    }
    return points;
}

// String representation
std::string grid::CartesianGrid::str(void) const {
    std::ostringstream os;
    os << "<CartesianGrid(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        if (i_dim != 0) {
            os << " ";
        }
        const DoubleVec grid_vector = this->grid_vector(i_dim);
        os << grid_vector.str();
    }
    os << ")>";
    return os.str();
}

// Destructor
grid::CartesianGrid::~CartesianGrid(void) {}

}  // namespace merlin
