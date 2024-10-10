// Copyright 2022 quocdang1998
#include "merlin/grid/cartesian_grid.hpp"

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <cstddef>    // nullptr
#include <cstdint>    // std::uint64_t
#include <iterator>   // std::distance
#include <numeric>    // std::iota
#include <sstream>    // std::ostringstream

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // merlin::Fatal, merlin::Warning, merlin::cuda_compile_error
#include "merlin/memory.hpp"       // merlin::memcpy_cpu_to_gpu
#include "merlin/utils.hpp"        // merlin::ptr_to_subsequence, merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Check duplication
// ---------------------------------------------------------------------------------------------------------------------

// Get sorted index
static UIntVec sorted_index(DoubleView input) {
    UIntVec index(input.size());
    std::iota(index.begin(), index.end(), 0);
    std::stable_sort(index.begin(), index.end(),
                     [&input](std::uint64_t i1, std::uint64_t i2) { return input[i1] < input[i2]; });
    return index;
}

// Check for duplicated element in a vector
static bool has_duplicated_element(DoubleView grid_vector) {
    UIntVec sorted_idx = sorted_index(grid_vector);
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
grid::CartesianGrid::CartesianGrid(const DVecArray & grid_vectors) :
grid_shape_(grid_vectors.size()), grid_vectors_(grid_vectors.size()) {
    // count number of nodes
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < grid_vectors.size(); i_dim++) {
        // check for duplicate element
        const DoubleVec & grid_vector = grid_vectors[i_dim];
        if (has_duplicated_element(grid_vector.get_view())) {
            Fatal<std::invalid_argument>("Found duplicated elements.\n");
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
            Warning("Input nodes are not sorted in increasing order, sorting the nodes...\n");
            DoubleVec sorted_nodes(grid_vector.begin(), grid_vector.end());
            std::stable_sort(sorted_nodes.begin(), sorted_nodes.end());
            for (double & node : sorted_nodes) {
                this->grid_nodes_[node_idx++] = node;
            }
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->grid_shape_.size(),
                       this->grid_vectors_.data());
    // calculate size
    this->size_ = prod_elements(this->grid_shape_.data(), this->grid_shape_.size());
}

// Constructor as a sub-grid from a larger grid
grid::CartesianGrid::CartesianGrid(const grid::CartesianGrid & whole, const SliceArray & slices) {
    // argument checking
    if (whole.ndim() != slices.size()) {
        Fatal<std::invalid_argument>("Slice array and original grid are not compatible.\n");
    }
    std::uint64_t ndim = whole.ndim();
    this->grid_shape_ = Index(ndim);
    this->grid_vectors_ = DPtrArray(ndim);
    // get new shape for each dimension and total number of nodes
    std::uint64_t num_nodes = 0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        auto [_, dim_shape, __] = slices[i_dim].slice_on(whole.grid_shape_[i_dim], sizeof(double));
        this->grid_shape_[i_dim] = dim_shape;
        num_nodes += dim_shape;
    }
    // copy value to grid node array
    this->grid_nodes_ = DoubleVec(num_nodes);
    std::uint64_t count_node = 0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        DoubleView grid_vector = whole.grid_vector(i_dim);
        for (std::uint64_t i_node = 0; i_node < this->grid_shape_[i_dim]; i_node++) {
            std::uint64_t idx_in_original = slices[i_dim].get_index_in_whole_array(i_node);
            this->grid_nodes_[count_node++] = grid_vector[idx_in_original];
        }
    }
    // calculate pointers per dimension
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), ndim, this->grid_vectors_.data());
    // calculate size
    this->size_ = prod_elements(this->grid_shape_.data(), ndim);
}

// Copy constructor
grid::CartesianGrid::CartesianGrid(const grid::CartesianGrid & src) :
grid_nodes_(src.grid_nodes_), grid_shape_(src.grid_shape_), grid_vectors_(src.ndim()), size_(src.size_) {
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim(), this->grid_vectors_.data());
}

// Copy assignment
grid::CartesianGrid & grid::CartesianGrid::operator=(const grid::CartesianGrid & src) {
    this->grid_nodes_ = src.grid_nodes_;
    this->grid_shape_ = src.grid_shape_;
    this->size_ = src.size_;
    this->grid_vectors_ = DPtrArray(src.ndim());
    ptr_to_subsequence(this->grid_nodes_.data(), this->grid_shape_.data(), this->ndim(), this->grid_vectors_.data());
    return *this;
}

void * grid::CartesianGrid::copy_to_gpu(grid::CartesianGrid * gpu_ptr, void * grid_data_ptr,
                                        std::uintptr_t stream_ptr) const {
    // copy grid node vector to GPU
    memcpy_cpu_to_gpu(grid_data_ptr, this->grid_nodes_.data(), this->num_nodes() * sizeof(double), stream_ptr);
    // initialize a clone and calculate the pointer to subsequence node
    grid::CartesianGrid cloned_obj;
    cloned_obj.grid_shape_ = this->grid_shape_;
    cloned_obj.size_ = this->size_;
    double * p_gpu_grid_nodes = reinterpret_cast<double *>(grid_data_ptr);
    cloned_obj.grid_nodes_.assign(p_gpu_grid_nodes, this->num_nodes());
    cloned_obj.grid_vectors_ = DPtrArray(this->ndim());
    ptr_to_subsequence(p_gpu_grid_nodes, this->grid_shape_.data(), this->ndim(), cloned_obj.grid_vectors_.data());
    memcpy_cpu_to_gpu(gpu_ptr, &cloned_obj, sizeof(grid::CartesianGrid), stream_ptr);
    return p_gpu_grid_nodes + this->num_nodes();
}

// Get all points in the grid
array::Array grid::CartesianGrid::get_points(void) const {
    array::Array points({this->size_, this->ndim()});
    Index point_idx(this->ndim());
    for (std::uint64_t i_point = 0; i_point < this->size_; i_point++) {
        double * point_data = &(points[{i_point, 0}]);
        this->get(i_point, point_data);
    }
    return points;
}

// String representation
std::string grid::CartesianGrid::str(void) const {
    std::ostringstream os;
    os << "<CartesianGrid(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (i_dim != 0) {
            os << " ";
        }
        DoubleView grid_vector = this->grid_vector(i_dim);
        os << grid_vector.str();
    }
    os << ")>";
    return os.str();
}

// Destructor
grid::CartesianGrid::~CartesianGrid(void) {}

}  // namespace merlin
