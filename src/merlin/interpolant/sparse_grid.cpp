// Copyright 2022 quocdang1998
#include "merlin/interpolant/sparse_grid.hpp"

#include <algorithm>  // std::is_sorted, std::stable_sort
#include <cstdint>  // std::uint64_t
#include <cinttypes>  // PRIu64
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream
#include <vector>  // std::vector

#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::calc_subgrid_index

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// The size of grid vector is valid
static inline bool is_valid_size(std::uint64_t size) {
    return ((size != 2) && ((size-1) & (size-2)) == 0);
}

// Check the validity of a set of grid vector
static inline void check_validity(const merlin::Vector<merlin::Vector<double>> & grid_vectors) {
    const merlin::Vector<double> * data = grid_vectors.cbegin();
    for (std::uint64_t i = 0; i < grid_vectors.size(); i++) {
        // check size
        if (!is_valid_size(data[i].size())) {
            FAILURE(std::invalid_argument, "Size of each grid vector must be 2^n + 1, dimension %" PRIu64 " got %"
                    PRIu64 ".\n", i, data[i].size());
        }
        // check sorted
        if (!std::is_sorted(data[i].cbegin(), data[i].cend())) {
            FAILURE(std::invalid_argument, "Grid vector of dimension %d is not sorted.\n", i);
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// SparseGrid
// --------------------------------------------------------------------------------------------------------------------

namespace merlin {

// Constructor isotropic grid from grid vectors
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    // calculate number of level in full grid
    intvec maxlevel = this->max_levels();
    std::uint64_t num_subgrid = 1;
    for (std::uint64_t i_dim = 0; i_dim < maxlevel.size(); i_dim++) {
        num_subgrid *= ++maxlevel[i_dim];
    }
    // initialize level index vector
    this->level_index_ = intvec(num_subgrid*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(num_subgrid+1, 0);
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid; i_subgrid++) {
        intvec subgrid_index = contiguous_to_ndim_idx(i_subgrid, maxlevel);
        intvec dummy_level;
        dummy_level.assign(&(this->level_index_[i_subgrid*this->ndim()]), this->ndim());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            dummy_level[i_dim] = subgrid_index[i_dim];
        }
        std::uint64_t subgrid_size = calc_subgrid_size(subgrid_index);
        this->sub_grid_start_index_[i_subgrid+1] = this->sub_grid_start_index_[i_subgrid] + subgrid_size;
    }
}

// Constructor anisotropic grid from grid vectors
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors,
                                    std::uint64_t max, const intvec & weight) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    // calculate number of level in full grid
    intvec maxlevel = this->max_levels();
    std::uint64_t num_subgrid_in_fullgrid = 1;
    for (std::uint64_t i_dim = 0; i_dim < maxlevel.size(); i_dim++) {
        num_subgrid_in_fullgrid *= ++maxlevel[i_dim];
    }
    // loop over each subgrid index and see if it is in the sparse grid
    std::vector<intvec> level_vector_storage;
    std::vector<std::uint64_t> alpha_level;
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid_in_fullgrid; i_subgrid++) {
        intvec subgrid_index = contiguous_to_ndim_idx(i_subgrid, maxlevel);
        std::uint64_t alpha_l = inner_prod(subgrid_index, weight);
        if (alpha_l > max) {
            continue;  // exclude if alpha_l > max
        }
        alpha_level.push_back(alpha_l);
        level_vector_storage.push_back(std::move(subgrid_index));
    }
    // sort according to its alpha_l
    std::vector<std::uint64_t> sorted_index(level_vector_storage.size());
    std::iota(sorted_index.begin(), sorted_index.end(), 0);
    std::stable_sort(sorted_index.begin(), sorted_index.end(),
                     [&alpha_level] (std::uint64_t i1, std::uint64_t i2) {return alpha_level[i1] < alpha_level[i2];});
    // copying value to level_vectors and calculate start index
    this->level_index_ = intvec(level_vector_storage.size()*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(level_vector_storage.size()+1, 0);
    for (const std::uint64_t & i_subgrid : sorted_index) {
        intvec dummy_level;
        dummy_level.assign(&(this->level_index_[i_subgrid*this->ndim()]), this->ndim());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            dummy_level[i_dim] = level_vector_storage[i_subgrid][i_dim];
        }
        std::uint64_t subgrid_size = calc_subgrid_size(level_vector_storage[i_subgrid]);
        this->sub_grid_start_index_[i_subgrid+1] = this->sub_grid_start_index_[i_subgrid] + subgrid_size;
    }
}

// Construct sparse grid from vector of components and level index
interpolant::SparseGrid::SparseGrid(std::initializer_list<Vector<double>> grid_vectors,
                                    const Vector<intvec> & level_vectors) : grid_vectors_(grid_vectors) {
    // check validity
    check_validity(this->grid_vectors_);
    intvec max_level = this->max_levels();
    for (const intvec & level_vector : level_vectors) {
        if (level_vector.size() != this->ndim()) {
            FAILURE(std::invalid_argument, "Expected level index vector of size %" PRIu64 ", got size %" PRIu64 ".\n",
                    this->ndim(), level_vector.size());
        }
        for (std::uint64_t i_dim = 0; i_dim < level_vectors.size(); i_dim++) {
            if (level_vector[i_dim] > max_level[i_dim]) {
                FAILURE(std::invalid_argument, "Index of level vector cannot be bigger than max level.\n");
            }
        }
    }
    // copy level vectors
    this->level_index_ = intvec(level_vectors.size()*this->ndim(), 0);
    this->sub_grid_start_index_ = intvec(level_vectors.size()+1, 0);
    for (std::uint64_t i_level = 0; i_level < level_vectors.size(); i_level++) {
        intvec dummy_level;
        dummy_level.assign(&(this->level_index_[i_level*this->ndim()]), this->ndim());
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            dummy_level[i_dim] = level_vectors[i_level][i_dim];
        }
        std::uint64_t subgrid_size = calc_subgrid_size(dummy_level);
        this->sub_grid_start_index_[i_level+1] = this->sub_grid_start_index_[i_level] + subgrid_size;
    }
}

// Get Cartesian Grid corresponding to a given level vector
interpolant::CartesianGrid interpolant::SparseGrid::get_cartesian_grid(const intvec & level_vector) const {
    // check for valid level vector
    intvec maxlevel = this->max_levels();
    if (level_vector.size() != this->ndim()) {
        FAILURE(std::invalid_argument, "Expected level vector of size %" PRIu64 ", got %" PRIu64 ".\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (level_vector[i_dim] > maxlevel[i_dim]) {
            FAILURE(std::invalid_argument, "Expected level index smaller or equal to max level.\n");
        }
    }
    // initialize cartesian grid vector
    Vector<Vector<double>> cart_grid_vectors(this->ndim());
    for (std::uint64_t i = 0; i < this->ndim(); i++) {
        intvec dim_index = hiearchical_index(level_vector[i], this->grid_vectors_[i].size());
        Vector<double> points(dim_index.size());
        for (std::uint64_t j = 0; j < points.size(); j++) {
            points[j] = this->grid_vectors_[i][dim_index[j]];
        }
        cart_grid_vectors[i] = points;
    }
    return interpolant::CartesianGrid(cart_grid_vectors);
}

// Representation
std::string interpolant::SparseGrid::str(void) const {
    std::ostringstream os;
    os << "Sparse grid:\n";
    os << "  Grid vectors:\n";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        os << "    Dimension " << i_dim << ": " << this->grid_vectors_[i_dim].str() << "\n";
    }
    os << "  Registered level vectors:\n";
    std::uint64_t num_level = this->num_level();
    for (std::uint64_t i_level = 0; i_level < num_level; i_level++) {
        intvec level_index;
        level_index.assign(const_cast<std::uint64_t *>(&(this->level_index_[i_level*this->ndim()])), this->ndim());
        os << "    Level vector " << i_level << ": " << level_index.str() << "\n";
    }
    os << "  Starting index of subgrid: " << this->sub_grid_start_index_.str();
    return os.str();
}

// Destructor
interpolant::SparseGrid::~SparseGrid(void) {}

// Retrieve values of points in the sparse grid from a full Cartesian array of value
void interpolant::copy_value_from_cartesian_array(array::NdData & dest, const array::NdData & src,
                                                  const interpolant::SparseGrid & grid) {
    // check size
    if (grid.ndim() != src.ndim()) {
        FAILURE(std::invalid_argument, "Expected grid and value array have equal number of dimension.\n");
    }
    std::uint64_t ndim = grid.ndim();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid.grid_vectors()[i_dim].size() != src.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "The grid and the value array should have the same shape.\n");
        }
    }
    if (dest.ndim() != 1) {
        FAILURE(std::invalid_argument, "Expected ndim of destination array is 1.\n");
    }
    std::uint64_t grid_size = grid.size();
    if (dest.shape()[0] != grid_size) {
        FAILURE(std::invalid_argument, "Expected length of destination array is %" PRIu64 ".\n", grid_size);
    }
    // get element by index
    for (std::uint64_t i_point = 0; i_point < grid_size; i_point++) {
        intvec index_in_full_grid = grid.index_from_contiguous(i_point);
        dest.set(intvec({i_point}), src.get(index_in_full_grid));
    }
}

}  // namespace merlin
