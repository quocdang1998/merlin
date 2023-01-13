// Copyright 2022 quocdang1998
#include "merlin/interpolant/sparse_grid.hpp"

#include <algorithm>  // std::is_sorted
#include <cstdint>  // std::uint64_t
#include <cinttypes>  // PRIu64

#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

static inline bool is_valid_size(std::uint64_t size) {
    return ((size != 2) && ((size-1) & (size-2)) == 0);
}

static inline void check_validity(const merlin::Vector<merlin::floatvec> & grid_vectors, bool isotropic = true) {
    const merlin::floatvec * data = grid_vectors.cbegin();
    for (int i = 0; i < grid_vectors.size(); i++) {
        // check size
        if (!is_valid_size(data[i].size())) {
            FAILURE(std::invalid_argument, "Size of each grid vector must be 2^n + 1, dimension %d got %" PRIu64 ".\n",
                    i, data[i].size());
        }
        // isotropic case : check if sizes are equals
        if (isotropic) {
            if (data[i].size() != data[0].size()) {
                FAILURE(std::invalid_argument, "Size of each grid vector must be the same in isotropic grid.\n");
            }
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
interpolant::SparseGrid::SparseGrid(std::initializer_list<floatvec> grid_vectors) : grid_vectors_(grid_vectors) {
    check_validity(this->grid_vectors_);
    this->max_ = get_level_from_size(this->grid_vectors_[0].size());
    this->weight_ = intvec(grid_vectors.size(), 1);
    this->calc_level_vectors();
}

// Constructor anisotropic grid from grid vectors
interpolant::SparseGrid::SparseGrid(std::initializer_list<floatvec> grid_vectors,
                                    std::uint64_t max, const intvec & weight) :
grid_vectors_(grid_vectors), max_(max), weight_(weight) {
    check_validity(this->grid_vectors_, false);
    this->calc_level_vectors();
}

// Calculate valid level vectors
void interpolant::SparseGrid::calc_level_vectors(void) {
    // terminate if the level vectors is already calculated
    if (this->level_vectors_.size() != 0) {
        return;
    }
    // calculate the full grids count
    std::uint64_t num_subgrid_in_fullgrid = 1;
    intvec maxlevel = this->max_levels();
    for (std::uint64_t i = 0; i < maxlevel.size(); i++) {
        num_subgrid_in_fullgrid *= ++maxlevel[i];
    }
    // loop over each subgrid index and see if it is in the sparse grid
    intvec level_vector_storage(num_subgrid_in_fullgrid * this->ndim());
    std::uint64_t num_subgrid_in_sparsegrid = 0;
    for (std::uint64_t i_subgrid = 0; i_subgrid < num_subgrid_in_fullgrid; i_subgrid++) {
        intvec subgrid_index = contiguous_to_ndim_idx(i_subgrid, maxlevel);
        std::uint64_t alpha_l = inner_prod(subgrid_index, this->weight_);
        if (alpha_l > this->max_) {
            continue;  // exclude if alpha_l > max
        }
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            level_vector_storage[num_subgrid_in_sparsegrid*this->ndim() + i_dim] = subgrid_index[i_dim];
        }
        ++num_subgrid_in_sparsegrid;
    }
    // resize by copying value to level_vectors
    this->level_vectors_ = intvec(level_vector_storage.cbegin(), num_subgrid_in_sparsegrid*this->ndim());
    // calculate start index of point for each level vector
    this->sub_grid_start_index_ = intvec(num_subgrid_in_sparsegrid, 0);
    for (std::uint64_t i_subgrid = 1; i_subgrid < num_subgrid_in_sparsegrid; i_subgrid++) {
        std::uint64_t subgrid_size = 1;
        std::uint64_t level_index = (i_subgrid-1) * this->ndim();
        for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
            std::uint64_t & dim_level = this->level_vectors_[level_index + i_dim];
            if (dim_level == 0) {
                continue;
            }
            subgrid_size *= ((dim_level == 1) ? 2 : (1 << (dim_level-1)));
        }
        this->sub_grid_start_index_[i_subgrid] = this->sub_grid_start_index_[i_subgrid-1] + subgrid_size;
    }
}

// Get Cartesian Grid corresponding to a given level vector
interpolant::CartesianGrid interpolant::SparseGrid::get_cartesian_grid(const intvec & level_vector) {
    // check for valid level vector
    intvec maxlevel = this->max_levels();
    if (level_vector.size() != this->ndim()) {
        FAILURE(std::invalid_argument, "Expected level vector of size %" PRIu64 ", got %" PRIu64 ".\n");
    }
    Vector<floatvec> cart_grid_vectors(this->ndim());
    for (int i = 0; i < this->ndim(); i++) {
        intvec dim_index = hiearchical_index(level_vector[i], this->grid_vectors_[i].size());
        floatvec points(dim_index.size());
        for (int j = 0; j < points.size(); j++) {
            points[j] = this->grid_vectors_[i][j];
        }
        cart_grid_vectors[i] = points;
    }
    return interpolant::CartesianGrid(cart_grid_vectors);
}

// Destructor
interpolant::SparseGrid::~SparseGrid(void) {}

}  // namespace merlin
