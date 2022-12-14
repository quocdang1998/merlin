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
    // terminate if the lelvel vectors is already calculated
    if (this->level_vectors_.size() != 0) {
        return;
    }
    // calculate the full grids count
    std::uint64_t full_grid_count = 1;
    intvec maxlevel = this->max_levels();
    for (int i = 0; i < maxlevel.size(); i++) {
        maxlevel[i] += 1;
        full_grid_count *= maxlevel[i];
    }
    // loop over each grid index and see if it is in the sparse grid
    intvec levels_vectors(full_grid_count * this->ndim());
    std::uint64_t sparse_grid_count = 0;
    for (int i = 0; i < full_grid_count; i++) {
        intvec grid_indx = contiguous_to_ndim_idx(i, maxlevel);
        std::uint64_t alpha_l = inner_prod(grid_indx, this->weight_);
        if (alpha_l > this->max_) {
            continue;
        }
        for (int j = 0; j < this->ndim(); j++) {
            levels_vectors[sparse_grid_count*this->ndim() + j] = grid_indx[j];
        }
        ++sparse_grid_count;
    }
    this->level_vectors_ = intvec(levels_vectors.cbegin(), sparse_grid_count*this->ndim());
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
