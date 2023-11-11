// Copyright 2022 quocdang1998
#include "spgrid/sparse_grid.hpp"

#include <algorithm>      // std::equal
#include <cinttypes>      // PRIu64
#include <sstream>        // std::ostringstream
#include <string_view>    // std::string_view
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"   // merlin::prod_elements, merlin::contiguous_to_ndim_idx

#include "spgrid/utils.hpp"  // spgrid::get_max_levels

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// The size of grid vector is valid (valid shape are 2 and 2^(k > 1) +1)
static inline bool is_valid_shape(std::uint64_t shape) { return ((shape != 1) && ((shape - 1) & (shape - 2)) == 0); }

// ---------------------------------------------------------------------------------------------------------------------
// SparseGrid
// ---------------------------------------------------------------------------------------------------------------------

// Constructor a full sparse grid from vectors of components
SparseGrid::SparseGrid(const merlin::Vector<merlin::floatvec> & full_grid_vectors,
                       const std::function<bool(const merlin::intvec &)> accept_condition) :
full_grid_(full_grid_vectors) {
    // check shape of each dimension
    const merlin::intvec & shape = this->full_grid_.shape();
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        if (!is_valid_shape(shape[i_dim])) {
            FAILURE(std::invalid_argument,
                    "Size of each grid vector must be 2^n + 1, dimension %" PRIu64 " got %" PRIu64 ".\n", i_dim,
                    shape[i_dim]);
        }
    }
    // loop on each level vector and add to grid if accept condition is satisfied
    merlin::intvec max_levels = get_max_levels(shape);
    std::uint64_t total_levels = merlin::prod_elements(max_levels);
    std::vector<std::uint64_t> level_index_buffer;
    for (std::uint64_t i = 0; i < total_levels; i++) {
        merlin::intvec level_index = merlin::contiguous_to_ndim_idx(i, max_levels);
        if (accept_condition(level_index)) {
            level_index_buffer.insert(level_index_buffer.end(), level_index.begin(), level_index.end());
        }
    }
    this->nd_level_ = merlin::intvec(level_index_buffer.data(), &(*level_index_buffer.end()));
}

// Constructor sparse grid from vector of components and level index vectors
SparseGrid::SparseGrid(const merlin::Vector<merlin::floatvec> & full_grid_vectors,
                       const merlin::intvec & level_vectors) :
full_grid_(full_grid_vectors), nd_level_(level_vectors) {
    // check shape of each dimension
    const merlin::intvec & shape = this->full_grid_.shape();
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        if (!is_valid_shape(shape[i_dim])) {
            FAILURE(std::invalid_argument,
                    "Size of each grid vector must be 2^n + 1, dimension %" PRIu64 " got %" PRIu64 ".\n", i_dim,
                    shape[i_dim]);
        }
    }
    // check level vector
    if (level_vectors.size() % this->full_grid_.ndim() != 0) {
        FAILURE(std::invalid_argument,
                "Size of level vector must be a multiple of ndim, got %" PRIu64 " and %" PRIu64 ".\n",
                level_vectors.size(), this->full_grid_.ndim());
    }
    // loop on each level vector
    merlin::intvec max_levels = get_max_levels(shape);
    std::uint64_t num_levels = level_vectors.size() / this->full_grid_.ndim();
    std::unordered_set<std::string_view> unique_vectors;
    for (std::uint64_t i_level = 0; i_level < num_levels; i_level++) {
        merlin::intvec nd_level = this->get_ndlevel_at_index(i_level);
        // check if each level vector are smaller than the max possible level
        bool is_less_than_max = std::equal(nd_level.begin(), nd_level.end(), max_levels.begin(),
                                           [](std::uint64_t & a, std::uint64_t & b) { return a < b; });
        if (!is_less_than_max) {
            FAILURE(std::invalid_argument, "Invalid level %s (max level allowed for provided grid is %s).\n",
                    nd_level.str().c_str(), max_levels.str().c_str());
        }
        // check for duplication
        std::string_view str_view(reinterpret_cast<char *>(nd_level.data()), sizeof(std::uint64_t) * nd_level.size());
        if (auto it = unique_vectors.find(str_view); it != unique_vectors.end()) {
            FAILURE(std::invalid_argument, "Duplicated level index found.\n");
        }
        unique_vectors.insert(str_view);
    }
}

// String representation
std::string SparseGrid::str(void) const {
    std::ostringstream os;
    os << "<SparseGrid(" << "full_grid=" << this->full_grid_.str().c_str() << ", levels=<";
    for (std::uint64_t i_level = 0; i_level < this->nlevel(); i_level++) {
        os << ((i_level != 0) ? " " : "");
        os << this->get_ndlevel_at_index(i_level).str();
    }
    os << ">)>";
    return os.str();
}

}  // namespace spgrid
