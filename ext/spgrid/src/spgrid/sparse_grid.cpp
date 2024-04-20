// Copyright 2022 quocdang1998
#include "spgrid/sparse_grid.hpp"

#include <algorithm>      // std::equal
#include <cinttypes>      // PRIu64
#include <sstream>        // std::ostringstream
#include <string_view>    // std::string_view
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

#include "merlin/logger.hpp"  // merlin::Fatal
#include "merlin/utils.hpp"   // merlin::prod_elements, merlin::contiguous_to_ndim_idx

#include "spgrid/utils.hpp"  // spgrid::get_max_levels, spgrid::get_hiearchical_index, spgrid::shape_from_level

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// The size of grid vector is valid (valid shape are 2 and 2^(k > 1) +1)
static inline bool is_valid_shape(std::uint64_t shape) { return ((shape != 1) && ((shape - 1) & (shape - 2)) == 0); }

// ---------------------------------------------------------------------------------------------------------------------
// LevelIterator
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from level number
LevelIterator::LevelIterator(const SparseGrid & grid, std::uint64_t level) :
p_grid(&grid), level(level), subgrid_shape(grid.ndim()) {
    if (level > grid.nlevel()) {
        mln::Fatal<std::invalid_argument>("Level must be smaller than the max level of the grid.\n");
    }
    // calculate signature and cumulated grid
    this->cum_idx = mln::Vector<std::set<std::uint64_t>>(grid.ndim());
    for (std::uint64_t l = 0; l <= level; l++) {
        mln::UIntVec subgrid_level = grid.get_ndlevel_at_index(l);
        ndlevel_to_shape(subgrid_level, this->subgrid_shape.data());
        this->signature += mln::prod_elements(this->subgrid_shape.data(), this->subgrid_shape.size());
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            for (std::uint64_t i_node = 0; i_node < this->subgrid_shape[i_dim]; i_node++) {
                this->cum_idx[i_dim].insert(get_hiearchical_index(i_node, subgrid_level[i_dim], grid.shape()[i_dim]));
            }
        }
    }
}

// Prefix increment
void LevelIterator::operator++(void) {
    ++this->level;
    if (this->level == this->p_grid->nlevel()) {
        this->signature = 0;
        return;
    }
    mln::UIntVec subgrid_level = this->p_grid->get_ndlevel_at_index(this->level);
    ndlevel_to_shape(subgrid_level, this->subgrid_shape.data());
    this->signature += mln::prod_elements(this->subgrid_shape.data(), this->subgrid_shape.size());
    for (std::uint64_t i_dim = 0; i_dim < this->subgrid_shape.size(); i_dim++) {
        for (std::uint64_t i_node = 0; i_node < this->subgrid_shape[i_dim]; i_node++) {
            this->cum_idx[i_dim].insert(
                get_hiearchical_index(i_node, subgrid_level[i_dim], this->p_grid->shape()[i_dim]));
        }
    }
}

// String representation
std::string LevelIterator::str(void) const {
    std::ostringstream os;
    os << "<LevelIterator(";
    for (const std::set<std::uint64_t> & idx : this->cum_idx) {
        os << "<";
        for (const std::uint64_t & i : idx) {
            os << i << " ";
        }
        os << ">";
    }
    os << ")>";
    return os.str();
}

// ---------------------------------------------------------------------------------------------------------------------
// SparseGrid
// ---------------------------------------------------------------------------------------------------------------------

// Constructor a full sparse grid from vectors of components
SparseGrid::SparseGrid(const mln::Vector<mln::DoubleVec> & full_grid_vectors,
                       const std::function<bool(const mln::UIntVec &)> accept_condition) :
full_grid_(full_grid_vectors) {
    // check shape of each dimension
    const mln::Index & shape = this->full_grid_.shape();
    for (std::uint64_t i_dim = 0; i_dim < full_grid_.ndim(); i_dim++) {
        if (!is_valid_shape(shape[i_dim])) {
            mln::Fatal<std::invalid_argument>("Size of each grid vector must be 2^n + 1, dimension %" PRIu64
                                              " got %" PRIu64 ".\n", i_dim, shape[i_dim]);
        }
    }
    // loop on each level vector and add to grid if accept condition is satisfied
    mln::UIntVec max_levels = get_max_levels(shape.data(), full_grid_.ndim());
    mln::UIntVec level_index(max_levels.size());
    std::uint64_t total_levels = mln::prod_elements(max_levels.data(), max_levels.size());
    std::vector<std::uint64_t> level_index_buffer;
    for (std::uint64_t i = 0; i < total_levels; i++) {
        mln::contiguous_to_ndim_idx(i, max_levels.data(), max_levels.size(), level_index.data());
        if (accept_condition(level_index)) {
            level_index_buffer.insert(level_index_buffer.end(), level_index.begin(), level_index.end());
        }
    }
    this->nd_level_ = mln::UIntVec(level_index_buffer.data(), &(*level_index_buffer.end()));
}

// Constructor sparse grid from vector of components and level index vectors
SparseGrid::SparseGrid(const mln::Vector<mln::DoubleVec> & full_grid_vectors, const mln::UIntVec & level_vectors) :
full_grid_(full_grid_vectors), nd_level_(level_vectors) {
    // check shape of each dimension
    const mln::Index & shape = this->full_grid_.shape();
    for (std::uint64_t i_dim = 0; i_dim < full_grid_vectors.size(); i_dim++) {
        if (!is_valid_shape(shape[i_dim])) {
            mln::Fatal<std::invalid_argument>("Size of each grid vector must be 2^n + 1, dimension %" PRIu64
                                              " got %" PRIu64 ".\n",
                                              i_dim, shape[i_dim]);
        }
    }
    // check level vector
    if (level_vectors.size() % this->full_grid_.ndim() != 0) {
        mln::Fatal<std::invalid_argument>("Size of level vector must be a multiple of ndim, got %" PRIu64
                                          " and %" PRIu64 ".\n",
                                          level_vectors.size(), this->full_grid_.ndim());
    }
    // loop on each level vector
    mln::UIntVec max_levels = get_max_levels(shape.data(), full_grid_vectors.size());
    std::uint64_t num_levels = level_vectors.size() / this->full_grid_.ndim();
    std::unordered_set<std::string_view> unique_vectors;
    for (std::uint64_t i_level = 0; i_level < num_levels; i_level++) {
        mln::UIntVec nd_level = this->get_ndlevel_at_index(i_level);
        // check if each level vector are smaller than the max possible level
        bool is_less_than_max = std::equal(nd_level.begin(), nd_level.end(), max_levels.begin(),
                                           [](std::uint64_t & a, std::uint64_t & b) { return a < b; });
        if (!is_less_than_max) {
            mln::Fatal<std::invalid_argument>("Invalid level %s (max level allowed for provided grid is %s).\n",
                                              nd_level.str().c_str(), max_levels.str().c_str());
        }
        // check for duplication
        std::string_view str_view(reinterpret_cast<char *>(nd_level.data()), sizeof(std::uint64_t) * nd_level.size());
        if (auto it = unique_vectors.find(str_view); it != unique_vectors.end()) {
            mln::Fatal<std::invalid_argument>( "Duplicated level index found.\n");
        }
        unique_vectors.insert(str_view);
    }
}

// Get Cartesian grid at a given level index
/*mln::grid::CartesianGrid SparseGrid::get_grid_at_level(std::uint64_t index) const noexcept {
    mln::UIntVec level = this->get_ndlevel_at_index(index);
    // initialize vector of grid nodes
    mln::Vector<mln::DoubleVec> grid_vectors(this->ndim());
    // create grid vector for each dimension
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        std::uint64_t subgrid_shape = shape_from_level(level[i_dim]);
        grid_vectors[i_dim] = mln::DoubleVec(subgrid_shape);
        for (std::uint64_t i_node = 0; i_node < subgrid_shape; i_node++) {
            std::uint64_t index = get_hiearchical_index(i_node, level[i_dim], this->shape()[i_dim]);
            grid_vectors[i_dim][i_node] = this->full_grid_.grid_vectors()[i_dim][index];
        }
    }
    return mln::grid::CartesianGrid(grid_vectors);
}*/

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
