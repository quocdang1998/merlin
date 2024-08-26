// Copyright 2022 quocdang1998
#include "spgrid/utils.hpp"

#include <algorithm>  // std::copy, std::set_union
#include <iterator>   // std::back_inserter
#include <vector>     // std::vector

#include "merlin/config.hpp"               // merlin::Index
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/utils.hpp"                // merlin::contiguous_to_ndim_idx

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// Get max level from a given sparse grid shape
mln::UIntVec get_max_levels(const std::uint64_t * spgrid_shape, std::uint64_t ndim) noexcept {
    mln::UIntVec max_levels(ndim, 0);
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        std::uint64_t shape = spgrid_shape[i_dim];
        while (shape >>= 1) {
            ++max_levels[i_dim];
        }
        max_levels[i_dim] += 1;
    }
    return max_levels;
}

// Get index of element in full grid of an hiearchical grid
std::uint64_t get_hiearchical_index(std::uint64_t subgrid_idx, std::uint64_t level,
                                    std::uint64_t fullgrid_shape) noexcept {
    std::uint64_t hiearchical_index = 0;
    if (level == 0) {
        hiearchical_index = (subgrid_idx == 0) ? 0 : (fullgrid_shape - 1);
    } else {
        std::uint64_t jump = (fullgrid_shape - 1) >> level;
        hiearchical_index = jump * (2 * subgrid_idx + 1);
    }
    return hiearchical_index;
}

// Get grid from index
mln::grid::CartesianGrid get_grid(const mln::grid::CartesianGrid & fullgrid,
                                  const mln::Vector<std::set<std::uint64_t>> & idx) {
    mln::Vector<mln::DoubleVec> grid_vectors(idx.size());
    for (std::uint64_t i_dim = 0; i_dim < grid_vectors.size(); i_dim++) {
        grid_vectors[i_dim] = mln::DoubleVec(idx[i_dim].size());
        std::set<std::uint64_t>::const_iterator it = idx[i_dim].cbegin();
        for (std::uint64_t i_node = 0; i_node < grid_vectors[i_dim].size(); i_node++) {
            grid_vectors[i_dim][i_node] = fullgrid.grid_vectors()[i_dim][*it];
            ++it;
        }
    }
    return mln::grid::CartesianGrid(grid_vectors);
}

// Get array from index
mln::array::Array get_data(const mln::array::Array & fulldata, const mln::Vector<std::set<std::uint64_t>> & idx) {
    // allocate array
    mln::Index array_shape;
    array_shape.fill(0);
    for (std::uint64_t i_dim = 0; i_dim < fulldata.ndim(); i_dim++) {
        array_shape[i_dim] = idx[i_dim].size();
    }
    mln::array::Array resulted_data(array_shape);
    // get element
    mln::Index full_index;
    full_index.fill(0);
    for (std::uint64_t i_point = 0; i_point < resulted_data.size(); i_point++) {
        mln::contiguous_to_ndim_idx(i_point, resulted_data.shape().data(), resulted_data.ndim(), full_index.data());
        for (std::uint64_t i_dim = 0; i_dim < resulted_data.ndim(); i_dim++) {
            std::set<std::uint64_t>::const_iterator it = idx[i_dim].cbegin();
            std::advance(it, full_index[i_dim]);
            full_index[i_dim] = *it;
        }
        /*std::cout << "        Ipoint: " << i_point << ", full_index:";
        for (int i = 0; i < fulldata.ndim(); i++) {
            std::cout << " " << full_index[i];
        }
        std::cout << "\n";*/
        resulted_data[i_point] = fulldata.get(full_index);
    }
    return resulted_data;
}

#ifdef __comment

// Get max level from a given sparse grid shape
mln::UIntVec get_max_levels(const mln::UIntVec & spgrid_shape) noexcept {
    mln::UIntVec max_levels(spgrid_shape.size(), 0);
    for (std::uint64_t i_dim = 0; i_dim < spgrid_shape.size(); i_dim++) {
        std::uint64_t shape = spgrid_shape[i_dim];
        while (shape >>= 1) {
            ++max_levels[i_dim];
        }
        max_levels[i_dim] += 1;
    }
    return max_levels;
}

// Copy elements from a full Cartesian data into a vector
mln::DoubleVec copy_sparsegrid_data_from_cartesian(const mln::array::NdData & full_data, const SparseGrid & grid) {
    std::vector<double> sparsegrid_data_buffer;
    std::uint64_t nlevel = grid.nlevel();
    for (std::uint64_t i_level = 0; i_level < nlevel; i_level++) {
        mln::UIntVec level_vector = grid.get_ndlevel_at_index(i_level);
        std::uint64_t num_points_in_level = get_npoint_in_level(level_vector);
        for (std::uint64_t i_point = 0; i_point < num_points_in_level; i_point++) {
            mln::UIntVec index_fullgrid = fullgrid_idx_from_subgrid(i_point, level_vector, grid.shape());
            sparsegrid_data_buffer.push_back(full_data.get(index_fullgrid));
        }
    }
    return mln::DoubleVec(sparsegrid_data_buffer.data(), sparsegrid_data_buffer.size());
}

// Merge 2 Cartesian grid
void merge_grid(mln::grid::CartesianGrid & total_grid, const mln::grid::CartesianGrid & added_grid) noexcept {
    // merge grid vector on each dimension
    mln::Vector<mln::DoubleVec> merged_grid_vectors(total_grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < total_grid.ndim(); i_dim++) {
        const mln::DoubleVec total_grid_vector(total_grid.grid_vector(i_dim));
        const mln::DoubleVec added_grid_vector(added_grid.grid_vector(i_dim));
        std::vector<double> merged_grid_vector;
        std::set_union(total_grid_vector.begin(), total_grid_vector.end(), added_grid_vector.begin(),
                       added_grid_vector.end(), std::back_inserter(merged_grid_vector));
        merged_grid_vectors[i_dim] = mln::DoubleVec(merged_grid_vector.data(), merged_grid_vector.size());
    }
    total_grid = mln::grid::CartesianGrid(merged_grid_vectors);
}

#endif

}  // namespace spgrid
