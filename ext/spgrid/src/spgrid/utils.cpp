// Copyright 2022 quocdang1998
#include "spgrid/utils.hpp"

#include <algorithm>  // std::set_union
#include <iterator>   // std::back_inserter
#include <vector>     // std::vector

#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid

#include "spgrid/sparse_grid.hpp"  // spgrid::SparseGrid

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// Get max level from a given sparse grid shape
merlin::intvec get_max_levels(const merlin::intvec & spgrid_shape) noexcept {
    merlin::intvec max_levels(spgrid_shape.size(), 0);
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
merlin::floatvec copy_sparsegrid_data_from_cartesian(const merlin::array::NdData & full_data, const SparseGrid & grid) {
    std::vector<double> sparsegrid_data_buffer;
    std::uint64_t nlevel = grid.nlevel();
    for (std::uint64_t i_level = 0; i_level < nlevel; i_level++) {
        merlin::intvec level_vector = grid.get_ndlevel_at_index(i_level);
        std::uint64_t num_points_in_level = get_npoint_in_level(level_vector);
        for (std::uint64_t i_point = 0; i_point < num_points_in_level; i_point++) {
            merlin::intvec index_fullgrid = fullgrid_idx_from_subgrid(i_point, level_vector, grid.shape());
            sparsegrid_data_buffer.push_back(full_data.get(index_fullgrid));
        }
    }
    return merlin::floatvec(sparsegrid_data_buffer.data(), sparsegrid_data_buffer.size());
}

// Merge 2 Cartesian grid
void merge_grid(merlin::grid::CartesianGrid & total_grid, const merlin::grid::CartesianGrid & added_grid) noexcept {
    // merge grid vector on each dimension
    merlin::Vector<merlin::floatvec> merged_grid_vectors(total_grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < total_grid.ndim(); i_dim++) {
        const merlin::floatvec total_grid_vector(total_grid.grid_vector(i_dim));
        const merlin::floatvec added_grid_vector(added_grid.grid_vector(i_dim));
        std::vector<double> merged_grid_vector;
        std::set_union(total_grid_vector.begin(), total_grid_vector.end(), added_grid_vector.begin(),
                       added_grid_vector.end(), std::back_inserter(merged_grid_vector));
        merged_grid_vectors[i_dim] = merlin::floatvec(merged_grid_vector.data(), merged_grid_vector.size());
    }
    total_grid = merlin::grid::CartesianGrid(merged_grid_vectors);
}

}  // namespace spgrid
