// Copyright 2022 quocdang1998
#include "spgrid/utils.hpp"

#include <vector>  // std::vector

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

}  // namespace spgrid
