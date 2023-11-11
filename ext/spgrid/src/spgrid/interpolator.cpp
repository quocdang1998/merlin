// Copyright 2022 quocdang1998
#include "spgrid/interpolator.hpp"

#include <vector>  // std::vector

#include <omp.h>  // #pragma omp

#include "merlin/logger.hpp"        // FAILURE
#include "merlin/splint/tools.hpp"  // merlin::splint::construct_coeff_cpu, merlin::splint::eval_intpl_cpu
#include "merlin/utils.hpp"         // merlin::ptr_to_subsequence

#include "spgrid/utils.hpp"        // spgrid::get_npoint_in_level, spgrid::fullgrid_idx_from_subgrid

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a hierarchical grid and a full Cartesian data
Interpolator::Interpolator(const SparseGrid & grid, const merlin::array::NdData & value,
                           const merlin::Vector<merlin::splint::Method> & method, std::uint64_t n_threads) :
grid_(grid), method_(method), coeff_by_level_(grid.nlevel()) {
    // check shape
    if (grid.shape() != value.shape()) {
        FAILURE(std::invalid_argument, "Inconsistent shape between grid and value.\n");
    }
    if (grid.ndim() != method.size()) {
        FAILURE(std::invalid_argument, "Invalid ndim of method vector.\n");
    }
    // isolate sparse grid data
    std::vector<double> sparsegrid_data_buffer;
    std::uint64_t nlevel = grid.nlevel();
    merlin::intvec shape_of_levels(nlevel);
    for (std::uint64_t i_level = 0; i_level < nlevel; i_level++) {
        merlin::intvec level_vector = grid.get_ndlevel_at_index(i_level);
        std::uint64_t num_points_in_level = get_npoint_in_level(level_vector);
        shape_of_levels[i_level] = num_points_in_level;
        for (std::uint64_t i_point = 0; i_point < num_points_in_level; i_point++) {
            merlin::intvec index_fullgrid = fullgrid_idx_from_subgrid(i_point, level_vector, grid.shape());
            sparsegrid_data_buffer.push_back(value.get(index_fullgrid));
        }
    }
    this->coeff_ = merlin::floatvec(sparsegrid_data_buffer.data(), sparsegrid_data_buffer.size());
    merlin::ptr_to_subsequence(this->coeff_.data(), shape_of_levels, this->coeff_by_level_.data());
    // multi-variate interpolation on each level
    std::vector<char> level_i_buffer(grid.fullgrid().cumalloc_size()), level_j_buffer(grid.fullgrid().cumalloc_size());
    for (std::uint64_t i_level = 0; i_level < grid.nlevel(); i_level++) {
        // calculate coefficient of the current level
        merlin::splint::CartesianGrid level_grid(grid.get_grid_at_level(i_level, level_i_buffer.data()));
        merlin::splint::construct_coeff_cpu(nullptr, this->coeff_by_level_[i_level], &level_grid, &method, n_threads);
        // subtract each successive level by the interpolated value
        for (std::uint64_t j_level; j_level < grid.nlevel(); j_level++) {
            merlin::splint::CartesianGrid levelj_grid(grid.get_grid_at_level(j_level, level_j_buffer.data()));
            #pragma omp parallel for num_threads(n_threads)
            for (std::uint64_t j_point = 0; j_point < levelj_grid.size(); j_point++) {
                merlin::floatvec point(levelj_grid[j_point]);
                double interpolated_value = 0;
                merlin::splint::eval_intpl_cpu(nullptr, this->coeff_by_level_[i_level], &level_grid, &method,
                                               point.data(), 1, &interpolated_value, 1);
                this->coeff_by_level_[j_level][j_point] -= interpolated_value;
            }
        }
    }
}

}  // namespace spgrid
