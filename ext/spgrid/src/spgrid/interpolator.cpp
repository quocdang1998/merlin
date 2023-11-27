// Copyright 2022 quocdang1998
#include "spgrid/interpolator.hpp"

#include <vector>  // std::vector

#include <omp.h>  // #pragma omp

#include "unistd.h"

#include "merlin/array/array.hpp"   // merlin::array::Array
#include "merlin/logger.hpp"        // FAILURE
#include "merlin/splint/tools.hpp"  // merlin::splint::construct_coeff_cpu, merlin::splint::eval_intpl_cpu
#include "merlin/utils.hpp"         // merlin::ptr_to_subsequence

#include "spgrid/utils.hpp"        // spgrid::merge_grid

namespace spgrid {

// ---------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// ---------------------------------------------------------------------------------------------------------------------




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
    // loop for each level and calculate coefficients
    std::vector<double> sparsegrid_data_buffer;
    merlin::grid::CartesianGrid total_grid;
    std::uint64_t nlevel = grid.nlevel();
    for (std::uint64_t i_level = 0; i_level < nlevel; i_level++) {
        // isolate cartesian grid
        merlin::grid::CartesianGrid level_grid(grid.get_grid_at_level(i_level));
        // merge cartesian grid
        merge_grid(total_grid, level_grid);
    }




/*
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
    std::printf("Shape_of_level: %s\n", shape_of_levels.str().c_str());
    // multi-variate interpolation on each level
    std::vector<char> level_i_buffer(grid.fullgrid().cumalloc_size()), level_j_buffer(grid.fullgrid().cumalloc_size());
    for (std::uint64_t i_level = 0; i_level < grid.nlevel(); i_level++) {
        // calculate coefficient of the current level
        merlin::grid::CartesianGrid level_grid(grid.get_grid_at_level(i_level, level_i_buffer.data()));
        merlin::splint::construct_coeff_cpu(nullptr, this->coeff_by_level_[i_level], &level_grid, &method, n_threads);
        // subtract each successive level by the interpolated value
        for (std::uint64_t j_level; j_level < grid.nlevel(); j_level++) {
            merlin::grid::CartesianGrid levelj_grid(grid.get_grid_at_level(j_level, level_j_buffer.data()));
            merlin::array::Array points(levelj_grid.get_points());
            merlin::floatvec interpolated_value(levelj_grid.size());
            merlin::splint::eval_intpl_cpu(nullptr, this->coeff_by_level_[i_level], &level_grid, &method,
                                           points.data(), levelj_grid.size(), interpolated_value.data(), n_threads);
            for (std::uint64_t i_point = 0; i_point < levelj_grid.size(); i_point++) {
                this->coeff_by_level_[j_level][i_point] -= interpolated_value[i_point];
            }
        }
    }
*/
}

// Evaluate interpolation by CPU parallelism
merlin::floatvec Interpolator::evaluate(const merlin::array::Array & points, std::uint64_t n_threads) {
    // check points array
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->grid_.ndim()) {
        FAILURE(std::invalid_argument, "Array of coordinates and interpolator have different dimension.\n");
    }
    // evaluate interpolation
    merlin::floatvec evaluated_values(points.shape()[0]);
    /*
    #pragma omp parallel for num_threads(n_threads)
    for (std::uint64_t i_point = 0; i_point < evaluated_values.size(); i_point++) {
        // get pointer to point data
        const double * point_data = points.data() + i_point * this->grid_.ndim();
        // evaluate for each level
        std::vector<char> level_buffer(this->grid_.fullgrid().cumalloc_size());
        for (std::uint64_t i_level = 0; i_level < this->grid_.nlevel(); i_level++) {
            merlin::grid::CartesianGrid level_grid(this->grid_.get_grid_at_level(i_level, level_buffer.data()));
            merlin::splint::eval_intpl_cpu(nullptr, this->coeff_by_level_[i_level], &level_grid, &this->method_,
                                           point_data, 1, &(evaluated_values[i_point]), 1);
        }
    }
    */
    return evaluated_values;
}

}  // namespace spgrid
