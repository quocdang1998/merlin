// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <omp.h>  // pragma omp

#include <algorithm>  // std::max
#include <cinttypes>  // PRIu64
#include <utility>  // std::pair

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CatesianGrid
#include "merlin/interpolant/sparse_grid.hpp"  // merlin::interpolant::SparseGrid
                                               // merlin::interpolant::get_cartesian_grid
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/vector.hpp"  // merlin::floatvec, merlin::intvec

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// --------------------------------------------------------------------------------------------------------------------

static Vector<array::Slice> get_slice_from_level(const intvec & level, const intvec & max_level) {
    Vector<array::Slice> result(level.size());
    for (std::uint64_t i_dim = 0; i_dim < level.size(); i_dim++) {
        // calculate the size of max_level
        const std::uint64_t & max = max_level[i_dim];
        std::uint64_t size = (max == 0) ? 1 : (1 << max) + 1;
        // calculate start point and step
        const std::uint64_t & lv = level[i_dim];
        if (lv == 0) {
            result[i_dim] = array::Slice({(size - 1) / 2});
        } else if (lv == 1) {
            result[i_dim].step() = size-1;
        } else {
            result[i_dim].step() = 1 << (max_level[i_dim] - level[i_dim] + 1);
            result[i_dim].start() = result[i_dim].step() >> 1;
        }
    }
    return result;
}

array::Array calc_lagrange_coeffs_cpu(const interpolant::SparseGrid * pgrid, const array::Array * pvalue) {
    // check size of p_value
    if (pvalue->ndim() != 1) {
        FAILURE(std::runtime_error, "Invalid shpae of value array.\n");
    }
    if (pvalue->size() != pgrid->size()) {
        FAILURE(std::invalid_argument, "Expected value array and grid having the same size.");
    }
    // loop over each level, updat maximum current level
    std::uint64_t num_level = pgrid->level_vectors().size() / pgrid->ndim();
    intvec max_current_level(pgrid->ndim(), 0);
    interpolant::CartesianGrid current_cart_grid(pgrid->ndim());
    array::Array result(pvalue->shape());
    for (std::uint64_t i_level = 0; i_level < num_level; ++i_level) {
        intvec level;
        level.data() = const_cast<std::uint64_t *>(pgrid->level_vectors().data()) + i_level*pvalue->ndim();
        level.size() = pvalue->ndim();
        // update max current level
        for (std::uint64_t i_dim = 0; i_dim < max_current_level.size(); i_dim++) {
            max_current_level[i_dim] = std::max(max_current_level[i_dim], level[i_dim]);
        }
        // calculate union of cartesian grid
        current_cart_grid += interpolant::get_cartesian_grid(*p_grid, i_level);
        // get values of cart grid
        array::Slice sub_slice(pgrid->sub_grid_start_index()[i_level], pgrid->sub_grid_start_index()[i_level+1], 1);
        Vector<array::Slice> vec_slice({sub_slice});
        array::Array level_value(*pvalue, Vector<array::Slice>({sub_slice}));
        // get slices
        Vector<array::Slice> level_slc = get_slice_from_level(level, max_current_level);
        // get sub array of result
        array::Array level_coeff(result, vec_slice);
        // calculate coefficient
        calc_lagrange_coeffs_cpu(&current_cart_grid, &level_value, level_slc, &level_coeff);
        // neutralize level vector
        level.data() = nullptr;
    }
    return result;
}

// --------------------------------------------------------------------------------------------------------------------
// CartesianInterpolant
// --------------------------------------------------------------------------------------------------------------------

interpolant::CartesianInterpolant::CartesianInterpolant(const CartesianGrid & grid, const array::Array & value,
                                                        const Vector<array::Slice> & slices,
                                                        interpolant::PolynomialInterpolant::Method method) :
grid_(&grid), coeff_(value.shape()) {
    // check ndim
    std::uint64_t ndim = grid.ndim();
    if (value.ndim() != ndim) {
        FAILURE(std::invalid_argument, "Ndim of grid (%" PRIu64 ") and value array (%" PRIu64 ") are different.\n",
                ndim, value.ndim());
    }
    if (slices.size() != ndim) {
        FAILURE(std::invalid_argument, "Ndim of grid (%" PRIu64 ") and slice vector size %" PRIu64 ") must equals.\n",
                ndim, slices.size());
    }
    // check size of value array
    for (std::uint64_t i_dim = 0; i_dim < ndim; i++) {
        auto [_, expected_shape, __] = slices[i_dim].slice_on(value.shape()[i_dim], sizeof(double));
    }
    // calculate coefficient
    switch (method) {
    case interpolant::PolynomialInterpolant::Method::Lagrange:
        interpolant::calc_lagrange_coeffs_cpu(grid, value, slices, this->coeff_);
        break;
    default:
        FAILURE(std::runtime_error, "Configuration not implemented.\n");
        break;
    }
}

}  // namespace merlin
