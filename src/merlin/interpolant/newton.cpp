// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include <cstring>  // std::memcpy

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/logger.hpp"  // CUHDERR
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::get_level_from_valid_size,
                             // merlin::get_level_shape
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------------------------------------------------

// Calculate divide diference between 2 arrays having the same shape: result <- (a1 - a2) / (x1 - x2)
static void divide_difference(const array::NdData & a1, const array::NdData & a2,
                              double x1, double x2, array::NdData & result) {
    long double denominator = x1 - x2;
    std::uint64_t size = a1.size();
    for (std::uint64_t i = 0; i < size; i++) {
        intvec index = contiguous_to_ndim_idx(i, a1.shape());
        double div_diff = (a1.get(index) - a2.get(index)) / denominator;
        result.set(i, div_diff);
    }
}

// Calculate divide diference when one of the array is empty: result <- a1 / (x1 - x2)
static void divide_difference(const array::NdData & a1, double x1, double x2, array::NdData & result) {
    long double denominator = x1 - x2;
    std::uint64_t size = a1.size();
    for (std::uint64_t i = 0; i < size; i++) {
        intvec index = contiguous_to_ndim_idx(i, a1.shape());
        double div_diff = a1.get(index) / denominator;
        result.set(i, div_diff);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
void interpolant::calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                                         const Vector<array::Slice> & slices, array::Array & coeff) {
    std::uint64_t full_dim_length = grid.grid_vectors()[0].size();
    const array::Slice & dim_slice = slices[0];
    for (std::uint64_t i = 1; i < full_dim_length; i++) {
        for (std::uint64_t j = full_dim_length-1; j >= i; j--) {
            // skip if both lines not in the slice
            bool in_slice_j = dim_slice.in_slice(j), in_slice_j_1 = dim_slice.in_slice(j-1);
            if ((!in_slice_j) && (!in_slice_j_1)) {
                continue;
            }
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------



}  // namespace merlin
