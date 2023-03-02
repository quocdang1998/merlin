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
static void divide_difference(const array::Array & a1, const array::Array & a2,
                              double x1, double x2, array::Array & result) {
    long double denominator = x1 - x2;
    std::uint64_t size = a1.size();
    for (std::uint64_t i = 0; i < size; i++) {
        intvec index = contiguous_to_ndim_idx(i, a1.shape());
        double div_diff = (a1.get(index) - a2.get(index)) / denominator;
        result.set(i, div_diff);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
void interpolant::calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                                         array::Array & coeff) {
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - value.ndim()];
    // copy the first array corresponding to index=0
    if (&coeff != &value) {
        array::array_copy(&coeff, &value, std::memcpy);
    }
    // trivial case (1D)
    if (coeff.ndim() == 1) {
        for (std::uint64_t i = 1; i < coeff.shape()[0]; i++) {
            for (std::uint64_t k = coeff.shape()[0]-1; k >=i; k--) {
                long double coeff_calc = (coeff.get({k}) - coeff.get({k-1})) / (grid_vector[k] - grid_vector[k-i]);
                coeff.set({k}, coeff_calc);
            }
        }
        return;
    }
    // calculate divdiff on dim i-th
    for (std::uint64_t i = 1; i < coeff.shape()[0]; i++) {
        for (std::uint64_t k = coeff.shape()[0]-1; k >= i; k--) {
            // get NdData of sub slice
            Vector<array::Slice> slice_k(coeff.ndim()), slice_k_1(coeff.ndim());
            slice_k[0] = array::Slice({k});
            slice_k_1[0] = array::Slice({k-1});
            const array::Array array_k(coeff, slice_k);
            const array::Array array_k_1(coeff, slice_k_1);
            array::Array array_result(coeff, slice_k);
            // calculate divide difference
            divide_difference(array_k, array_k_1, grid_vector[k], grid_vector[k-i], array_result);
        }
    }
    // recursively calculate divide difference for dimension from i-1-th
    #pragma omp parallel for
    for (std::int64_t i = 0; i < coeff.shape()[0]; i++) {
        Vector<array::Slice> slice_i(coeff.ndim());
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array::Array array_coeff_i(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        interpolant::calc_newton_coeffs_cpu(grid, array_coeff_i, array_coeff_i);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------

// Evaluate Newton interpolation on a full Cartesian grid using CPU (supposed shape of grid == shape of coeff)
double interpolant::eval_newton_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                                    const Vector<double> & x) {
    long double result = 0;
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - coeff.ndim()];
    // trivial case
    if (coeff.ndim() == 1) {
        const std::uint64_t & shape = coeff.shape()[0];
        result += coeff.get({shape-1});
        for (std::int64_t i = shape-2; i >= 0; i--) {
            result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
            result += coeff.get({static_cast<std::uint64_t>(i)});
        }
        return result;
    }
    // recursively calculate for non-trivial case
    const std::uint64_t & shape = coeff.shape()[0];
    Vector<array::Slice> slice_i(coeff.ndim());
    slice_i[0] = array::Slice({shape-1});
    array::Array array_coeff_i(coeff, slice_i);
    array_coeff_i.remove_dim(0);
    result += interpolant::eval_newton_cpu(grid, array_coeff_i, x);
    for (std::int64_t i = shape-2; i >= 0; i--) {
        result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array_coeff_i = array::Array(coeff, slice_i);
        array_coeff_i.remove_dim(0);
        result += interpolant::eval_newton_cpu(grid, array_coeff_i, x);
    }
    return result;
}

}  // namespace merlin
