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

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficient
// --------------------------------------------------------------------------------------------------------------------

// Calculate coefficients for cartesian grid (supposed shape value == shape of coeff)
void interpolant::calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::NdData & value,
                                         array::NdData & coeff) {
    // get associated 1D grid to calculate on
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - value.ndim()];
    std::printf("Current grid vector: %s\n", grid_vector.str().c_str());
    // copy the first array corresponding to index=0
    if (&coeff != &value) {
        array_copy(&coeff, &value, std::memcpy);  // to be reviewed
    }
    // trivial case (1D)
    if (coeff.ndim() == 1) {
        std::printf("Get trivial case\n");
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
            std::printf("Parsing non-trivial-case for i = %d, k = %d:\n", int(i), int(k));
            // get NdData of sub slice
            Vector<array::Slice> slice_k(coeff.ndim()), slice_k_1(coeff.ndim());
            slice_k[0] = array::Slice({k});
            slice_k_1[0] = array::Slice({k-1});
            const array::NdData * p_array_k = slice_on(coeff, slice_k);
            std::printf("Array k: %s\n", p_array_k->str().c_str());
            const array::NdData * p_array_k_1 = slice_on(coeff, slice_k_1);
            std::printf("Array k-1: %s\n", p_array_k_1->str().c_str());
            array::NdData * p_result = slice_on(coeff, slice_k);
            // calculate divide difference
            divide_difference(*p_array_k, *p_array_k_1, grid_vector[k], grid_vector[k-i], *p_result);
            std::printf("Array result: %s\n", p_result->str().c_str());
            // deallocate memory
            delete p_array_k;
            delete p_array_k_1;
            delete p_result;
        }
    }
    std::printf("Result after parsing non-trivial case: %s\n", coeff.str().c_str());
    // recursively calculate divide difference for dimension from i-1-th
    #pragma omp parallel for
    for (std::int64_t i = 0; i < coeff.shape()[0]; i++) {
        Vector<array::Slice> slice_i(coeff.ndim());
        slice_i[0] = array::Slice({static_cast<std::uint64_t>(i)});
        array::NdData * p_coeff_i = slice_on(coeff, slice_i);
        p_coeff_i->collapse(true);
        interpolant::calc_newton_coeffs_cpu(grid, *p_coeff_i, *p_coeff_i);
        delete p_coeff_i;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// --------------------------------------------------------------------------------------------------------------------

double interpolant::eval_newton_cpu(const interpolant::CartesianGrid & grid, const array::NdData & coeff,
                                    const Vector<double> & x) {
    long double result = 0;
    std::uint64_t ndim = grid.ndim();
    const Vector<double> & grid_vector = grid.grid_vectors()[ndim - coeff.ndim()];
    // trivial case
    if (coeff.ndim() == 1) {
        const std::uint64_t & shape = coeff.shape()[0];
        result += coeff.get({shape-1});
        for (std::int64_t i = shape-2; i >= 0; i++) {
            result *= (x[ndim - coeff.ndim()] - grid_vector[i]);
            result += coeff.get({static_cast<std::uint64_t>(i)});
        }
        return result;
    }

}

}  // namespace merlin
