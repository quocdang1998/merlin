// Copyright 2022 quocdang1998
#include "merlin/interpolant/newton.hpp"

#include <cstring>  // std::memcpy

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/logger.hpp"  // CUHDERR
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/utils.hpp"  // merlin::prod_elements, merlin::contiguous_to_ndim_idx,
                             // merlin::get_level_from_valid_size, merlin::get_level_shape
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

// Concatenating 3 intvec
static intvec merge_3vectors(const intvec & v1, const intvec & v2, const intvec & v3) {
    intvec result(v1.size() + v2.size() + v3.size());
    for (std::uint64_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i];
    }
    for (std::uint64_t i = 0; i < v2.size(); i++) {
        result[v1.size()+i] = v2[i];
    }
    for (std::uint64_t i = 0; i < v3.size(); i++) {
        result[v2.size()+v1.size()+i] = v3[i];
    }
    return result;
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
            for (std::uint64_t k = coeff.shape()[0]-1; k >= i; k--) {
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

// Calculate coefficients for cartesian grid (no recursive, no slicing prototype)
void interpolant::calc_newton_coeffs_cpu2(const interpolant::CartesianGrid & grid, const array::Array & value,
                                          array::Array & coeff) {
    // copy the first array corresponding to index=0
    if (&coeff != &value) {
        std::printf("Copy data from value to array.\n");
        array::array_copy(&coeff, &value, std::memcpy);
    }
    // loop on each dimension
    for (std::uint64_t i_dim = 0; i_dim < coeff.ndim(); i_dim++) {
        std::printf("At i_dim = %d: %s\n", int(i_dim), coeff.str().c_str());
        // get grid vector at current diemnsion
        const Vector<double> & grid_vector = grid.grid_vectors()[grid.ndim() - coeff.ndim() + i_dim];
        // get shape and size of previous dimensions
        intvec shape_previous_dims;
        shape_previous_dims.assign(const_cast<std::uint64_t *>(coeff.shape().begin()), i_dim);
        std::uint64_t size_previous_dims = prod_elements(shape_previous_dims);
        std::printf("    Size of previous elements: %d\n", int(size_previous_dims));
        // get shape and size of divdiff subspace
        intvec shape_divdiff_space;
        shape_divdiff_space.assign(const_cast<std::uint64_t *>(coeff.shape().begin())+i_dim+1,
                                   const_cast<std::uint64_t *>(coeff.shape().end()));
        std::uint64_t size_divdiff_space = prod_elements(shape_divdiff_space);
        std::printf("    Size of divdiff space: %d\n", int(size_divdiff_space));
        // loop on each previous dims point
        for (std::uint64_t i_previous_dims = 0; i_previous_dims < size_previous_dims; i_previous_dims++) {
            intvec index_previous_dims = contiguous_to_ndim_idx(i_previous_dims, shape_previous_dims);
            // loop on indices of current dim for divide difference
            for (std::uint64_t i = 1; i < coeff.shape()[i_dim]; i++) {
                for (std::uint64_t k = coeff.shape()[i_dim]-1; k >= i; k--) {
                    // loop on each point in divdiff space
                    for (std::uint64_t i_divdiff_space = 0; i_divdiff_space < size_divdiff_space; i_divdiff_space++) {
                        intvec index_divdiff_space = contiguous_to_ndim_idx(i_divdiff_space, shape_divdiff_space);
                        intvec point_index_k = merge_3vectors(index_previous_dims, {k}, index_divdiff_space);
                        intvec point_index_k_1 = merge_3vectors(index_previous_dims, {k-1}, index_divdiff_space);
                        long double divdiff_result = (coeff.get(point_index_k) - coeff.get(point_index_k_1));
                        divdiff_result /= grid_vector[k] - grid_vector[k-i];
                        intvec point_index_result = std::move(point_index_k);
                        coeff.set(point_index_result, divdiff_result);
                    }
                }
            }
        }
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
