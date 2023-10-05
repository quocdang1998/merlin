// Copyright 2023 quocdang1998
#include "merlin/statistics/moment.hpp"

#include <cinttypes>  // PRIu64
#include <cmath>      // std::isnormal
#include <set>        // std::set

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // FAILURE
#include "merlin/slice.hpp"        // merlin::slicevec
#include "merlin/utils.hpp"        // merlin::contiguous_to_ndim_idx, merlin::prod_elements
#include "merlin/vector.hpp"       // merlin::intvec

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Moment
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean on all elements of the array
double statistics::mean_cpu(const array::Array & data, std::uint64_t nthreads) {
    return statistics::powered_mean<1>(data, nthreads)[0];
}

// Calculate variance on all elements of the array
double statistics::variance_cpu(const array::Array & data, std::uint64_t nthreads) {
    std::array<double, 2> means = statistics::powered_mean<2>(data, nthreads);
    return statistics::moment_cpu<2>(means);
}

// Calculate max element of the array.
double statistics::max_cpu(const array::Array & data, std::uint64_t nthreads) {
    // parallel get max element in array
    double * storing = new double[nthreads];
    std::memset(storing, 0, nthreads * sizeof(double));
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk) num_threads(nthreads)
    for (std::int64_t i_point = 0; i_point < data.size(); i_point++) {
        std::uint64_t i_thread = ::omp_get_thread_num();
        intvec index = contiguous_to_ndim_idx(i_point, data.shape());
        double element = data.get(index);
        if (!std::isnormal(element)) {
            continue;
        }
        storing[i_thread] = (element > storing[i_thread]) ? element : storing[i_thread];
    }
    // return max
    double result = 0.0;
    for (std::int64_t i_thread = 0; i_thread < nthreads; i_thread++) {
        result = (storing[i_thread] > result) ? storing[i_thread] : result;
    }
    delete[] storing;
    return result;
}

// Calculate mean for a given set of dimensions
array::Array statistics::mean_cpu(const array::Array & data, const intvec & dims, std::uint64_t nthreads) {
    // check the dimension vector
    std::set<std::uint64_t> dim_set(dims.cbegin(), dims.cend());
    if (dim_set.size() < dims.size()) {
        FAILURE(std::invalid_argument, "Dimension vector has duplicated element.\n");
    }
    intvec sorted_dims(dims.size());
    std::copy(dim_set.begin(), dim_set.end(), sorted_dims.begin());
    for (std::uint64_t i = 0; i < sorted_dims.size(); i++) {
        if (sorted_dims[i] >= data.ndim()) {
            FAILURE(std::invalid_argument, "Expected dimension smaller than %" PRIu64 ", got %" PRIu64 ".\n",
                    data.ndim(), sorted_dims[i]);
        }
    }
    // get shape and size of resulted array
    intvec result_shape(data.ndim() - dims.size());
    intvec non_collapsed_dims(result_shape.size());
    for (std::uint64_t i_whole = 0, i_collapsed = 0, i_non_collapsed = 0; i_whole < data.ndim(); i_whole++) {
        if (i_whole == sorted_dims[i_collapsed]) {
            i_collapsed++;
            continue;
        }
        result_shape[i_non_collapsed] = data.shape()[i_whole];
        non_collapsed_dims[i_non_collapsed] = i_whole;
        i_non_collapsed++;
    }
    std::uint64_t result_size = prod_elements(result_shape);
    // calculate mean by collapsed index
    array::Array result(result_shape);
    #pragma omp parallel for schedule(guided, Environment::parallel_chunk) num_threads(nthreads)
    for (std::int64_t i_point = 0; i_point < result_size; i_point++) {
        intvec non_collapsed_index = contiguous_to_ndim_idx(i_point, result_shape);
        slicevec slices(data.ndim());
        for (std::uint64_t i_dim = 0; i_dim < non_collapsed_dims.size(); i_dim++) {
            slices[non_collapsed_dims[i_dim]] = {non_collapsed_index[i_dim]};
        }
        array::Array array_slice(data, slices);
        result[non_collapsed_index] = statistics::mean_cpu(array_slice, 1);
    }
    return result;
}

}  // namespace merlin
