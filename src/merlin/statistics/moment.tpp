// Copyright 2022 quocdang1998
#ifndef MERLIN_STATISTICS_MOMENT_TPP_
#define MERLIN_STATISTICS_MOMENT_TPP_

#include <cstring>  // std::memset
#include <cmath>  // std::pow

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

// --------------------------------------------------------------------------------------------------------------------
// Moment
// --------------------------------------------------------------------------------------------------------------------

namespace merlin {

// Calculate moment on all elements of the array
template <std::uint64_t order>
std::array<double, order> statistics::powered_mean(const array::Array & data, std::uint64_t nthreads) {
    // check nthreads and order
    static_assert(order > 0, "Expected order bigger than 0.\n");
    if (nthreads == 0) {
        FAILURE(std::invalid_argument, "Expected number of threads strictly positive, got 0.\n");
    }
    // parallel calculate the sum of elements in array
    double * storing = new double[nthreads*order];
    std::memset(storing, 0, nthreads*order*sizeof(double));
    #pragma omp parallel for num_threads(nthreads)
    for (std::int64_t i_point = 0; i_point < data.size(); i_point++) {
        std::uint64_t i_thread = ::omp_get_thread_num();
        intvec index = contiguous_to_ndim_idx(i_point, data.shape());
        double element = data.get(index);
        double element_i_order = element;
        for (std::uint64_t i_order = 0; i_order < order; i_order++) {
            storing[i_thread*order + i_order] += element_i_order;
            element_i_order *= element;
        }
    }
    // return mean
    std::array<double, order> result;
    result.fill(0.0);
    #pragma omp parallel for num_threads(nthreads)
    for (std::int64_t i_order = 0; i_order < order; i_order++) {
        for (std::uint64_t i_thread = 0; i_thread < nthreads; i_thread++) {
            result[i_order] += storing[i_thread*order + i_order];
        }
        result[i_order] /= data.size();
    }
    delete[] storing;
    return result;
}

// Calculate moment
template<std::uint64_t order>
double statistics::moment_cpu(const std::array<double, order> & esperances) {
    double result = 0.0;
    std::uint64_t binom_coeff = 1;
    double pow_mean = 1.0;
    for(std::uint64_t i = 0; i < order; i++) {
        result += binom_coeff * esperances[order-1-i] * pow_mean;
        pow_mean *= - esperances[0];
        binom_coeff *= (order-i);
        binom_coeff /= i+1;
    }
    result += pow_mean;
    return result;
}

}  // namespace merlin

#endif  // MERLIN_STATISTICS_MOMENT_TPP_
