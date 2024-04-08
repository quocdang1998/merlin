// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include <cmath>  // std::abs, std::sqrt, std::isnormal

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/logger.hpp"       // FAILURE
#include "merlin/utils.hpp"        // merlin::contiguous_to_ndim_idx, merlin::is_normal, merlin::is_finite

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RMSE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean error with CPU parallelism
void candy::rmse_cpu(const candy::Model * p_model, const array::Array * p_data, double & result, std::uint64_t & count,
                     std::uint64_t thread_idx, std::uint64_t n_threads, Index & index_mem) noexcept {
    // initialize result
    double thread_rmse = 0.0;
    std::uint64_t thread_count = 0;
    _Pragma("omp single") {
        result = 0.0;
        count = 0;
    }
    _Pragma("omp barrier")
    // summing on all points
    for (std::uint64_t i_point = thread_idx; i_point < p_data->size(); i_point += n_threads) {
        contiguous_to_ndim_idx(i_point, p_data->shape().data(), p_data->ndim(), index_mem.data());
        double x_data = (*p_data)[index_mem];
        if (!is_normal(x_data)) {
            continue;
        }
        thread_count += 1;
        double x_model = p_model->eval(index_mem);
        double rel_err = (x_model - x_data) / x_data;
        thread_rmse += rel_err * rel_err;
    }
    _Pragma("omp barrier")
    // accumulate
    Environment::mutex.lock();
    count += thread_count;
    result += thread_rmse;
    Environment::mutex.unlock();
    _Pragma("omp single")
    result = std::sqrt(result / count);
    _Pragma("omp barrier")
}

// ---------------------------------------------------------------------------------------------------------------------
// RMAE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate relative max error with CPU parallelism
void candy::rmae_cpu(const candy::Model * p_model, const array::Array * p_data, double & result, std::uint64_t & count,
                     std::uint64_t thread_idx, std::uint64_t n_threads, Index & index_mem) noexcept {
    // initialize result
    double thread_rmae = 0.0;
    std::uint64_t thread_count = 0;
    _Pragma("omp single") {
        result = 0.0;
        count = 0;
    }
    _Pragma("omp barrier")
    // summing on all points
    for (std::uint64_t i_point = thread_idx; i_point < p_data->size(); i_point += n_threads) {
        contiguous_to_ndim_idx(i_point, p_data->shape().data(), p_data->ndim(), index_mem.data());
        double x_data = p_data->get(index_mem);
        if (!is_finite(x_data)) {
            continue;
        }
        thread_count += 1;
        double x_model = p_model->eval(index_mem);
        double rel_err = std::abs(x_model - x_data) / x_data;
        thread_rmae = ((thread_rmae < rel_err) ? rel_err : thread_rmae);
    }
    _Pragma("omp barrier")
    // accumulate
    Environment::mutex.lock();
    count += thread_count;
    result = ((result < thread_rmae) ? thread_rmae : result);
    Environment::mutex.unlock();
}

}  // namespace merlin
