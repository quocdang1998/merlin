// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include <cmath>  // std::abs, std::sqrt

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/logger.hpp"       // FAILURE
#include "merlin/utils.hpp"        // merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RMSE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean error with CPU parallelism
double candy::rmse_cpu(const candy::Model * p_model, const array::Array * p_data, std::uint64_t * buffer,
                       std::uint64_t n_threads) noexcept {
    // initialize result
    std::uint64_t non_zero_element = 0;
    double rmse = 0.0;
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t i_thread = ::omp_get_thread_num();
        // initialize thread result
        std::uint64_t non_zero = 0;
        double thread_rmse = 0.0;
        // assign index vector
        intvec index;
        index.assign(buffer + i_thread * p_data->ndim(), p_data->ndim());
        // summing on all points
        for (std::uint64_t i_point = i_thread; i_point < p_data->size(); i_point += n_threads) {
            contiguous_to_ndim_idx(i_point, p_data->shape(), index.data());
            double x_data = (*p_data)[index];
            if (x_data == 0.0) {
                continue;
            }
            non_zero += 1;
            double x_model = p_model->eval(index);
            double rel_err = (x_model - x_data) / x_data;
            thread_rmse += rel_err * rel_err;
        }
        #pragma omp atomic update
        non_zero_element += non_zero;
        #pragma omp atomic update
        rmse += thread_rmse;
        #pragma omp barrier
    }
    return std::sqrt(rmse / non_zero_element);
}

// ---------------------------------------------------------------------------------------------------------------------
// RMAE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate relative max error with CPU parallelism
double candy::rmae_cpu(const candy::Model * p_model, const array::Array * p_data, std::uint64_t * buffer,
                       std::uint64_t n_threads) noexcept {
    // initialize result
    double rmae = 0.0;
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t i_thread = ::omp_get_thread_num();
        // initialize thread result
        double thread_rmae = 0.0;
        // assign index vector
        intvec index;
        index.assign(buffer + i_thread * p_data->ndim(), p_data->ndim());
        // summing on all points
        for (std::uint64_t i_point = i_thread; i_point < p_data->size(); i_point += n_threads) {
            contiguous_to_ndim_idx(i_point, p_data->shape(), index.data());
            double x_data = p_data->get(index);
            if (x_data == 0.0) {
                continue;
            }
            double x_model = p_model->eval(index);
            double rel_err = std::abs(x_model - x_data) / x_data;
            thread_rmae = ((thread_rmae < rel_err) ? rel_err : thread_rmae);
        }
        Environment::mutex.lock();
        rmae = ((rmae < thread_rmae) ? thread_rmae : rmae);
        Environment::mutex.unlock();
    }
    return rmae;
}

}  // namespace merlin
