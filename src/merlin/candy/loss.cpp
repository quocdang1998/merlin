// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include <array>  // std::array
#include <omp.h>  // pragma omp

#include <cinttypes>

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::contiguous_to_model_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Loss function
// --------------------------------------------------------------------------------------------------------------------

// Calculate loss function with CPU parallelism
double candy::calc_loss_function_cpu(const candy::Model & model, const array::Array & train_data) {
    floatvec loss_vector(::omp_get_max_threads(), 0.0);
    #pragma omp parallel for
    for (std::int64_t i_point = 0; i_point < train_data.size(); i_point++) {
        intvec index = contiguous_to_ndim_idx(i_point, train_data.shape());
        double data_point = train_data.get(index);
        double error = (data_point == 0) ? 0.0 : (model.eval(index) / data_point - 1.f);
        error *= error;
        loss_vector[i_point % loss_vector.size()] += error;
    }
    double result = 0.0;
    for (std::uint64_t i_thread = 0; i_thread < loss_vector.size(); i_thread++) {
        result += loss_vector[i_thread];
    }
    return result;
}

#ifndef __MERLIN_CUDA__

// Calculate loss function with GPU parallelism
void candy::calc_loss_function_gpu(const candy::Model & model, const array::Parcel & train_data, floatvec & result,
                                   const cuda::Stream & stream, std::uint64_t n_thread) {
    FAILURE(cuda_compile_error, "Compile the package with CUDA option enabled to access this feature.\n");
}

#endif  // __MERLIN_CUDA__

// --------------------------------------------------------------------------------------------------------------------
// Model gradient
// --------------------------------------------------------------------------------------------------------------------

// Calculate gradient of canonical decomposition model with CPU parallelism
void candy::calc_gradient_vector_cpu(const candy::Model & model, const array::Array & train_data, floatvec & result,
                                     std::uint64_t n_thread) {
    // check shape
    intvec model_shape = model.get_model_shape();
    const intvec & data_shape = train_data.shape();
    if (model_shape.size() != data_shape.size()) {
        FAILURE(std::invalid_argument, "Model and data must have the same number of dimension.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < model_shape.size(); i_dim++) {
        if (data_shape[i_dim] * model.rank() != model_shape[i_dim]) {
            FAILURE(std::invalid_argument, "Shape of model and data must be the same.\n");
        }
    }
    // check size of vector
    std::uint64_t n_param = model.size();
    if (result.size() != n_param) {
        FAILURE(std::invalid_argument, "Result vector must have the size of the number of parameters in the model.\n");
    }
    // initialize result
    for (std::uint64_t i_param = 0; i_param < n_param; i_param++) {
        result[i_param] = 0.0;
    }
    // update each parameter
    std::uint64_t n_point = train_data.size(), n_dim = model_shape.size();
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < n_param; i_param++) {
        auto [param_dim, param_index, param_rank] = contiguous_to_model_idx(i_param, model.rank(), model_shape);
        // loop over each point in the dataset
        std::uint64_t n_subset = n_point / data_shape[param_dim];
        for (std::uint64_t i_point = 0; i_point < n_subset; i_point++) {
            intvec index_data = candy::contiguous_to_ndim_idx_1(i_point, data_shape, param_dim);
            index_data[param_dim] = param_index;
            double data = train_data.get(index_data);
            if (data == 0) {
                continue;
            }
            double gradient = 1.0;
            // divide by 1/data^2
            gradient /= data*data;
            // multiply by coefficient of the same rank from other dimension
            for (std::uint64_t i_dim = 0; i_dim < n_dim; i_dim++) {
                if (i_dim == param_dim) {
                    continue;
                }
                gradient *= model.get(i_dim, index_data[i_dim], param_rank);
            }
            // multiply by value evaluation
            double eval = model.eval(index_data);
            gradient *= eval - data;
            // add gradient to gradient vector
            result[i_param] += gradient;
        }
    }
}

}  // namespace merlin
