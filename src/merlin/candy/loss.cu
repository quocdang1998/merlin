// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_model_idx, merlin::flatten_thread_index, merlin::size_of_block

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Model gradient
// --------------------------------------------------------------------------------------------------------------------

// Calculate gradient of a model over dataset on GPU
__cudevice__ void candy::calc_gradient_vector_gpu(const candy::Model * p_model,
                                                  const array::Parcel * p_train_data,
                                                  std::uint64_t * cache_memory, double * gradient_vector) {
    // calculate model shape
    std::uint64_t n_threads = size_of_block(), thread_idx = flatten_thread_index();
    std::uint64_t n_param = p_model->size(), n_point = p_train_data->size(), n_dim = p_train_data->ndim();
    intvec model_shape;
    model_shape.assign(cache_memory, n_dim);
    for (std::uint64_t i_dim = thread_idx; i_dim < n_dim; i_dim += n_threads) {
        model_shape[i_dim] = p_model->parameters()[i_dim].size();
    }
    __syncthreads();
    cache_memory += n_dim;
    // loop for each parameter
    const intvec & data_shape = p_train_data->shape();
    for (std::uint64_t i_param = thread_idx; i_param < n_param; i_param += n_threads) {
        gradient_vector[i_param] = 0.0;
        auto [param_dim, param_index, param_rank] = contiguous_to_model_idx(i_param, p_model->rank(), model_shape);
        // loop over each point in the dataset
        std::uint64_t n_subset = n_point / data_shape[param_dim];
        for (std::uint64_t i_point = 0; i_point < n_subset; i_point++) {
            intvec index_data = candy::contiguous_to_ndim_idx_1(i_point, data_shape, param_dim,
                                                                cache_memory + thread_idx*n_dim);
            index_data[param_dim] = param_index;
            const double & data = (*p_train_data)[index_data];
            double gradient = 1.0;
            // divide by 1/data^2
            gradient /= data*data;
            // multiply by coefficient of the same rank from other dimension
            for (std::uint64_t i_dim = 0; i_dim < n_dim; i_dim++) {
                if (i_dim == param_dim) {
                    continue;
                }
                gradient *= p_model->get(i_dim, index_data[i_dim], param_rank);
            }
            // multiply by value evaluation
            double eval = p_model->eval(index_data);
            gradient *= eval - data;
            // add gradient to gradient vector
            gradient_vector[i_param] += gradient;
        }
    }
    __syncthreads();

}

}  // namespace merlin
