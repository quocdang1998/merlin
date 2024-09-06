// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include <cmath>  // std::abs, std::sqrt, std::isnormal

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/utils.hpp"        // merlin::contiguous_to_ndim_idx, merlin::is_normal, merlin::is_finite

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RMSE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean error with CPU
void candy::rmse_cpu(const candy::Model * p_model, const array::Array * p_data, double & result, std::uint64_t & count,
                     Index & index_mem) noexcept {
    // initialize result
    result = 0.0;
    count = 0;
    // summing on all points
    for (std::uint64_t i_point = 0; i_point < p_data->size(); i_point++) {
        contiguous_to_ndim_idx(i_point, p_data->shape().data(), p_data->ndim(), index_mem.data());
        double x_data = (*p_data)[index_mem];
        if (!is_normal(x_data)) {
            continue;
        }
        count += 1;
        double x_model = p_model->eval(index_mem);
        double rel_err = (x_model - x_data) / x_data;
        result += rel_err * rel_err;
    }
    // finalize
    result = std::sqrt(result / count);
}

// ---------------------------------------------------------------------------------------------------------------------
// RMAE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate relative max error with CPU
void candy::rmae_cpu(const candy::Model * p_model, const array::Array * p_data, double & result, std::uint64_t & count,
                     Index & index_mem) noexcept {
    // initialize result
    result = 0.0;
    count = 0;
    // summing on all points
    for (std::uint64_t i_point = 0; i_point < p_data->size(); i_point++) {
        contiguous_to_ndim_idx(i_point, p_data->shape().data(), p_data->ndim(), index_mem.data());
        double x_data = p_data->get(index_mem);
        if (!is_finite(x_data)) {
            continue;
        }
        count += 1;
        double x_model = p_model->eval(index_mem);
        double rel_err = std::abs(x_model - x_data) / x_data;
        result = ((result < rel_err) ? rel_err : result);
    }
}

}  // namespace merlin
