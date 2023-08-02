// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/utils.hpp"        // merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Loss function
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean error with CPU parallelism
template <typename ErrorUpdater, typename Averager>
static double calc_error_cpu(const candy::Model & model, const array::Array & train_data, std::uint64_t n_threads,
                             ErrorUpdater err_updater, Averager averager) noexcept {
    // create an array for storing result from each thread
    floatvec thread_result(n_threads, 0.0);
    intvec non_zero_element(n_threads, 0);
    // get error on each non-zero element
    std::uint64_t size = train_data.size();
    intvec temporay_index(train_data.ndim());
    std::uint64_t i_thread;
    #pragma omp parallel for num_threads(n_threads) firstprivate(temporay_index, i_thread)
    for (std::int64_t i_point = 0; i_point < size; i_point++) {
        i_thread = ::omp_get_thread_num();
        contiguous_to_ndim_idx(i_point, train_data.shape(), temporay_index.data());
        double x_data = train_data[temporay_index];
        if (x_data == 0.0) {
            continue;
        }
        non_zero_element[i_thread] += 1;
        double x_model = model.eval(temporay_index);
        double temp = (x_model - x_data) / x_data;
        err_updater(thread_result[i_thread], x_model, x_data);
    }
    return averager(thread_result.data(), non_zero_element.data(), n_threads);
}

// ---------------------------------------------------------------------------------------------------------------------
// RMSE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean error with CPU parallelism
double candy::rmse_cpu(const candy::Model & model, const array::Array & train_data,
                            std::uint64_t n_threads) noexcept {
    return calc_error_cpu(model, train_data, n_threads, candy::rmse_updater, candy::rmse_averager);
}

// ---------------------------------------------------------------------------------------------------------------------
// RMAE
// ---------------------------------------------------------------------------------------------------------------------

// Calculate relative max error with CPU parallelism
double candy::rmae_cpu(const candy::Model & model, const array::Array & train_data,
                            std::uint64_t n_threads) noexcept {
    return calc_error_cpu(model, train_data, n_threads, candy::rmae_updater, candy::rmae_averager);
}

}  // namespace merlin
