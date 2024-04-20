// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include <algorithm>  // std::copy_n
#include <future>     // std::async, std::future

#include <omp.h>  // ::omp_get_thread_num

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // merlin::Fatal, merlin::cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Evaluate regression by CPU parallelism
void regpl::eval_by_cpu(std::future<void> && synch, const regpl::Polynomial * p_poly, const double * point_data,
                        double * p_result, std::uint64_t n_points, std::uint64_t n_threads) noexcept {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // evaluate polynomial
    _Pragma("omp parallel num_threads(n_threads)") {
        // initialize buffer
        std::uint64_t thread_idx = ::omp_get_thread_num();
        Point thread_buffer;
        thread_buffer.fill(0.0);
        Point coordinates;
        coordinates.fill(0.0);
        std::uint64_t ndim = p_poly->ndim();
        // calculate evaluation for each point
        for (std::uint64_t i_point = thread_idx; i_point < n_points; i_point += n_threads) {
            const double * point = point_data + i_point * p_poly->ndim();
            std::copy_n(point, ndim, coordinates.begin());
            p_result[i_point] = p_poly->eval(coordinates, thread_buffer);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------------------------------------------------

// Evaluate regression by CPU parallelism
void regpl::Regressor::evaluate(const array::Array & points, DoubleVec & result, std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (this->on_gpu()) {
        Fatal<std::invalid_argument>("Interpolator is initialized on GPU.\n");
    }
    // check points array
    if (points.ndim() != 2) {
        Fatal<std::invalid_argument>("Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        Fatal<std::invalid_argument>("Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->poly_.ndim()) {
        Fatal<std::invalid_argument>("Array of coordinates and regressor have different dimension.\n");
    }
    if (points.shape()[0] != result.size()) {
        Fatal<std::invalid_argument>("Result array have incorrect size.\n");
    }
    // asynchronous calculate
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, regpl::eval_by_cpu, std::move(current_sync),
                                            &(this->poly_), points.data(), result.data(), points.shape()[0], n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

#ifndef __MERLIN_CUDA__

// Evaluate regression by GPU parallelism
void regpl::Regressor::evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads) {
    Fatal<cuda_compile_error>("Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
