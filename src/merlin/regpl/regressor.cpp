// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include <algorithm>  // std::copy_n
#include <future>     // std::async, std::future

#include <omp.h>  // ::omp_get_thread_num

#include "merlin/array/array.hpp"        // merlin::array::Array
#include "merlin/array/parcel.hpp"       // merlin::array::Parcel
#include "merlin/cuda/copy_helpers.hpp"  // merlin::cuda::Dispatcher
#include "merlin/cuda/device.hpp"        // merlin::cuda::CtxGuard
#include "merlin/logger.hpp"             // merlin::Fatal
#include "merlin/memory.hpp"             // merlin::mem_alloc_device, merlin::mem_free_device, merlin::memcpy_gpu_to_cpu

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
        std::uint64_t ndim = p_poly->ndim();
        Point thread_buffer(ndim);
        // calculate evaluation for each point
        for (std::uint64_t i_point = thread_idx; i_point < n_points; i_point += n_threads) {
            const double * point = point_data + i_point * ndim;
            Point coordinates(point, ndim);
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

// Evaluate regression by GPU parallelism
void regpl::Regressor::evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads) {
    // check if interpolator is on GPU
    if (!this->on_gpu()) {
        Fatal<std::invalid_argument>("Interpolator is initialized on CPU.\n");
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
    // initialize context
    std::uint64_t num_points = points.shape()[0];
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    std::uintptr_t stream_ptr = stream.get_stream_ptr();
    // asynchronous calculate
    double * result_gpu;
    Message("GPU pointer pre-allocated:") << result_gpu << "\n";
    mem_alloc_device(reinterpret_cast<void **>(&result_gpu), num_points * sizeof(double), stream_ptr);
    Message("GPU pointer post-allocated:") << result_gpu << "\n";
    cuda::Dispatcher mem(stream_ptr, this->poly_);
    regpl::eval_by_gpu(mem.get<0>(), points.data(), result_gpu, num_points, n_threads, this->poly_.sharedmem_size(),
                       stream);
    memcpy_gpu_to_cpu(result.data(), result_gpu, num_points * sizeof(double), stream_ptr);
    mem_free_device(result_gpu, stream_ptr);
}

}  // namespace merlin
