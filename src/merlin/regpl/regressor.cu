// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include "merlin/array/parcel.hpp"       // merlin::array::Parcel
#include "merlin/cuda/copy_helpers.hpp"  // merlin::cuda::Dispatcher
#include "merlin/cuda/device.hpp"        // merlin::cuda::CtxGuard
#include "merlin/logger.hpp"             // merlin::Fatal

namespace merlin {

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
    // asynchronous calculate
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    double * result_gpu;
    ::cudaMallocAsync(&result_gpu, num_points * sizeof(double), cuda_stream);
    cuda::Dispatcher mem(stream.get_stream_ptr(), this->poly_);
    regpl::eval_by_gpu(mem.get<0>(), points.data(), result_gpu, num_points, n_threads, this->poly_.sharedmem_size(),
                       stream);
    ::cudaMemcpyAsync(result.data(), result_gpu, num_points * sizeof(double), ::cudaMemcpyDeviceToHost, cuda_stream);
    ::cudaFreeAsync(result_gpu, cuda_stream);
}

}  // namespace merlin
