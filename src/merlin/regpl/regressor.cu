// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/cuda_interface.hpp"  // merlin::cuda_mem_alloc
#include "merlin/regpl/polynomial.hpp"  // merlin::regpl::Polynomial

namespace merlin {

// Allocate memory for regressor object on GPU
void regpl::allocate_mem_gpu(const regpl::Polynomial & polynom, regpl::Polynomial *& p_poly, double *& matrix_data,
                             std::uintptr_t stream_ptr) {
    // copy polynomial object to GPU
    cuda::Memory gpu_mem(stream_ptr, polynom);
    p_poly = gpu_mem.get<0>();
    gpu_mem.disown();
    // allocate data on GPU
    matrix_data = reinterpret_cast<double *>(cuda_mem_alloc(polynom.size() * polynom.size(), stream_ptr));
}

// Evaluate regression by GPU parallelism
void regpl::Regressor::evaluate(const array::Parcel & points, double * p_result, std::uint64_t n_threads) {
    // check if interpolator is on GPU
    if (!this->on_gpu()) {
        FAILURE(std::invalid_argument, "Interpolator is initialized on CPU.\n");
    }
    // check points array
    if (points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of coordinates a 2D table.\n");
    }
    if (!points.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Expected array of coordinates to be C-contiguous.\n");
    }
    if (points.shape()[1] != this->ndim_) {
        FAILURE(std::invalid_argument, "Array of coordinates and regressor have different dimension.\n");
    }
    // asynchronous calculate
    cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
    double * result_gpu = (double *) cuda_mem_alloc(points.shape()[0] * sizeof(double), stream.get_stream_ptr());
    regpl::eval_by_gpu(this->p_poly_, points.data(), result_gpu, points.shape()[0], this->ndim_, this->shared_mem_size_,
                       n_threads, stream);
    cuda_mem_cpy_device_to_host(p_result, result_gpu, points.shape()[0] * sizeof(double), stream.get_stream_ptr());
    cuda_mem_free(result_gpu, stream.get_stream_ptr());
}

}  // namespace merlin
