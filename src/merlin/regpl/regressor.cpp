// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include <cstring>  // std::memset
#include <future>   // std::async, std::shared_future

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/grid/regular_grid.hpp"  // merlin::grid::RegularGrid
#include "merlin/linalg/matrix.hpp"  // merlin::linalg::Matrix
#include "merlin/linalg/qr_solve.hpp"  // merlin::linalg::qr_solve_cpu
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/regpl/core.hpp"  // merlin::regpl::calc_vector, merlin::regpl::calc_system
#include "merlin/regpl/polynomial.hpp"  // merlin::regpl::Polynomial

namespace merlin {

#define push_gpu(gpu)                                                                                                  \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    cuda::Device::pop_context(current_ctx)

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Allocate memory for regressor object on GPU
void regpl::allocate_mem_gpu(const regpl::Polynomial & polynom, regpl::Polynomial *& p_poly, double *& matrix_data,
                             std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Cannot invoke GPU function since merlin is not compiled with CUDA option.\n");
}

#endif  // __MERLIN_CUDA__

// Calculate coefficient for regression on Cartesian grid
void regpl::fit_cartgrid_by_cpu(std::shared_future<void> synch, regpl::Polynomial * p_poly, double * matrix_data,
                                const grid::CartesianGrid * p_grid, const array::Array * p_data,
                                std::uint64_t n_threads, char * cpu_buffer) noexcept {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // solve regression matrix
    linalg::Matrix linear_system(matrix_data, {p_poly->size(), p_poly->size()});
    linalg::Matrix linear_vector(p_poly->coeff().data(), {p_poly->size(), 1}, {sizeof(double), 0});
    std::uint64_t * buffer_create = reinterpret_cast<std::uint64_t *>(cpu_buffer);
    double * buffer_solve = reinterpret_cast<double *>(cpu_buffer);
    double norm;
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        regpl::calc_vector(*p_grid, *p_data, *p_poly, buffer_create, thread_idx, n_threads);
        #pragma omp barrier
        regpl::calc_system(*p_grid, *p_poly, linear_system, buffer_create, thread_idx, n_threads);
        #pragma omp barrier
        linalg::qr_solve_cpu(linear_system, linear_vector, buffer_solve, norm, thread_idx, n_threads);
    }
}

// Calculate coefficient for regression on regular grid
void regpl::fit_reggrid_by_cpu(std::shared_future<void> synch, regpl::Polynomial * p_poly, double * matrix_data,
                               const grid::RegularGrid * p_grid, const floatvec * p_data, std::uint64_t n_threads,
                               char * cpu_buffer) noexcept {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // solve regression matrix
    linalg::Matrix linear_system(matrix_data, {p_poly->size(), p_poly->size()});
    linalg::Matrix linear_vector(p_poly->coeff().data(), {p_poly->size(), 1}, {sizeof(double), 0});
    std::uint64_t * buffer_create = reinterpret_cast<std::uint64_t *>(cpu_buffer);
    double * buffer_solve = reinterpret_cast<double *>(cpu_buffer);
    double norm;
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        regpl::calc_vector(*p_grid, *p_data, *p_poly, buffer_create, thread_idx, n_threads);
        #pragma omp barrier
        regpl::calc_system(*p_grid, *p_poly, linear_system, buffer_create, thread_idx, n_threads);
        #pragma omp barrier
        linalg::qr_solve_cpu(linear_system, linear_vector, buffer_solve, norm, thread_idx, n_threads);
    }
}

// Evaluate regression
void regpl::eval_by_cpu(std::shared_future<void> synch, const regpl::Polynomial * p_poly, const array::Array * p_data,
                        double * p_result, std::uint64_t n_threads, char * cpu_buffer) noexcept {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // evaluate polynomial
    double * buffer = reinterpret_cast<double *>(cpu_buffer);
    #pragma omp parallel num_threads(n_threads)
    {
        // assign buffer to thread
        std::uint64_t thread_idx = ::omp_get_thread_num();
        double * thread_buffer = buffer + thread_idx * p_poly->ndim();
        // calculate evaluation for each point
        for (std::uint64_t i_point = thread_idx; i_point < p_data->shape()[0]; i_point += n_threads) {
            const double * point_data = p_data->data() + i_point * p_poly->ndim();
            p_result[i_point] = p_poly->eval(point_data, thread_buffer);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from polynomial object
regpl::Regressor::Regressor(const regpl::Polynomial & polynom, ProcessorType proc_type) :
num_coeff_(polynom.size()), ndim_(polynom.ndim()) {
    if (proc_type == ProcessorType::Cpu) {
        this->p_poly_ = new regpl::Polynomial(polynom);
        this->matrix_data_ = new double[this->num_coeff_ * this->num_coeff_];
        std::memset(this->matrix_data_, 0, sizeof(double) * this->num_coeff_ * this->num_coeff_);
        this->cpu_buffer_size_ = sizeof(double) * num_coeff_;
        this->cpu_buffer_ = new char[this->cpu_buffer_size_];
        this->synch_ = Synchronizer(std::shared_future<void>());
    } else {
        this->shared_mem_size_ = polynom.sharedmem_size();
        this->synch_ = Synchronizer(cuda::Stream(cuda::StreamSetting::NonBlocking));
        cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
        regpl::allocate_mem_gpu(polynom, this->p_poly_, this->matrix_data_, stream.get_stream_ptr());
    }
}

// Resize CPU buffer
void regpl::Regressor::resize_cpu_buffer(std::uint64_t new_size) {
    if (this->cpu_buffer_size_ < new_size) {
        delete[] this->cpu_buffer_;
        this->cpu_buffer_ = new char[new_size];
        this->cpu_buffer_size_ = new_size;
    }
}

// Get a copy of the polynomial
regpl::Polynomial regpl::Regressor::get_polynom(void) const {
    return regpl::Polynomial(*(this->p_poly_));
}

// Regression on a cartesian dataset using CPU parallelism
void regpl::Regressor::fit_cpu(const grid::CartesianGrid & grid, const array::Array & data, std::uint64_t n_threads) {
    // error when object is initialized on GPU
    if (this->on_gpu()) {
        FAILURE(std::invalid_argument, "Regressor is initialized on GPU.\n");
    }
    // check data shape
    if (grid.ndim() != this->ndim_) {
        FAILURE(std::invalid_argument, "Grid must have the same ndim as the input polynomial.\n");
    }
    if (data.ndim() != this->ndim_) {
        FAILURE(std::invalid_argument, "Data must have the same ndim as the input polynomial.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        if (grid.shape()[i_dim] != data.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Grid and data have inconsistent shape.\n");
        }
    }
    // resize buffer
    this->resize_cpu_buffer(2 * n_threads * this->ndim_ * sizeof(std::uint64_t));
    // asynchronous calculate
    std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synch_.synchronizer);
    std::shared_future<void> new_sync = std::async(std::launch::async, regpl::fit_cartgrid_by_cpu, current_sync,
                                                   this->p_poly_, this->matrix_data_, &grid, &data, n_threads,
                                                   this->cpu_buffer_).share();
    this->synch_ = Synchronizer(std::move(new_sync));
}

// Regression on a random dataset using CPU parallelism.
void regpl::Regressor::fit_cpu(const grid::RegularGrid & grid, const floatvec & data, std::uint64_t n_threads) {
    // error when object is initialized on GPU
    if (this->on_gpu()) {
        FAILURE(std::invalid_argument, "Regressor is initialized on GPU.\n");
    }
    // check data shape
    if (grid.ndim() != this->ndim_) {
        FAILURE(std::invalid_argument, "Grid must have the same ndim as polynom.\n");
    }
    if (data.size() != grid.size()) {
        FAILURE(std::invalid_argument, "Grid and data must have the same number of points.\n");
    }
    // resize buffer
    this->resize_cpu_buffer(2 * n_threads * this->ndim_ * sizeof(std::uint64_t));
    // asynchronous calculate
    std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synch_.synchronizer);
    std::shared_future<void> new_sync = std::async(std::launch::async, regpl::fit_reggrid_by_cpu, current_sync,
                                                   this->p_poly_, this->matrix_data_, &grid, &data, n_threads,
                                                   this->cpu_buffer_).share();
    this->synch_ = Synchronizer(std::move(new_sync));
}

// Evaluate regression by CPU parallelism
void regpl::Regressor::evaluate(const array::Array & points, double * p_result, std::uint64_t n_threads) {
    // check if interpolator is on CPU
    if (this->on_gpu()) {
        FAILURE(std::invalid_argument, "Interpolator is initialized on GPU.\n");
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
    // resize buffer
    this->resize_cpu_buffer(n_threads * this->ndim_ * sizeof(double));
    // asynchronous calculate
    std::memset(this->cpu_buffer_, 0, this->cpu_buffer_size_);
    std::shared_future<void> & current_sync = std::get<std::shared_future<void>>(this->synch_.synchronizer);
    std::shared_future<void> new_sync = std::async(std::launch::async, regpl::eval_by_cpu, current_sync, this->p_poly_,
                                                   &points, p_result, n_threads, this->cpu_buffer_).share();
    this->synch_ = Synchronizer(std::move(new_sync));
}

// Default destructor
regpl::Regressor::~Regressor(void) {
    if (!(this->on_gpu())) {
        if (this->p_poly_ != nullptr) {
            delete this->p_poly_;
        }
        if (this->matrix_data_ != nullptr) {
            delete[] this->matrix_data_;
        }
        if (this->cpu_buffer_ != nullptr) {
            delete[] this->cpu_buffer_;
        }
    } else {
        push_gpu(cuda::Device(this->gpu_id()));
        cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
        if (this->p_poly_ != nullptr) {
            cuda_mem_free(this->p_poly_, stream.get_stream_ptr());
        }
        if (this->matrix_data_ != nullptr) {
            cuda_mem_free(this->matrix_data_, stream.get_stream_ptr());
        }
        pop_gpu();
    }
}

}  // namespace merlin
