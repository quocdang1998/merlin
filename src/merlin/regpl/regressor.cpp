// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include <future>   // std::async, std::shared_future

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/regpl/polynomial.hpp"  // merlin::regpl::Polynomial

namespace merlin {

#define push_gpu(gpu)                                                                                                  \
    Environment::mutex.lock();                                                                                         \
    std::uintptr_t current_ctx = gpu.push_context()
#define pop_gpu()                                                                                                      \
    cuda::Device::pop_context(current_ctx);                                                                            \
    Environment::mutex.unlock()

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

// ---------------------------------------------------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from polynomial object
regpl::Regressor::Regressor(const regpl::Polynomial & polynom, ProcessorType proc_type) :
num_coeff_(polynom.size()), ndim_(polynom.ndim()) {
    if (proc_type == ProcessorType::Cpu) {
        this->p_poly_ = new regpl::Polynomial(polynom);
        this->matrix_data_ = new double[num_coeff_ * num_coeff_];
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
