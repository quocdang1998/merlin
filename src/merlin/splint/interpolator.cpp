// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"            // merlin::array::Array
#include "merlin/cuda_interface.hpp"         // merlin::cuda_mem_free
#include "merlin/cuda/device.hpp"            // merlin::cuda::Device
#include "merlin/env.hpp"                    // merlin::Environment
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/tools.hpp"           // merlin::splint::construct_coeff_cpu

#define safety_lock() bool lock_success = Environment::mutex.try_lock()
#define safety_unlock()                                                                                                \
    if (lock_success) Environment::mutex.unlock()

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolator
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolator::Interpolator(const splint::CartesianGrid & grid, array::Array & values,
                                   const Vector<splint::Method> & method, std::uint64_t n_threads) :
p_coeff_(&values), ndim_(grid.ndim()) {
    // check shape
    if (grid.shape() != values.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    if (grid.ndim() != method.size()) {
        FAILURE(std::invalid_argument, "Grid and method vector have different shape.\n");
    }
    // check if data array is C-contiguous
    if (!values.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Data must be C-contiguous.\n");
    }
    // copy grid
    this->p_grid_ = new splint::CartesianGrid(grid);
    this->p_method_ = new Vector<splint::Method>(method);
    // calculate coefficients directly on the data
    splint::construct_coeff_cpu(values.data(), grid, method, n_threads);
}

// Interpolation by CPU.
floatvec splint::Interpolator::interpolate(const array::Array & points, std::uint64_t n_threads) {
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
        FAILURE(std::invalid_argument, "Array of coordinates and interpolator have different dimension.\n");
    }
    // evaluate interpolation
    floatvec evaluated_values(points.shape()[0]);
    splint::eval_intpl_cpu(this->p_coeff_->data(), *(this->p_grid_), *(this->p_method_), points.data(),
                           evaluated_values.size(), evaluated_values.data(), n_threads);
    return evaluated_values;
}

// Destructor
splint::Interpolator::~Interpolator(void) {
    if (!(this->on_gpu())) {
        // delete pointer to grid and method if the interpolator are on CPU
        if (this->p_grid_ != nullptr) {
            delete this->p_grid_;
        }
        if (this->p_method_ != nullptr) {
            delete this->p_method_;
        }
    } else {
        // delete pointer to everything on GPU
        safety_lock();
        cuda::Device gpu(this->gpu_id_);
        gpu.set_as_current();
        if (this->p_grid_ != nullptr) {
            cuda_mem_free(this->p_grid_, this->allocation_stream_);
        }
        safety_unlock();
    }
}

}  // namespace merlin
