// Copyright 2022 quocdang1998
#include "merlin/splint/tools.hpp"

#include <array>  // std::array

#include <omp.h>  // #pragma omp

#include "merlin/cuda/stream.hpp"            // merlin::cuda::Stream
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/intpl/lagrange.hpp"  // merlin::splint::intpl::construct_lagrange
#include "merlin/splint/intpl/linear.hpp"    // merlin::splint::intpl::construct_linear
#include "merlin/splint/intpl/newton.hpp"    // merlin::splint::intpl::construction_newton
#include "merlin/thread_divider.hpp"         // merlin::ThreadDivider
#include "merlin/utils.hpp"                  // merlin::increment_index

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Construct coefficients
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients with CPU parallelism
void splint::construct_coeff_cpu(std::future<void> && current_job, double * coeff, const grid::CartesianGrid * p_grid,
                                 const std::array<unsigned int, max_dim> * p_method, std::uint64_t n_threads) noexcept {
    // functor to coefficient construction methods
    static const std::array<splint::ConstructionMethod, 3> construction_funcs{
        splint::intpl::construct_linear,
        splint::intpl::construct_lagrange,
        splint::intpl::construction_newton,
    };
    // finish old job
    if (current_job.valid()) {
        current_job.get();
    }
    // parallel calcualte coefficients
    _Pragma("omp parallel num_threads(n_threads)") {
        // initialization
        const Index & shape = p_grid->shape();
        std::uint64_t num_subsystem = 1, element_size = p_grid->size();
        std::uint64_t subsystem_size = 0;
        int thread_idx = ::omp_get_thread_num();
        // loop on each dimension
        for (std::uint64_t i_dim = 0; i_dim < p_grid->ndim(); i_dim++) {
            // calculate number of thread per groups
            subsystem_size = element_size;
            element_size /= shape[i_dim];
            const unsigned int & i_method = (*p_method)[i_dim];
            // parallel subsystem over the number of groups
            ThreadDivider thr_grp(num_subsystem, thread_idx, n_threads);
            for (std::uint64_t i_task = thr_grp.group_idx; i_task < num_subsystem; i_task += thr_grp.num_groups) {
                double * subsystem_start = coeff + i_task * subsystem_size;
                construction_funcs[i_method](subsystem_start, p_grid->grid_vectors()[i_dim], shape[i_dim], element_size,
                                             thr_grp.thread_idx_in_group, thr_grp.numthreads_pertask);
            }
            num_subsystem *= shape[i_dim];
            // force a barrier before moving on to the next dimension
            _Pragma("omp barrier");
        }
    }
}

#ifndef __MERLIN_CUDA__

// Construct interpolation coefficients with GPU parallelism
void splint::construct_coeff_gpu(double * coeff, const grid::CartesianGrid * p_grid,
                                 const std::array<unsigned int, max_dim> * p_method, std::uint64_t n_threads,
                                 std::uint64_t shared_mem_size, const cuda::Stream * stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "The library is not compiled with CUDA.\n");
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// ---------------------------------------------------------------------------------------------------------------------

// Evaluate interpolation with CPU parallelism
void splint::eval_intpl_cpu(std::future<void> && current_job, const double * coeff, const grid::CartesianGrid * p_grid,
                            const std::array<unsigned int, max_dim> * p_method, const double * points,
                            std::uint64_t n_points, double * result, std::uint64_t n_threads) noexcept {
    // finish old job
    if (current_job.valid()) {
        current_job.get();
    }
    // parallel interpolation evaluation
    _Pragma("omp parallel num_threads(n_threads)") {
        // initialization
        Index loop_index;
        loop_index.fill(0);
        Point cache;
        cache.fill(0.0);
        std::uint64_t thread_idx = ::omp_get_thread_num();
        // parallel calculation for each point
        for (std::uint64_t i_point = thread_idx; i_point < n_points; i_point += n_threads) {
            const double * point_data = points + i_point * p_grid->ndim();
            std::int64_t last_updated_dim = p_grid->ndim() - 1;
            std::uint64_t contiguous_index = 0;
            // loop on each index and save evaluation by each coefficient to the cache array
            do {
                splint::recursive_interpolate(coeff, p_grid->size(), contiguous_index, loop_index.data(), cache.data(),
                                              point_data, last_updated_dim, p_grid->shape().data(),
                                              p_grid->grid_vectors().data(), *p_method, p_grid->ndim());
                last_updated_dim = increment_index(loop_index.data(), p_grid->shape().data(), p_grid->ndim());
                contiguous_index++;
            } while (last_updated_dim != -1);
            // perform one last iteration on the last coefficient
            splint::recursive_interpolate(coeff, p_grid->size(), contiguous_index, loop_index.data(), cache.data(),
                                          point_data, 0, p_grid->shape().data(), p_grid->grid_vectors().data(),
                                          *p_method, p_grid->ndim());
            // save result and reset the cache
            result[i_point] = cache[0];
            cache[0] = 0.0;
        }
    }
}

#ifndef __MERLIN_CUDA__

// Evaluate interpolation with GPU parallelism
void splint::eval_intpl_gpu(double * coeff, const grid::CartesianGrid * p_grid,
                            const std::array<unsigned int, max_dim> * p_method, double * points, std::uint64_t n_points,
                            double * result, std::uint64_t n_threads, std::uint64_t ndim, std::uint64_t shared_mem_size,
                            const cuda::Stream * stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "The library is not compiled with CUDA.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
