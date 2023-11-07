// Copyright 2022 quocdang1998
#include "merlin/splint/tools.hpp"

#include <array>  // std::array

#include <omp.h>  // #pragma omp

#include "merlin/array/operation.hpp"        // merlin::array::contiguous_strides
#include "merlin/cuda/stream.hpp"            // merlin::cuda::Stream
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/intpl/linear.hpp"    // merlin::splint::intpl::construct_linear
#include "merlin/splint/intpl/lagrange.hpp"  // merlin::splint::intpl::construct_lagrange
#include "merlin/splint/intpl/newton.hpp"    // merlin::splint::intpl::construction_newton
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/thread_divider.hpp"         // merlin::ThreadDivider
#include "merlin/utils.hpp"                  // merlin::prod_elements, merlin::increment_index

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Construct coefficients
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients with CPU parallelism
void splint::construct_coeff_cpu(double * coeff, const splint::CartesianGrid & grid,
                                 const Vector<splint::Method> & method, std::uint64_t n_threads) noexcept {

    // functor to coefficient construction methods
    static const std::array<splint::ConstructionMethod, 3> construction_funcs {
        splint::intpl::construct_linear,
        splint::intpl::construct_lagrange,
        splint::intpl::construction_newton
    };
    // initialization
    const intvec & shape = grid.shape();
    std::uint64_t num_subsystem = 1, element_size = prod_elements(shape);
    // solve matrix for each dimension
    // std::uint64_t subsystem_size = 0, numthreads_subsystem = 0, num_groups = 0;
    std::uint64_t subsystem_size = 0;
    unsigned int i_method = 0;
    #pragma omp parallel num_threads(n_threads)
    {
        int thread_idx = ::omp_get_thread_num();
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            // calculate number of thread per groups
            #pragma omp single
            {
                subsystem_size = element_size;
                element_size /= shape[i_dim];
                i_method = static_cast<unsigned int>(method[i_dim]);
            }
            #pragma omp barrier
            // parallel subsystem over the number of groups
            ThreadDivider thr_grp(num_subsystem, thread_idx, n_threads);
            for (std::uint64_t i_task = thr_grp.group_idx; i_task < num_subsystem; i_task += thr_grp.num_groups) {
                double * subsystem_start = coeff + i_task * subsystem_size;
                construction_funcs[i_method](subsystem_start, grid.grid_vectors()[i_dim], shape[i_dim], element_size,
                                             thr_grp.thread_idx_in_group, thr_grp.numthreads_pertask);
            }
            #pragma omp barrier
            // update number of sub-system
            #pragma omp single
            {
                num_subsystem *= shape[i_dim];
            }
            #pragma omp barrier
        }
    }
}

#ifndef __MERLIN_CUDA__

// Construct interpolation coefficients with GPU parallelism
void splint::construct_coeff_gpu(double * coeff, const splint::CartesianGrid * p_grid,
                                 const Vector<splint::Method> * p_method, std::uint64_t n_threads,
                                 std::uint64_t shared_mem_size, const cuda::Stream * stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "The library is not compiled with CUDA.\n");
}

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Evaluate interpolation
// ---------------------------------------------------------------------------------------------------------------------

// Evaluate interpolation with CPU parallelism
void splint::eval_intpl_cpu(const double * coeff, const splint::CartesianGrid & grid,
                            const Vector<splint::Method> & method, const double * points, std::uint64_t n_points,
                            double * result, std::uint64_t n_threads) noexcept {
    // initialize index vector and cache memory for each thread
    intvec total_index(grid.ndim() * n_threads, 0);
    floatvec cache_memory(grid.ndim() * n_threads);
    // parallel calculation
    #pragma omp parallel num_threads(n_threads)
    {
        // get corresponding index vector and cache memory
        int thread_idx = ::omp_get_thread_num();
        intvec loop_index;
        loop_index.assign(total_index.data() + grid.ndim() * thread_idx, grid.ndim());
        floatvec cache;
        cache.assign(cache_memory.data() + grid.ndim() * thread_idx, grid.ndim());
        // parallel calculation for each point
        for (std::uint64_t i_point = thread_idx; i_point < n_points; i_point += n_threads) {
            const double * point_data = points + i_point * grid.ndim();
            std::int64_t last_updated_dim = grid.ndim()-1;
            std::uint64_t contiguous_index = 0;
            // loop on each index and save evaluation by each coefficient to the cache array
            do {
                splint::recursive_interpolate(coeff, grid.size(), contiguous_index, loop_index.data(), cache.data(),
                                              point_data, last_updated_dim, grid.shape().data(),
                                              grid.grid_vectors().data(), method, grid.ndim());
                last_updated_dim = increment_index(loop_index, grid.shape());
                contiguous_index++;
            } while (last_updated_dim != -1);
            // perform one last iteration on the last coefficient
            splint::recursive_interpolate(coeff, grid.size(), contiguous_index, loop_index.data(), cache.data(),
                                          point_data, 0, grid.shape().data(), grid.grid_vectors().data(), method,
                                          grid.ndim());
            // save result and reset the cache
            result[i_point] = cache[0];
            cache[0] = 0.0;
        }
    }
}

#ifndef __MERLIN_CUDA__

// Evaluate interpolation with GPU parallelism
void splint::eval_intpl_gpu(double * coeff, const splint::CartesianGrid * p_grid,
                            const Vector<splint::Method> * p_method, double * points, std::uint64_t n_points,
                            double * result, std::uint64_t n_threads, std::uint64_t ndim, std::uint64_t shared_mem_size,
                            cuda::Stream * stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "The library is not compiled with CUDA.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
