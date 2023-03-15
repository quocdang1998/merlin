// Copyright 2023 quocdang1998
#ifndef MERLIN_INTERPOLANT_NEWTON_HPP_
#define MERLIN_INTERPOLANT_NEWTON_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel, merlin::array::Slice
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {

// Uitls
// -----

/** @brief Calculate Newton coefficients by a signle CPU or GPU core.
 *  @note Keep in mind:
 *   - Function values are pre-copied to ``coeff``.
 *   - N-dim of ``coeff`` may be less than ``grid``. In this case, only dimensions from ``grid.ndim() - coeff.ndim()``
 *  are considered.
 */
__cuhostdev__ void calc_newton_coeffs_single_core(const interpolant::CartesianGrid & grid, array::NdData & coeff);

/** @brief Call divide difference function on GPU.*/
void call_divdiff_kernel(const array::Parcel * p_a1, const array::Parcel * p_a2,
                         double x1, double x2, array::Parcel * p_result, std::uint64_t size,
                         std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);











/** @brief Calculate Newton coefficients without using recursion (for loop only).*/
void calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                             array::Array & coeff);

/** @brief Calculate Newton interpolation coefficients on a full Cartesian grid using GPU.
 *  @param grid Cartesian grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 *  @param stream CUDA stream of execution of the CUDA kernel.
 *  @note This is asynchronous calculation. User should call ``merlin::cuda::Stream::synchronize(void)`` to force the
 *  CPU to wait until the calculation has finished.
 */
void calc_newton_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                            array::Parcel & coeff, const cuda::Stream & stream = cuda::Stream());



/** @brief Evaluate Newton interpolation on a full Cartesian grid using CPU.*/
double eval_newton_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                       const Vector<double> & x);

}  // namespace merlin::interpolant

#endif  // MERLIN_INTERPOLANT_NEWTON_HPP_
