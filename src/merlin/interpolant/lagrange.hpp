// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_LAGRANGE_HPP_
#define MERLIN_INTERPOLANT_LAGRANGE_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                              array::Array & coeff);

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU.*/
void calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                              array::Parcel & coeff, const cuda::Stream & stream = cuda::Stream());

/** @brief Evaluate Lagrange interpolation on a full Cartesian grid using CPU.*/
double eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                         const Vector<double> & x);

#ifdef __comment
/** @brief Calculate Lagrage interpolation coefficients on a Cartesian grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::NdData & value,
                              const Vector<array::Slice> & slices, array::NdData & coeff);

/** @brief Calculate Lagrage interpolation coefficients on a sparse grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::SparseGrid & grid, array::NdData & coeff);

/** @brief Evaluate Lagrange interpolation on a Cartesian grid using CPU.*/
double eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::NdData & coeff,
                         const Vector<array::Slice> & slices, const Vector<double> & x);

/** @brief Evaluate Lagrange interpolation on a Sparse grid using CPU.*/
double eval_lagrange_cpu(const interpolant::SparseGrid & grid, const array::NdData & coeff, const Vector<double> & x);
#endif

}  // namespace merlin::interpolant

#endif  // MERLIN_INTERPOLANT_LAGRANGE_HPP_
