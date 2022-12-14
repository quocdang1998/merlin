// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_LAGRANGE_HPP_
#define MERLIN_INTERPOLANT_LAGRANGE_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {

/** @brief Calculate Lagrage interpolation coefficients on a Cartesian grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                              const Vector<array::Slice> & slices, array::Array & coeff);

/** @brief Calculate Lagrage interpolation coefficients on a sparse grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::SparseGrid & grid, array::NdData & coeff);

/** @brief Evaluate Lagrange interpolation on a Cartesian grid using CPU.*/
double eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                         const Vector<array::Slice> & slices, const Vector<double> & x);

/** @brief Evaluate Lagrange interpolation on a Sparse grid using CPU.*/
double eval_lagrange_cpu(const interpolant::SparseGrid & grid, const array::Array & coeff, const Vector<double> & x);

}  // namespace merlin::interpolant

#endif  // MERLIN_INTERPOLANT_LAGRANGE_HPP_
