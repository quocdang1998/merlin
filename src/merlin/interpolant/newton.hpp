// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_NEWTON_HPP_
#define MERLIN_INTERPOLANT_NEWTON_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {

/** @brief Calculate Newton interpolation coefficients on a full Cartesian grid using CPU.*/
void calc_newton_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                            array::Array & coeff);

/** @brief Evaluate Newton interpolation on a full Cartesian grid using CPU.*/
double eval_newton_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                       const Vector<double> & x);

}  // namespace merlin::interpolant

#endif  // MERLIN_INTERPOLANT_NEWTON_HPP_
