// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_TOOLS_HPP_
#define MERLIN_SPLINT_TOOLS_HPP_

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/cuda_interface.hpp"      // __cuhostdev__
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid, merlin::splint::Interpolant
#include "merlin/vector.hpp"              // merlin::Vector

namespace merlin {

namespace splint {

// Interpolation Methods
// ---------------------

/** @brief Interpolation method.*/
enum class Method : unsigned int {
    /** @brief Linear interpolation.*/
    Linear = 0x00,
    /** @brief Polynomial interpolation by Lagrange method.*/
    Lagrange = 0x01,
    /** @brief Polynomial interpolation by Newton method.*/
    Newton = 0x02
};

// Construct Coefficients
// ----------------------

/** @brief Construct interpolation coefficients.
 *  @param coeff C-contiguous array of coefficients (value are pre-copied to this array).
 *  @param grid Cartesian grid to interpolate.
 *  @param method Interpolation method to use on each dimension.
 *  @param n_threads Number of threads to calculate.
 */
void construct_coeff_cpu(double * coeff, const splint::CartesianGrid & grid, const Vector<splint::Method> & method,
                         std::uint64_t n_threads) noexcept;

// Evaluate Interpolation
// ----------------------

/** @brief Decrease an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return Lowest changed dimension.
 */
// __cuhostdev__ std::int64_t decrement_index(intvec & index, const intvec & shape) noexcept;

/** @brief Evaluate interpolation from constructed coefficients.
 *  @param coeff C-contiguous array of coefficients.
 *  @param grid Cartesian grid to interpolate.
 *  @param method Interpolation method to use on each dimension.
 *  @param points Pointer to the first coordinate of the first point. Coordinates of the same point are placed
 *  side-by-side in the array
 *  @param n_points Number of points to interpolate.
 *  @param result Pointer to the array storing the result.
 *  @param n_threads Number of threads to perform the interpolation.
 */
void eval_intpl_cpu(const double * coeff, const splint::CartesianGrid & grid, const Vector<splint::Method> & method,
                    const double * points, std::uint64_t n_points, double * result, std::uint64_t n_threads) noexcept;

}  // namespace splint

}  // namespace merlin

#endif  // MERLIN_SPLINT_TOOLS_HPP_
