// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLANT_HPP_
#define MERLIN_SPLINT_INTERPOLANT_HPP_

#include "merlin/array/nddata.hpp"        // merlin::array::NdData, merlin::array::Array
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/cuda_interface.hpp"      // __cuhostdev__
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid, merlin::splint::Interpolant
#include "merlin/vector.hpp"              // merlin::Vector

namespace merlin {

namespace splint {

/** @brief Interpolation method.*/
enum class Method : unsigned int {
    /** @brief Linear interpolation.*/
    Linear = 0x00,
    /** @brief Polynomial interpolation by Lagrange method.*/
    Lagrange = 0x01,
    /** @brief Polynomial interpolation by Newton method.*/
    Newton = 0x02
};

/** @brief Construct interpolation coefficients.
 *  @param coeff C-contiguous array of coefficients (value are pre-copied to this array).
 *  @param grid Cartesian grid to interpolate.
 *  @param method Interpolation method to use on each dimension.
 *  @param n_threads Number of threads to calculate.
 */
void construct_coeff_cpu(double * coeff, const splint::CartesianGrid & grid, const Vector<splint::Method> & method,
                         std::uint64_t n_threads) noexcept;

}  // namespace splint

/** @brief Interpolate on a multi-dimensional object.
 *  @details A multi-dimensional interpolation is a linear combination of basis functions. Each basis function is a
 *  product of mono-variable functions.
 *
 *  Thanks to this property, the linear system of equations to solve for the interpolation coefficients can be
 *  decomposed as Kronecker product of one-dimensional matrices, allowing the reduction per dimension of the original
 *  matrix. Furthermore, the process of solving each component matrix can be parallelized over each linear sub-system
 *  created, thus reducing the calculation time of the coefficients.
 */
class splint::Interpolant {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Interpolant(void) = default;
    /** @brief Construct from a CPU array.*/
    MERLIN_EXPORTS Interpolant(const splint::CartesianGrid & grid, const array::Array & data,
                               const Vector<splint::Method> & method);
    /// @}

    /// @name Calculate coefficient
    /// @{
    /** @brief Calculate interpolation coefficients.*/
    void construct_coefficients(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Interpolant(void);
    /// @}

  private:
    /** @brief Cartesian grid to interpolate.*/
    const splint::CartesianGrid * p_grid_ = nullptr;
    /** @brief Pointer to coefficient array.*/
    array::NdData * p_coeff_ = nullptr;
    /** @brief Interpolation method to applied on each dimension.*/
    Vector<splint::Method> method_;
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_INTERPOLANT_HPP_
