// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLATOR_HPP_
#define MERLIN_SPLINT_INTERPOLATOR_HPP_

#include "merlin/array/nddata.hpp"        // merlin::array::NdData, merlin::array::Array
#include "merlin/cuda/stream.hpp"         // merlin::cuda::Stream
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid, merlin::splint::Interpolant
#include "merlin/splint/tools.hpp"        // merlin::splint::Method
#include "merlin/vector.hpp"              // merlin::Vector

namespace merlin {

/** @brief Interpolate on a multi-dimensional object.
 *  @details A multi-dimensional interpolation is a linear combination of basis functions. Each basis function is a
 *  product of mono-variable functions.
 *
 *  Thanks to this property, the linear system of equations to solve for the interpolation coefficients can be
 *  decomposed as Kronecker product of one-dimensional matrices, allowing the reduction per dimension of the original
 *  matrix. Furthermore, the process of solving each component matrix can be parallelized over each linear sub-system
 *  created, thus reducing the calculation time of the coefficients.
 */
class splint::Interpolator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Interpolator(void) = default;
    /** @brief Construct from a CPU array.*/
    MERLIN_EXPORTS Interpolator(const splint::CartesianGrid & grid, array::Array & data,
                                const Vector<splint::Method> & method);
    /** @brief Construct from a GPU array*/
    MERLIN_EXPORTS Interpolator(const splint::CartesianGrid & grid, array::Parcel & data,
                                const Vector<splint::Method> & method, const cuda::Stream & stream = cuda::Stream());
    /// @}

    /// @name Calculate coefficient
    /// @{
    /** @brief Calculate interpolation coefficients.*/
    void construct_coefficients(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Interpolator(void);
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

#endif  // MERLIN_SPLINT_INTERPOLATOR_HPP_
