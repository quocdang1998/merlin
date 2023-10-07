// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLANT_HPP_
#define MERLIN_SPLINT_INTERPOLANT_HPP_

#include "merlin/array/nddata.hpp"        // merlin::array::NdData, merlin::array::Array
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
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

}  // namespace splint

/** @brief Interpolate on a multi-dimensional object.*/
class splint::Interpolant {
  public:
    /** @brief */

    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Interpolant(void) = default;
    /** @brief Construct from a CPU array.*/
    MERLIN_EXPORTS Interpolant(const splint::CartesianGrid & grid, const array::Array & data,
                               const Vector<splint::Method> & method);
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
