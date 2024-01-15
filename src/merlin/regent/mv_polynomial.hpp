// Copyright 2024 quocdang1998
#ifndef MERLIN_REGENT_MV_POLYNOM_HPP_
#define MERLIN_REGENT_MV_POLYNOM_HPP_

#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/regent/declaration.hpp"  // merlin::regent::MvPolynomial
#include "merlin/vector.hpp"  // merlin::floatvec, merlin::intvec

namespace merlin {

/** @brief Multi-variate polynomial.*/
class regent::MvPolynomial {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    MvPolynomial(void) = default;
    /** @brief Constructor of an empty polynomial from order per dimension.*/
    MERLIN_EXPORTS MvPolynomial(const intvec & order_per_dim);
    /** @brief Constructor of a pre-allocated array of coefficients and order per dimension.*/
    MERLIN_EXPORTS MvPolynomial(double * coeff_data, const intvec & order_per_dim);
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get number of dimension.*/
    __cuhostdev__ constexpr std::uint64_t & ndim(void) const noexcept { return this->order_.size(); }
    /** @brief Get number of terms in the polynomial.*/
    __cuhostdev__ constexpr std::uint64_t & size(void) const noexcept { return this->coeff_.size(); }
    /// @}

    /// @name Evaluation
    /// @{
    /** @brief Evaluate polynomial value at a given point.
     *  @param point Pointer to coordinates of the point.
     *  @param buffer Buffer memory for calculation,
    */
    __cuhostdev__ double eval(const double * point, double * buffer) const noexcept;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~MvPolynomial(void) = default;
    /// @}

  protected:
    /** @brief Coefficient data.*/
    floatvec coeff_;
    /** @brief Order per dimension.*/
    intvec order_;
};

}  // namespace merlin

#endif  // MERLIN_REGENT_MV_POLYNOM_HPP_
