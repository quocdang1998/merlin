// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_VANDERMONDE_HPP_
#define MERLIN_REGPL_VANDERMONDE_HPP_

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid, merlin::grid::RegularGrid
#include "merlin/config.hpp"    // merlin::Index
#include "merlin/linalg/qrp_decomp.hpp"  // merlin::linalg::QRPDecomp
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial, merlin::regpl::Vandermonde
#include "merlin/vector.hpp"  // merlin::UIntVec

namespace merlin {

/** @brief Vandermonde matrix of a polynomial and a grid.*/
class regpl::Vandermonde {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Vandermonde(void) = default;
    /** @brief Constructor from a full polynomial and Cartesian grid.
     *  @param order Max power per dimension (one-more than highest power).
     *  @param grid Cartesian grid of points.
     *  @param n_threads Number of threads to perform the calculation.
     */
    MERLIN_EXPORTS Vandermonde(const Index & order, const grid::CartesianGrid & grid, std::uint64_t n_threads = 1);
    /** @brief Constructor from a full polynomial and grid points.
     *  @param order Max power per dimension (one-more than highest power).
     *  @param grid Regular grid of points.
     *  @param n_threads Number of threads to perform the calculation.
     */
    MERLIN_EXPORTS Vandermonde(const Index & order, const grid::RegularGrid & grid, std::uint64_t n_threads = 1);
    /// @}

    /// @name Solve for coefficients
    /// @{
    /** @brief Solve for the coefficients given the data.*/
    MERLIN_EXPORTS regpl::Polynomial solve(DoubleVec & values_to_fit) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~Vandermonde(void) = default;
    /// @}

  protected:
    /** @brief Polynomial order.*/
    Index order_;
    /** @brief Index of eligible terms in the full polynomial.*/
    UIntVec term_idx_;
    /** @brief Pointer to SVD decomposition of the Vandermonde matrix.*/
    linalg::QRPDecomp solver_;
};

}  // namespace merlin

#endif  // MERLIN_REGPL_VANDERMONDE_HPP_
