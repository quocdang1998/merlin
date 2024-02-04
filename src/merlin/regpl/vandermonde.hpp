// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_VANDERMONDE_HPP_
#define MERLIN_REGPL_VANDERMONDE_HPP_

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Vandermonde
#include "merlin/vector.hpp"  // merlin::intvec

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
    MERLIN_EXPORTS Vandermonde(const intvec & order, const grid::CartesianGrid & grid, std::uint64_t n_threads = 24);
    /** @brief Constructor from a full polynomial and grid points.
     *  @param order Max power per dimension (one-more than highest power).
     *  @param grid_points Coordinates of points in the grid.
     *  @param n_threads Number of threads to perform the calculation.
     */
    MERLIN_EXPORTS Vandermonde(const intvec & order, const array::Array & grid_points, std::uint64_t n_threads = 24);
    /// @}

    /// @name Solve for coefficients
    /// @{
    MERLIN_EXPORTS regpl::Polynomial solve(double * data);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~Vandermonde(void);
    /// @}

  protected:
    /** @brief Pointer to SVD decomposition of the Vandermonde matrix.*/
    void * svd_decomp_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_REGPL_VANDERMONDE_HPP_
