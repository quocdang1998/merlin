// Copyright 2024 quocdang1998
#ifndef MERLIN_LINALG_QRP_DECOMP_HPP_
#define MERLIN_LINALG_QRP_DECOMP_HPP_

#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::QRPDecomp
#include "merlin/linalg/matrix.hpp"       // merlin::linalg::Matrix
#include "merlin/permutation.hpp"         // merlin::Permutation
#include "merlin/vector.hpp"              // merlin::DoubleVec

namespace merlin {

/** @brief QR decomposition with column pivot.*/
class linalg::QRPDecomp {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    QRPDecomp(void) = default;
    /** @brief Constructor of an empty matrix from shape.
     *  @param nrow Number of rows.
     *  @param ncol Number of columns.
     */
    MERLIN_EXPORTS QRPDecomp(std::uint64_t nrow, std::uint64_t ncol);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    QRPDecomp(const linalg::QRPDecomp & src) = default;
    /** @brief Copy assignment.*/
    linalg::QRPDecomp & operator=(const linalg::QRPDecomp & src) = default;
    /** @brief Move constructor.*/
    QRPDecomp(linalg::QRPDecomp && src) = default;
    /** @brief Move assignment.*/
    linalg::QRPDecomp & operator=(linalg::QRPDecomp && src) = default;
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get reference to the core matrix.*/
    constexpr linalg::Matrix & core(void) noexcept { return this->core_; }
    /** @brief Get constant reference to the core matrix.*/
    constexpr const linalg::Matrix & core(void) const noexcept { return this->core_; }
    /** @brief Get number of rows.*/
    constexpr const std::uint64_t & nrow(void) const noexcept { return this->core_.nrow(); }
    /** @brief Get number of columns.*/
    constexpr const std::uint64_t & ncol(void) const noexcept { return this->core_.ncol(); }
    /** @brief Get reference to the diagonal matrix.*/
    constexpr DoubleVec & diag(void) noexcept { return this->diag_; }
    /** @brief Get constant reference to the diagonal matrix.*/
    constexpr const DoubleVec & diag(void) const noexcept { return this->diag_; }
    /** @brief Flag indicating if the current instance is initialized and decomposed.*/
    bool is_decomposed = false;
    /// @}

    /// @name Perform the decomposition
    /// @{
    /** @brief Perform the QR decomposition with column pivoting.
     *  @details The core matrix @f$ \mathbf{A} @f$ is decomposed into:
     *  @f[ \mathbf{A} = \mathbf{Q} \mathbf{D} \mathbf{R} \mathbf{P}^{\intercal} @f],
     *  in which @f$ \mathbf{Q} @f$ is an orthogonal matrix, @f$ \mathbf{D} @f$ is a diagonal matrix whose entries are
     *  in non-increasing order, @f$ \mathbf{R} @f$ is an upper triangular matrix with ``1.0`` on its diagonal, and @f$
     *  \mathbf{P} @f$ is a permutation matrix.
     */
    MERLIN_EXPORTS void decompose(std::uint64_t nthreads = 1);
    /** @brief Solve the linear least-square problem.*/
    MERLIN_EXPORTS double solve(double * solution) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~QRPDecomp(void) = default;
    /// @}

  protected:
    /** @brief Core matrix.*/
    linalg::Matrix core_;
    /** @brief Diagonal elements.*/
    DoubleVec diag_;
    /** @brief Permutation matrix.*/
    Permutation permut_;
};

}  // namespace merlin

#endif  // MERLIN_LINALG_QRP_DECOMP_HPP_
