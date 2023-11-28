// Copyright 2023 quocdang1998
#ifndef SPGRID_INTERPOLATOR_HPP_
#define SPGRID_INTERPOLATOR_HPP_

#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/cuda/stream.hpp"   // merlin::cuda::Stream
#include "merlin/vector.hpp"        // merlin::Vector, merlin::floatvec
#include "merlin/splint/tools.hpp"  // merlin::splint::Method

#include "spgrid/declaration.hpp"  // spgrid::Interpolator
#include "spgrid/sparse_grid.hpp"  // spgrid::SparseGrid

namespace spgrid {

/** @brief Interpolator on hierarchical grid.*/
class Interpolator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor*/
    Interpolator(void) = default;
    /** @brief Construct from a hierarchical grid and a full Cartesian data.*/
    Interpolator(const SparseGrid & grid, const merlin::array::NdData & value,
                 const merlin::Vector<merlin::splint::Method> & method, std::uint64_t n_threads = 4);
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Get constant reference to coefficients.*/
    const merlin::floatvec & get_coeff(void) const noexcept { return this->coeff_; }
    /// @}

    /// @name Evaluation
    /// @{
    /** @brief Evaluate interpolation by CPU parallelism.*/
    merlin::floatvec evaluate(const merlin::array::Array & points, std::uint64_t n_threads = 1);
    /** @brief Evaluate interpolation by GPU parallelism.*/
    merlin::floatvec evaluate(const merlin::array::Parcel & points, std::uint64_t n_threads = 32,
                              const merlin::cuda::Stream & stream = merlin::cuda::Stream());
    /// @}

  protected:
    /** @brief Sparse grid to interpolate over.*/
    SparseGrid grid_;
    /** @brief Coefficients.*/
    merlin::floatvec coeff_;
    /** @brief Method coefficients.*/
    merlin::Vector<merlin::splint::Method> method_;

  private:
    /** @brief Vector to coefficients of each level.*/
    merlin::Vector<double *> coeff_by_level_;
};

}  // namespace spgrid

#endif  // SPGRID_INTERPOLATOR_HPP_
