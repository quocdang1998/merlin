// Copyright 2023 quocdang1998
#ifndef SPGRID_INTERPOLATOR_HPP_
#define SPGRID_INTERPOLATOR_HPP_

#include <string>  // std::string
#include <vector>  // std::vector

#include "merlin/array/nddata.hpp"         // merlin::array::NdData
#include "merlin/cuda/stream.hpp"          // merlin::cuda::Stream
#include "merlin/splint/interpolator.hpp"  // merlin::splint::Interpolator
#include "merlin/splint/tools.hpp"         // merlin::splint::Method
#include "merlin/vector.hpp"               // merlin::Vector, merlin::DoubleVec


#include "spgrid/declaration.hpp"  // spgrid::Interpolator

namespace mln = merlin;

namespace spgrid {

/** @brief Interpolator on hierarchical grid.
 *  @note Not to be too greedy, because the interpolation is very prone to catastrophic cancellation.
 */
class HierInterpolator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor*/
    HierInterpolator(void) = default;
    /** @brief Construct from a hierarchical grid and a full Cartesian data.*/
    HierInterpolator(const SparseGrid & grid, const mln::array::Array & full_data,
                           const mln::Vector<mln::splint::Method> & method);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    HierInterpolator(const HierInterpolator & src) = delete;
    /** @brief Copy assignment.*/
    HierInterpolator & operator=(const HierInterpolator & src) = delete;
    /** @brief Move constructor.*/
    HierInterpolator(HierInterpolator && src) = default;
    /** @brief Move assignment.*/
    HierInterpolator & operator=(HierInterpolator && src) = default;
    /// @}

    /// @name Evaluation
    /// @{
    /** @brief Evaluate interpolation.*/
    void evaluate(const mln::array::Array & points, mln::DoubleVec & result);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const;
    /// @}

  protected:
    /** @brief Vector of Cartesian interpolator.*/
    std::vector<mln::splint::Interpolator> intpl;
};

}  // namespace spgrid

#endif  // SPGRID_INTERPOLATOR_HPP_
