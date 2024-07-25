// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_TRIAL_POLICY_HPP_
#define MERLIN_CANDY_TRIAL_POLICY_HPP_

#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/candy/declaration.hpp"  // merlin::candy::TrialPolicy
#include "merlin/exports.hpp"            // MERLIN_EXPORTS

namespace merlin {

/** @brief Trial policy for test run (dry run).
 *  @details The dry run for Candecomp models necessitates 3 phases:
 *  - **discarded** : chaotic evolution at the beginning of the training process.
 *  - **strict** : strictly descent of the error in the direction of the gradient. Relative decrement of relative error
 *    of a given step must be greater than @f$ 10^{-6} @f$ the relative error of the previous step.
 *  - **loose** : random walk near a local minimum due to rounding errors. Relative error of a given step is allowed to
 *    increase up to @f$ 10^{-10} @f$ times the one of the previous step.
 */
class candy::TrialPolicy {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from attributes.*/
    MERLIN_EXPORTS TrialPolicy(std::uint64_t discarded = 1, std::uint64_t strict = 199, std::uint64_t loose = 800);
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Get constant reference to number of discarded steps.*/
    constexpr const std::uint64_t & discarded(void) const noexcept { return this->discarded_; }
    /** @brief Get constant reference to number of steps with strictly descent of error.*/
    constexpr const std::uint64_t & strict(void) const noexcept { return this->strict_; }
    /** @brief Get constant reference to number of steps tolerated for random rounding error near local minima.*/
    constexpr const std::uint64_t & loose(void) const noexcept { return this->loose_; }
    /** @brief Get constant reference to total number of steps.*/
    constexpr const std::uint64_t & sum(void) const noexcept { return this->sum_; }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

  protected:
    /** @brief Number of discarded steps.*/
    std::uint64_t discarded_;
    /** @brief Number of steps with strictly descent of error.*/
    std::uint64_t strict_;
    /** @brief Number of steps tolerated for random rounding error near local minima.*/
    std::uint64_t loose_;
    /** @brief Total number of steps.*/
    std::uint64_t sum_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRIAL_POLICY_HPP_
