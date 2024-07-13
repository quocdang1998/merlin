// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_RAND_UNIFORM_HPP_
#define MERLIN_CANDY_RAND_UNIFORM_HPP_

#include <cstdint>  // std::uint64_t
#include <random>   // std::uniform_real_distribution

#include "merlin/candy/rand/declaration.hpp"  // merlin::candy::rand::Uniform
#include "merlin/exports.hpp"                 // MERLIN_EXPORTS

namespace merlin {

// Uniform
// --------

/** @brief Initialization by uniform distribution from data mean.
 *  @details Let @f$ \mu @f$ the mean of the hyper-slice corresponding to a given parameter @f$ v @f$ of the CP model.
 *  The value of the parameter is then initialized as:
 *  @f[ v \sim \mathcal{U}(\mu(1-k), \mu(1-k)) @f]
 *  in which @f$ \mathcal{U} @f$ is the uniform distribution, and @f$ k @f$ is the relative scale wrt. the mean value
 *  chosen by user.
 */
class candy::rand::Uniform {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from relative range value.*/
    MERLIN_EXPORTS Uniform(double k = 0.01);
    /// @}

    /// @name Set parameters
    /// @{
    /** @brief Change mean of the distribution.*/
    MERLIN_EXPORTS void set_params(double mean);
    /// @}

    /// @name Sample data
    /// @{
    /** @brief Sample a value for initializing CP model.*/
    MERLIN_EXPORTS double sample(void);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Uniform(void) = default;
    /// @}

  protected:
    /** @brief Random generator.*/
    std::uniform_real_distribution<double> generator_;
    /** @brief Relative range.*/
    double k_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_RAND_UNIFORM_HPP_
