// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_RAND_GAUSSIAN_HPP_
#define MERLIN_CANDY_RAND_GAUSSIAN_HPP_

#include <cstdint>  // std::uint64_t
#include <string>   // std::string
#include <random>   // std::normal_distribution

#include "merlin/candy/rand/declaration.hpp"  // merlin::candy::rand::Gaussian
#include "merlin/exports.hpp"                 // MERLIN_EXPORTS

namespace merlin {

// Gaussian
// --------

/** @brief Initialization by normal distribution from data mean and variance.
 *  @details Let @f$ \mu @f$ and @f$ \sigma @f$ respectively the mean and the standard deviation of the hyper-slice
 *  corresponding to a given parameter @f$ v @f$ of the CP model. The value of the parameter is then initialized as:
 *  @f[ v \sim \mathcal{N}(\mu, \sigma^{2}) @f]
 *  in which @f$ \mathcal{N} @f$ represents the normal distribution.
 */
class candy::rand::Gaussian {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Gaussian(void) = default;
    /// @}

    /// @name Set parameters
    /// @{
    /** @brief Change mean and standard deviation of the distribution.*/
    MERLIN_EXPORTS void set_params(double mean, double std_dev);
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
    ~Gaussian(void) = default;
    /// @}

  protected:
    /** @brief Random generator.*/
    std::normal_distribution<double> generator_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_RAND_GAUSSIAN_HPP_
