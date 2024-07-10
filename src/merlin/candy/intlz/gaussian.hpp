// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_INTLZ_GAUSSIAN_HPP_
#define MERLIN_CANDY_INTLZ_GAUSSIAN_HPP_

#include <cstdint>  // std::uint64_t
#include <random>   // std::normal_distribution

#include "merlin/candy/intlz/declaration.hpp"  // merlin::candy::intlz::Gaussian
#include "merlin/candy/intlz/initializer.hpp"  // merlin::candy::intlz::Initializer
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

// Gaussian
// --------

/** @brief Initialization by normal distribution from data mean and variance.*/
class candy::intlz::Gaussian : public candy::intlz::Initializer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from mean and standard deviation.*/
    MERLIN_EXPORTS Gaussian(double mean = 0.0, double scale = 1.0);
    /// @}

    /// @name Sample data
    /// @{
    /** @brief Sample a value for initializing CP model.*/
    MERLIN_EXPORTS double sample(void) override;
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

#endif  // MERLIN_CANDY_INTLZ_GAUSSIAN_HPP_
