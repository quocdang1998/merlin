// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_INTLZ_UNIFORM_HPP_
#define MERLIN_CANDY_INTLZ_UNIFORM_HPP_

#include <cstdint>  // std::uint64_t
#include <random>   // std::uniform_real_distribution

#include "merlin/candy/intlz/declaration.hpp"  // merlin::candy::intlz::Uniform
#include "merlin/candy/intlz/initializer.hpp"  // merlin::candy::intlz::Initializer
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

// Uniform
// --------

/** @brief Initialization by uniform distribution from data mean and scale.*/
class candy::intlz::Uniform : public candy::intlz::Initializer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from mean and scale.*/
    MERLIN_EXPORTS Uniform(double mean = 0.0, double scale = 0.01);
    /// @}

    /// @name Sample data
    /// @{
    /** @brief Sample a value for initializing CP model.*/
    MERLIN_EXPORTS double sample(void) override;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Uniform(void) = default;
    /// @}

  protected:
    /** @brief Random generator.*/
    std::uniform_real_distribution<double> generator_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_INTLZ_UNIFORM_HPP_
