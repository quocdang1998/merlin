// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_RANDOMIZER_HPP_
#define MERLIN_CANDY_RANDOMIZER_HPP_

#include <ostream>  // std::ostream
#include <variant>  // std::variant
#include <utility>  // std::forward

#include "merlin/candy/declaration.hpp"    // merlin::candy::Randomizer
#include "merlin/candy/rand/gaussian.hpp"  // merlin::candy::rand::Gaussian
#include "merlin/candy/rand/uniform.hpp"   // merlin::candy::rand::Uniform
#include "merlin/exports.hpp"              // MERLIN_EXPORTS

namespace merlin::candy {

/** @brief Randomization method for each axis.*/
using Randomizer = std::variant<candy::rand::Gaussian, candy::rand::Uniform>;

/** @brief Change mean and standard deviation of the distribution.*/
MERLIN_EXPORTS void set_params(Randomizer & initializer, double mean, double std_dev);

/** @brief Sample a value for initializing CP model.*/
MERLIN_EXPORTS double sample(Randomizer & initializer);

}  // namespace merlin::candy

#endif  // MERLIN_CANDY_RANDOMIZER_HPP_
