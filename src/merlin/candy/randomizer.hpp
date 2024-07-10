// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_RANDOMIZER_HPP_
#define MERLIN_CANDY_RANDOMIZER_HPP_

#include "merlin/candy/intlz/gaussian.hpp"     // merlin::candy::intlz::Gaussian
#include "merlin/candy/intlz/initializer.hpp"  // merlin::candy::intlz::Initializer
#include "merlin/candy/intlz/uniform.hpp"      // merlin::candy::intlz::Uniform

namespace merlin::candy {

/** @brief Initializer type.*/
enum class Randomizer : unsigned int {
    /** @brief Gaussian randomization from mean and standard deviation of data hyper-slice.*/
    Gaussian = 0x00,
    /** @brief Uniform randomization around the mean value of data hyper-slice.*/
    Uniform = 0x01
};

}  // namespace merlin::candy

#endif  // MERLIN_CANDY_RANDOMIZER_HPP_
