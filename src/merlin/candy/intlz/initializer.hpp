// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_INTLZ_INITIALIZER_HPP_
#define MERLIN_CANDY_INTLZ_INITIALIZER_HPP_

#include "merlin/candy/intlz/declaration.hpp"  // merlin::candy::Initializer

namespace merlin {

/** @brief Initializer for each dimension for a Candecomp model.*/
class candy::intlz::Initializer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Initializer(void) = default;
    /// @}

    /// @name Sample data
    /// @{
    /** @brief Sample a value for initializing CP model.*/
    virtual double sample(void) = 0;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    virtual ~Initializer(void) = default;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_INTLZ_INITIALIZER_HPP_
