// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include <cstdio>

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/vector.hpp"  // merlin::Vector
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief Base class for optimizer of model.*/
class candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Optimizer(void) = default;
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Optimizer(const candy::Optimizer & src) = default;
    /** @brief Copy assignment.*/
    candy::Optimizer & operator=(const candy::Optimizer & src) = default;
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src) = default;
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.*/
    virtual void update_cpu(candy::Model & model, const floatvec & gradient) {}
    /// @}

    /// @name Destructor
    /// @{
    MERLIN_EXPORTS virtual ~Optimizer(void);
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
