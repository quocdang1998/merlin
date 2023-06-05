// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
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
    #ifdef __NVCC__
    /** @brief Update model by gradient on GPU.*/
    // virtual device function bugged !!!
    __cudevice__ virtual void update_gpu(candy::Model * p_model, const double * p_gradient, std::uint64_t size) {}
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    __cuhostdev__ virtual ~Optimizer(void) {}
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
