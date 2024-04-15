// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
#define MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/config.hpp"                   // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

/** @brief %Optimizer by stochastic gradient descent method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ p_{t+1} = p_t - \eta \left( \frac{\partial L}{\partial p} \right)_t @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate, and
 *  @f$ L @f$ is the loss function.
 */
struct candy::optmz::GradDescent {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    GradDescent(void) = default;
    /** @brief Constructor from learning rate.*/
    GradDescent(double lr) : learning_rate(lr) {}
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS static void update_cpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                          std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a GPU parallel region.*/
    __cudevice__ static void update_gpu(void * optimizer_algor, candy::Model & model, const candy::Gradient & grad,
                                        std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Learning rate.*/
    double learning_rate;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
