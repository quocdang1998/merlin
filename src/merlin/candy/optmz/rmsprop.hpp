// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_RMSPROP_HPP_
#define MERLIN_CANDY_OPTMZ_RMSPROP_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::RmsProp
#include "merlin/config.hpp"                   // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

/** @brief %Optimizer by root mean square propagation method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ s_{t+1} = \beta s_t + (1 - \beta) {\left(\frac{\partial L}{\partial p}\right)_{t}}^2 @f]
 *  @f[ p_{t+1} = p_t - \frac{\eta}{\sqrt{\varepsilon + s_{t+1}}} \left(\frac{\partial L}{\partial p}\right)_{t} @f]
 *
 *  in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate, @f$
 *  \beta @f$ is a constant controlling the decay of the previous parameter update, @f$ \varepsilon @f$ is the
 *  correction factor (bias), and @f$ L @f$ is the loss function.
 */
struct candy::optmz::RmsProp {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    RmsProp(void) = default;
    /** @brief Constructor from members.
     *  @param lr Learning rate.
     *  @param b Constant controlling the decay of the previous parameter update.
     *  @param e Bias to prevent division error.
     */
    RmsProp(double lr, double b, double e = 1.0e-16) : learning_rate(lr), beta(b), bias(e) {}
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS static void update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                          const candy::Gradient & grad, std::uint64_t time_step,
                                          std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a GPU parallel region.*/
    __cudevice__ static void update_gpu(void * optimizer_algor, double * history, candy::Model & model,
                                        const candy::Gradient & grad, std::uint64_t time_step, std::uint64_t thread_idx,
                                        std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Learning rate.*/
    double learning_rate;
    /** @brief Constant controlling the decay of the previous parameter update.*/
    double beta;
    /** @brief Bias to prevent division error.*/
    double bias;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_RMSPROP_HPP_
