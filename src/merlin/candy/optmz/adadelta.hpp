// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADADELTA_HPP_
#define MERLIN_CANDY_OPTMZ_ADADELTA_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::AdaDelta
#include "merlin/config.hpp"                   // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

/** @brief %Optimizer by adaptive delta method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ s_{t+1} = \rho s_t + (1 - \rho) {\left(\frac{\partial L}{\partial p}\right)_{t}}^2 @f]
 *  @f[ g_{t+1} = \sqrt{\frac{\Delta_t + \varepsilon}{s_{t+1} + \varepsilon}} \left( \frac{\partial L}{\partial p}
 *  \right)_{t} @f]
 *  @f[ p_{t+1} = p_t - \eta g_{t+1} @f]
 *  @f[ \Delta_{t+1} = \rho \Delta_t + (1 - \rho) {g_{t+1}}^2 @f]
 *
 *  in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate, @f$
 *  \rho @f$ is a constant controlling the decay of the previous parameter update, @f$ \varepsilon @f$ is the correction
 *  factor (bias), and @f$ L @f$ is the loss function.
 */
struct candy::optmz::AdaDelta {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    AdaDelta(void) = default;
    /** @brief Constructor from members.
     *  @param lr Learning rate.
     *  @param r Constant controlling the decay of the previous parameter update.
     *  @param b Bias.
     */
    AdaDelta(double lr, double r, double b = 1.0e-8) : learning_rate(lr), rho(r), bias(b) {}
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS static void update_cpu(void * optimizer_algor, double * history, candy::Model & model,
                                          const candy::Gradient & grad, std::uint64_t time_step) noexcept;
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
    double rho;
    /** @brief Bias to prevent division error.*/
    double bias;
    /// @}
};

namespace candy::optmz {

/** @brief Create an optimizer with adadelta algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_adadelta(double learning_rate, double rho, std::uint64_t num_params,
                                                double bias = 1.0e-8);

}  // namespace candy::optmz

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_ADADELTA_HPP_
