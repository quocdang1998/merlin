// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADAGRAD_HPP_
#define MERLIN_CANDY_OPTMZ_ADAGRAD_HPP_

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::AdaGrad
#include "merlin/cuda_interface.hpp"           // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS
#include "merlin/vector.hpp"                   // merlin::floatvec

namespace merlin {

/** @brief %Optimizer by adaptive gradient method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ g_t = g_{t-1} + {\left(\frac{\partial L}{\partial p}\right)_{t}}^2 @f]
 *  @f[ p_{t+1} = p_t - \frac{\eta}{\sqrt{\varepsilon + g_t}} \left( \frac{\partial L}{\partial p} \right)_t @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate,
 *  @f$ \varepsilon @f$ is the correction factor (bias), and @f$ L @f$ is the loss function.
 */
struct candy::optmz::AdaGrad {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    AdaGrad(void) = default;
    /** @brief Constructor from members.
     *  @param lr Initial learning rate.
     *  @param grad_mean_mem Pre-allocated data for storing mean of gradient values.
     *  @param b Bias.
     */
    AdaGrad(double lr, char * grad_history_mem, double b = 1.0e-8) :
    learning_rate(lr), grad_history(reinterpret_cast<double *>(grad_history_mem)), bias(b) {}
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
    /** @brief Initial learning rate.*/
    double learning_rate;
    /** @brief Pointer to array containing sum of squares of gradients in history.*/
    double * grad_history;
    /** @brief Bias to prevent division error.*/
    double bias;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
