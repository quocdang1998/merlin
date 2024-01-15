// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADAM_HPP_
#define MERLIN_CANDY_OPTMZ_ADAM_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::AdaGrad
#include "merlin/cuda_interface.hpp"           // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

/** @brief %Optimizer by adaptive estimates of lower-order moments method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ m_{t+1} = \frac{1}{1 - (\beta_m)^t} \left[ \beta_m m_t + (1 - \beta_m) \left( \frac{\partial L}{\partial p}
 *  \right)_t \right] @f]
 *  @f[ v_{t+1} = \frac{1}{1 - (\beta_v)^t} \left[ \beta_v v_t + (1 - \beta_v) {\left( \frac{\partial L}{\partial p}
 *  \right)_t}^2 \right] @f]
 *  @f[ p_{t+1} = p_t - \eta \frac{m_{t+1}}{\varepsilon + \sqrt{v_{t+1}}} @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ m_t @f$ is the first moment,
 *  @f$ v_t @f$ is the second moment, @f$ \beta_m @f$ is the exponential decay rate of the first moment,
 *  @f$ \beta_v @f$ is the exponential decay rate of the second moment, @f$ \eta @f$ is the learning rate,
 *  @f$ \varepsilon @f$ is the correction factor (bias), and @f$ L @f$ is the loss function.
 */
struct candy::optmz::Adam {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Adam(void) = default;
    /** @brief Constructor from members.
     *  @param lr Initial learning rate.
     *  @param bm First moment decay constant.
     *  @param bv Second moment decay constant.
     *  @param moment_mem Pre-allocated data for storing values of first and second moments.
     *  @param b Bias.
     *  @param t Time step.
     */
    Adam(double lr, double bm, double bv, char * moment_mem, double b = 1.0e-8, std::uint64_t t = 0) :
    learning_rate(lr), beta_m(bm), beta_v(bv), moments(reinterpret_cast<double *>(moment_mem)), bias(b), time_step(t) {}
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
    /** @brief First moment decay constant.*/
    double beta_m;
    /** @brief Second moment decay constant.*/
    double beta_v;
    /** @brief Pointer to array containing first and second moments.*/
    double * moments;
    /** @brief Bias to prevent division error.*/
    double bias;
    /** @brief Time step.*/
    std::uint64_t time_step;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_ADAM_HPP_
