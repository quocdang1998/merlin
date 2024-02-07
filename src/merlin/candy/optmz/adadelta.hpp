// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADADELTA_HPP_
#define MERLIN_CANDY_OPTMZ_ADADELTA_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::AdaDelta
#include "merlin/cuda_interface.hpp"           // __cudevice__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

/** @brief %Optimizer by adaptive learning rate method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ s_{t+1} = \rho s_t + (1 - \rho) {\left(\frac{\partial L}{\partial p}\right)_{t}}^2 @f]
 *  @f[ p_{t+1} = p_t - \sqrt{\frac{\Delta_t + \varepsilon}{s_{t+1} + \varepsilon}} \left( \frac{\partial L}{\partial p}
 *  \right)_{t} @f]
 *  @f[ \Delta_{t+1} = \rho \Delta_t + (1 - \rho) {\left( p_{t+1} - p_t \right)_{t}}^2 @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \rho @f$ is a constant controlling
 *  the decay of the previous parameter update, @f$ \varepsilon @f$ is the correction factor (bias), and @f$ L @f$ is
 *  the loss function.
 */
struct candy::optmz::AdaDelta {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    AdaDelta(void) = default;
    /** @brief Constructor from members.
     *  @param r Constant controlling the decay of the previous parameter update.
     *  @param rms_delta_mem Pre-allocated data for storing values of RMS and rescaled moments.
     *  @param b Bias.
     */
    AdaDelta(double r, char * rms_delta_mem, double b = 1.0e-8) :
    rho(r), rms_delta(reinterpret_cast<double *>(rms_delta_mem)), bias(b) {}
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
    /** @brief Constant controlling the decay of the previous parameter update.*/
    double rho;
    /** @brief Pointer to array containing RMS and rescaled moments.*/
    double * rms_delta;
    /** @brief Bias to prevent division error.*/
    double bias;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_ADADELTA_HPP_
