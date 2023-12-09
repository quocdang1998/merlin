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
    /** @brief Constructor from learning rate.
     *  @param lr Initial learning rate.
     *  @param num_params Number of parameter in the model to train.
     *  @param b Bias.
     */
    AdaGrad(double lr, std::uint64_t num_params, double b = 1.0e-8) :
    learning_rate(lr), bias(b), grad_history(num_params) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    AdaGrad(const candy::optmz::AdaGrad & src) = default;
    /** @brief Copy assignment.*/
    candy::optmz::AdaGrad & operator=(const candy::optmz::AdaGrad & src) = default;
    /** @brief Move constructor.*/
    AdaGrad(candy::optmz::AdaGrad && src) = default;
    /** @brief Move assignment.*/
    candy::optmz::AdaGrad & operator=(candy::optmz::AdaGrad && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                   std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a CPU parallel region.*/
    __cudevice__ void update_gpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                 std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate additional number of bytes to allocate for dynamic data.*/
    std::uint64_t additional_cumalloc(void) const noexcept { return 0; }
    /** @brief Copy the optimizer from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::optmz::GradDescent * gpu_ptr, void * dynamic_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate additional number of bytes to allocate in CUDA shared memory for dynamic data.*/
    std::uint64_t additional_sharedmem(void) const noexcept { return 0; }
#ifdef __NVCC__
    /** @brief Copy object to pre-allocated memory region by current CUDA block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the object is copied to.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(candy::optmz::GradDescent * dest_ptr, void * dynamic_data_ptr,
                                      std::uint64_t thread_idx, std::uint64_t block_size) const;
    /** @brief Copy object to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the object is copied to.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     */
    __cudevice__ void * copy_by_thread(candy::optmz::GradDescent * dest_ptr, void * dynamic_data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Initial learning rate.*/
    double learning_rate;
    /** @brief Bias to prevent division error.*/
    double bias;
    /** @brief Sum of squares of gradients in history.*/
    floatvec grad_history;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
