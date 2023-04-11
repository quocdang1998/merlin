// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADAGRAD_HPP_
#define MERLIN_CANDY_OPTMZ_ADAGRAD_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"          // merlin::candy::Optimizer
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::AdaGrad
#include "merlin/cuda_interface.hpp"           // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS
#include "merlin/vector.hpp"                   // merlin::floatvec

namespace merlin {

// AdaGrad
// -------

/** @brief %Optimizer by adaptive gradient method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ g_t = g_{t-1} + {\left(\frac{\partial L}{\partial p}\right)_{t}}^2 @f]
 *  @f[ p_{t+1} = p_t - \frac{\eta}{\sqrt{\varepsilon + g_t}} \left( \frac{\partial L}{\partial p} \right)_t @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate,
 *  @f$ \varepsilon @f$ is the correction factor (bias), and @f$ L @f$ is the loss function.
 */
class candy::optmz::AdaGrad : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from members.*/
    __cuhostdev__ AdaGrad(double learning_rate, std::uint64_t model_size, double bias = 1.0e-8) :
    learning_rate_(learning_rate), bias_(bias), grad_history_(model_size) {}
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

    /// @name Get elements
    /// @{
    /** @brief Get reference to learning rate.*/
    __cuhostdev__ constexpr double & learning_rate(void) noexcept { return this->learning_rate_; }
    /** @brief Get constant reference to learning rate.*/
    __cuhostdev__ constexpr const double & learning_rate(void) const noexcept { return this->learning_rate_; }
    /** @brief Get reference to gradient norm history.*/
    __cuhostdev__ constexpr floatvec & grad_history(void) noexcept { return this->grad_history_; }
    /** @brief Get constant reference to gradient norm history.*/
    __cuhostdev__ constexpr const floatvec & grad_history(void) const noexcept { return this->grad_history_; }
    /** @brief Get reference to bias.*/
    __cuhostdev__ constexpr double & bias(void) noexcept { return this->bias_; }
    /** @brief Get constant reference to bias.*/
    __cuhostdev__ constexpr const double & bias(void) const noexcept { return this->bias_; }
    /// @}

    /// @name Erase history
    /// @{
    /** @brief Erase train history.*/
    MERLIN_EXPORTS void erase_history(void) noexcept;
#ifdef __NVCC__
    /** @brief Erase train history GPU.*/
    __cudevice__ void erase_history_gpu(std::uint64_t thread_idx, std::uint64_t block_size) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread = 1) noexcept;
#ifdef __NVCC__
    /** @brief Update model by gradient value of current thread.*/
    __cudevice__ void update_gpu(candy::Model * p_model, floatvec * p_gradient, void * share_ptr,
                                 std::uint64_t thread_idx, std::uint64_t block_size) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Create an object on GPU by the GPU.
     *  @details Create object by GPU allow register v-table on the GPU, which is required for calling virtual
     *  functions. This function is synchronous.
     */
    candy::Optimizer * new_gpu(void) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the object.*/
    std::uint64_t sharedmem_size(void) const noexcept { return 2 * sizeof(double) + sizeof(double *); }
#ifdef __NVCC__
    /** @brief Copy necessary data to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param data_ptr Memory region where the data of the new created optimizer is stored.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_data_by_block(void * data_ptr, std::uint64_t thread_idx, std::uint64_t block_size);
    /** @brief Copy data to a pre-allocated memory region by a single GPU threads.
     *  @param data_ptr Pre-allocated pointer to memory region storing data.
     */
    __cudevice__ void * copy_data_by_thread(void * data_ptr);
#endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    __cuhostdev__ ~AdaGrad(void) {}
    /// @}

  protected:
    /** @brief Initial learning rate.*/
    double learning_rate_;
    /** @brief Bias to prevent division error.*/
    double bias_;

  private:
    /** @brief Sum of squares of gradients in history.*/
    floatvec grad_history_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_ADAGRAD_HPP_
