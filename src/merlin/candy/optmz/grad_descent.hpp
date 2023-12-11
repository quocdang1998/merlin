// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
#define MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_

#include "merlin/candy/declaration.hpp"        // merlin::candy::Gradient, merlin::candy::Model
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/cuda_interface.hpp"           // __cudevice__
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

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    GradDescent(const candy::optmz::GradDescent & src) = default;
    /** @brief Copy assignment.*/
    candy::optmz::GradDescent & operator=(const candy::optmz::GradDescent & src) = default;
    /** @brief Move constructor.*/
    GradDescent(candy::optmz::GradDescent && src) = default;
    /** @brief Move assignment.*/
    candy::optmz::GradDescent & operator=(candy::optmz::GradDescent && src) = default;
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
#endif  // __NVCC__
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Learning rate.*/
    double learning_rate;
    /// @}
};

#ifdef __NVCC__

namespace candy::optmz {

/** @brief Copy GradDescent object to a pre-allocated memory region by a single GPU threads.
 *  @param src_ptr Memory region where the object resides.
 *  @param dest_ptr Memory region where the object is copied to.
 *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
 */
__cudevice__ void * copy_grad_descent_by_thread(void * dest_ptr, const void * src_ptr, void * dynamic_data_ptr);

}  // namespace candy::optmz

#endif

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_