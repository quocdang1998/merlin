// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
#define MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_

#include "merlin/candy/declaration.hpp"        // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"          // merlin::candy::Optimizer
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/cuda_interface.hpp"           // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS

namespace merlin {

// Gradient descent
// ----------------

/** @brief %Optimizer by stochastic gradient descent method.
 *  @details Each parameter will be updated by the formula:
 *  @f[ p_{t+1} = p_t - \eta \left( \frac{\partial L}{\partial p} \right)_t @f]
 *
 *  , in which @f$ t @f$ is update instance,  @f$ p @f$ is value of parameter, @f$ \eta @f$ is the learning rate, and
 *  @f$ L @f$ is the loss function.
 */
class candy::optmz::GradDescent : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from learning rate.*/
    __cuhostdev__ GradDescent(double learning_rate = 0.5) : learning_rate_(learning_rate) {}
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
    /** @brief Update model by gradient.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread = 1) noexcept;
#ifdef __NVCC__
    /** @brief Update model by gradient value of current thread.*/
    __cudevice__ void update_gpu(candy::Model * p_model, floatvec * p_gradient, void * share_ptr,
                                 std::uint64_t thread_idx, std::uint64_t block_size) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get reference to learning rate.*/
    __cuhostdev__ constexpr double & learning_rate(void) noexcept { return this->learning_rate_; }
    /** @brief Get constant reference to learning rate.*/
    __cuhostdev__ constexpr const double & learning_rate(void) const noexcept { return this->learning_rate_; }
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Create an object on GPU by the GPU.
     *  @details Create object by GPU allow register v-table on the GPU, which is required for calling virtual
     *  functions. This function is synchronous.
     */
    candy::Optimizer * new_gpu(void) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the object.*/
    std::uint64_t sharedmem_size(void) const noexcept { return sizeof(double); }
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
    __cuhostdev__ ~GradDescent(void) {}
    /// @}

  protected:
    /** @brief Learning rate.*/
    double learning_rate_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
