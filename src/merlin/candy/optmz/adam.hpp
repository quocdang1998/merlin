// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_ADAM_HPP_
#define MERLIN_CANDY_OPTMZ_ADAM_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/candy/declaration.hpp"        // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"          // merlin::candy::Optimizer
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::Adam
#include "merlin/cuda_interface.hpp"           // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS
#include "merlin/vector.hpp"                   // merlin::floatvec

namespace merlin {

// Adam
// -------

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
class candy::optmz::Adam : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from members.*/
    __cuhostdev__ Adam(double learning_rate, double beta_m, double beta_v, std::uint64_t model_size,
                       double bias = 1.0e-8) :
    learning_rate_(learning_rate), beta_m_(beta_m), beta_v_(beta_v), bias_(bias), register_moments_(2 * model_size) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Adam(const candy::optmz::Adam & src) = default;
    /** @brief Copy assignment.*/
    candy::optmz::Adam & operator=(const candy::optmz::Adam & src) = default;
    /** @brief Move constructor.*/
    Adam(candy::optmz::Adam && src) = default;
    /** @brief Move assignment.*/
    candy::optmz::Adam & operator=(candy::optmz::Adam && src) = default;
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get reference to learning rate.*/
    __cuhostdev__ constexpr double & learning_rate(void) noexcept { return this->learning_rate_; }
    /** @brief Get constant reference to learning rate.*/
    __cuhostdev__ constexpr const double & learning_rate(void) const noexcept { return this->learning_rate_; }
    /** @brief Get reference to decay coefficient of first moment.*/
    __cuhostdev__ constexpr double & beta_m(void) noexcept { return this->beta_m_; }
    /** @brief Get constant reference to decay coefficient of first moment.*/
    __cuhostdev__ constexpr const double & beta_m(void) const noexcept { return this->beta_m_; }
    /** @brief Get reference to decay coefficient of second moment.*/
    __cuhostdev__ constexpr double & beta_v(void) noexcept { return this->beta_v_; }
    /** @brief Get constant reference to decay coefficient of second moment.*/
    __cuhostdev__ constexpr const double & beta_v(void) const noexcept { return this->beta_v_; }
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
    std::uint64_t sharedmem_size(void) const noexcept {
        return 4 * sizeof(double) + sizeof(double *) + sizeof(std::uint64_t);
    }
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
    __cuhostdev__ ~Adam(void) {}
    /// @}

  protected:
    /** @brief Initial learning rate.*/
    double learning_rate_;
    /** @brief First moment decay constant.*/
    double beta_m_;
    /** @brief Second moment decay constant.*/
    double beta_v_;
    /** @brief Bias to prevent division error.*/
    double bias_;

  private:
    /** @brief Values of first and second moments.*/
    floatvec register_moments_;
    /** @brief Time step.*/
    std::uint64_t time_step_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTMZ_ADAM_HPP_
