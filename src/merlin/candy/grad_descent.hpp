// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_GRAD_DESCENT_HPP_
#define MERLIN_CANDY_GRAD_DESCENT_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

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
class candy::GradDescent : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from learning rate.*/
    GradDescent(double learning_rate = 0.5) : learning_rate_(learning_rate) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    GradDescent(const candy::GradDescent & src) = default;
    /** @brief Copy assignment.*/
    candy::GradDescent & operator=(const candy::GradDescent & src) = default;
    /** @brief Move constructor.*/
    GradDescent(candy::GradDescent && src) = default;
    /** @brief Move assignment.*/
    candy::GradDescent & operator=(candy::GradDescent && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const floatvec & gradient);
    #ifdef __NVCC__
    /** @brief Update model by gradient on GPU.*/
    __cudevice__ void update_gpu(candy::Model * p_model, const double * p_gradient, std::uint64_t size);
    #endif  // __NVCC__
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get reference to learning rate.*/
    __cuhostdev__ constexpr double & learning_rate(void) noexcept {return this->learning_rate_;}
    /** @brief Get constant reference to learning rate.*/
    __cuhostdev__ constexpr const double & learning_rate(void) const noexcept {return this->learning_rate_;}
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the object and its data.*/
    std::uint64_t malloc_size(void) const {return sizeof(candy::GradDescent);}
    /** @brief Copy data from CPU to a pre-allocated memory on GPU.
     *  @details The meta-data should be to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param next_ptr Pointer to a pre-allocated GPU memory for storing next object's meta-data.
     *  @param stream_ptr Pointer to CUDA sytream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Optimizer * gpu_ptr, void * next_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~GradDescent(void);
    /// @}

  protected:
    /** @brief Learning rate.*/
    double learning_rate_;
};

// AdaGrad
// -------

/** @brief %Optimizer by adaptive gradient method.*/
class candy::AdaGrad : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from learning rate.*/
    AdaGrad(double learning_rate = 0.5, double bias = 1.0e-8) : learning_rate_(learning_rate), bias_(bias) {}
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    AdaGrad(const candy::AdaGrad & src) = default;
    /** @brief Copy assignment.*/
    candy::AdaGrad & operator=(const candy::AdaGrad & src) = default;
    /** @brief Move constructor.*/
    AdaGrad(candy::AdaGrad && src) = default;
    /** @brief Move assignment.*/
    candy::AdaGrad & operator=(candy::AdaGrad && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const floatvec & gradient);
    #ifdef __NVCC__
    /** @brief Update model by gradient on GPU.*/
    __cudevice__ void update_gpu(candy::Model * p_model, const double * p_gradient, std::uint64_t size);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~AdaGrad(void);
    /// @}

  protected:
    /** @brief Initial learning rate.*/
    double learning_rate_;
    /** @brief Bias to prevent division error.*/
    double bias_;

  private:
    /** @brief Cumulative gradient norm.*/
    floatvec cumulative_gradient_norm_;
};

// ADAM
// ----

/** @brief %Optimizer by adaptive estimates of lower-order moments method.*/
class candy::Adam : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from learning rate.*/
    MERLIN_EXPORTS Adam(double learning_rate = 0.5, double beta_m = 0.9, double beta_v = 0.999, double bias = 1.0e-8);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Adam(const candy::Adam & src) = default;
    /** @brief Copy assignment.*/
    candy::Adam & operator=(const candy::Adam & src) = default;
    /** @brief Move constructor.*/
    Adam(candy::Adam && src) = default;
    /** @brief Move assignment.*/
    candy::Adam & operator=(candy::Adam && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const floatvec & gradient);
    #ifdef __NVCC__
    /** @brief Update model by gradient on GPU.*/
    __cudevice__ void update_gpu(candy::Model * p_model, const double * p_gradient, std::uint64_t size);
    #endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Adam(void);
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

#endif  // MERLIN_CANDY_GRADIENT_DESCENT_HPP_
