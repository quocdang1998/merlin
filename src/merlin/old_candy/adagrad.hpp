// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_ADAGRAD_HPP_
#define MERLIN_CANDY_ADAGRAD_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

// AdaGrad
// -------

/** @brief %Optimizer by adaptive gradient method.*/
class candy::AdaGrad : public candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from learning rate.*/
    AdaGrad(double learning_rate = 0.5, double bias = 1.0e-8) : learning_rate_(learning_rate), bias_(bias) {}
    #ifdef __NVCC__
    /** @brief Constructor on GPU.*/
    __cudevice__ AdaGrad(double learning_rate, double bias, std::uint64_t gradient_size);
    #endif  // __NVCC__
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

    /// @name GPU related features
    /// @{
    /** @brief Create an object on GPU by the GPU.
     *  @details Create object by GPU allow register v-table on the GPU, which is required for calling virtual
     *  functions. This function is synchronous.
     */
    MERLIN_EXPORTS static candy::AdaGrad * create_object_on_gpu(double learning_rate, double bias,
                                                                std::uint64_t gradient_size);
    /** @brief Destroy an object by GPU.*/
    MERLIN_EXPORTS static void delete_object_on_gpu(candy::AdaGrad * p_optimizer);
    #ifdef __NVCC__
    /** @brief Copy data to shared memory.*/
    __cudevice__ void * copy_to_shared_mem(candy::Optimizer * share_ptr, void * data_ptr) const;
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
    /** @brief Cumulative gradient norm.*/
    floatvec cumulative_gradient_norm_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_ADAGRAD_HPP_
