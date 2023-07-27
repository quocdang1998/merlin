// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_
#define MERLIN_CANDY_OPTMZ_GRAD_DESCENT_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/candy/optmz/declaration.hpp"  // merlin::candy::optmz::GradDescent
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
    MERLIN_EXPORTS void update_cpu(candy::Model & model, double gradient, std::uint64_t i_param,
                                   std::uint64_t param_dim, std::uint64_t param_index, std::uint64_t param_rank);
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
    /** @brief Create an object on GPU by the GPU.
     *  @details Create object by GPU allow register v-table on the GPU, which is required for calling virtual
     *  functions. This function is synchronous.
     */
    MERLIN_EXPORTS static candy::optmz::GradDescent * new_gpu(double learning_rate = 0.5);
    /** @brief Destroy an object by GPU.*/
    MERLIN_EXPORTS static void delete_gpu(candy::optmz::GradDescent * p_optimizer);
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
