// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_GRADIENT_HPP_
#define MERLIN_CANDY_GRADIENT_HPP_

#include <string>       // std::string
#include <type_traits>  // std::add_pointer

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Gradient
#include "merlin/candy/model.hpp"        // merlin::candy::Model
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::floatvec

namespace merlin {

namespace candy {

/** @brief Loss function used for training canonical model.*/
enum class TrainMetric {
    /** @brief Relative square error, skipping all data points that are not normal (``0``, ``inf`` or ``nan``).*/
    RelativeSquare = 0x00,
    /** @brief Absolute square error, skipping all data points that are not finite (``inf`` or ``nan``).*/
    AbsoluteSquare = 0x01,
};

/** @brief Function type calculating gradient.*/
using GradientCalc = std::add_pointer<void(const candy::Model &, const array::NdData &, floatvec &,
                                           std::uint64_t, std::uint64_t, intvec &, floatvec &)>::type;

/** @brief Calculate gradient of a model based on relative square metric.
 *  @param model Model on which the gradient is calculated.
 *  @param train_data Data to calculate loss function.
 *  @param gradient Vector storing the resulted gradient.
 *  @param thread_idx Index of the current thread calculating the gradient.
 *  @param n_threads Number of threads calculating the gradient.
 *  @param index_mem Cache memory foreach thread to calculate the gradient, should be at least ``std::uint64_t[ndim]``.
 *  @param parallel_mem Cache memory storing result by each thread before accumulated to the gradient vector.
 */
__cuhostdev__ void relquare_grad(const candy::Model & model, const array::NdData & train_data, floatvec & gradient,
                                 std::uint64_t thread_idx, std::uint64_t n_threads, intvec & index_mem,
                                 floatvec & parallel_mem) noexcept;

}  // namespace candy

/** @brief Model gradient.*/
class candy::Gradient {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Gradient(void) {}
    /** @brief Assignment from initialized members.*/
    __cuhostdev__ Gradient(double * data, candy::Model * model_ptr, candy::TrainMetric train_metric) :
    model_ptr_(model_ptr), train_metric_(train_metric) {
        this->value_.assign(data, model_ptr->num_params());
    }
    /// @}

    /// @name Check
    /// @{
    /** @brief Check if the gradient is initialize and compatible with the assigned model.*/
    __cuhostdev__ bool check(void) const {
        return (this->model_ptr_ != nullptr) && (this->value_.size() == this->model_ptr_->num_params());
    }
    /// @}

    /// @name Calculation
    /// @{
    /** @brief Calculate gradient from data in parallel section.
     *  @param train_data Data to train the model.
     *  @param thread_idx Index of the current thread calculating the gradient.
     *  @param n_threads Number of threads calculating the gradient.
     *  @param index_mem Cache memory foreach thread, should be at least ``std::uint64_t[ndim]``.
     *  @param parallel_mem Cache memory storing result by each thread before accumulated to the gradient vector.
     */
    MERLIN_EXPORTS void calc(const array::Array & train_data, std::uint64_t thread_idx,
                             std::uint64_t n_threads, intvec & index_mem, floatvec & parallel_mem) noexcept;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Gradient(void);
    /// @}


  protected:
    /** @brief Pointer to canonical model.*/
    candy::Model * model_ptr_ = nullptr;
    /** @brief Gradient of canonical model with respect to a data.*/
    floatvec value_;
    /** @brief Training metric.*/
    candy::TrainMetric train_metric_ = candy::TrainMetric::RelativeSquare;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_GRADIENT_HPP_
