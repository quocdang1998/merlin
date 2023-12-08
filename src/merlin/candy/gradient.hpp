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
enum class TrainMetric : unsigned int {
    /** @brief Relative square error, skipping all data points that are not normal (``0``, ``inf`` or ``nan``).*/
    RelativeSquare = 0x00,
    /** @brief Absolute square error, skipping all data points that are not finite (``inf`` or ``nan``).*/
    AbsoluteSquare = 0x01
};

/** @brief Function type calculating gradient.*/
using GradientCalc = std::add_pointer<void(const candy::Model &, const array::NdData &, floatvec &, std::uint64_t,
                                           std::uint64_t, std::uint64_t *)>::type;

/** @brief Calculate gradient of a model based on relative square metric.
 *  @param model Model on which the gradient is calculated.
 *  @param train_data Data to calculate loss function.
 *  @param gradient Vector storing the resulted gradient.
 *  @param thread_idx Index of the current thread calculating the gradient.
 *  @param n_threads Number of threads calculating the gradient.
 *  @param cache_mem Cache memory foreach thread to calculate the gradient, should be at least
 *  ``std::uint64_t[n_threads*ndim]``.
 */
__cuhostdev__ void rlsquare_grad(const candy::Model & model, const array::NdData & train_data, floatvec & gradient,
                                 std::uint64_t thread_idx, std::uint64_t n_threads, std::uint64_t * cache_mem) noexcept;

/** @brief Calculate gradient of a model based on absolute square metric.
 *  @param model Model on which the gradient is calculated.
 *  @param train_data Data to calculate loss function.
 *  @param gradient Vector storing the resulted gradient.
 *  @param thread_idx Index of the current thread calculating the gradient.
 *  @param n_threads Number of threads calculating the gradient.
 *  @param cache_mem Cache memory foreach thread to calculate the gradient, should be at least
 *  ``std::uint64_t[n_threads*ndim]``.
 */
__cuhostdev__ void absquare_grad(const candy::Model & model, const array::NdData & train_data, floatvec & gradient,
                                 std::uint64_t thread_idx, std::uint64_t n_threads, std::uint64_t * cache_mem) noexcept;

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

    /// @name Get members
    /// @{
    /** @brief Get reference to gradient vector.*/
    __cuhostdev__ floatvec & value(void) noexcept { return this->value_; }
    /** @brief Get constant reference to gradient vector.*/
    __cuhostdev__ const floatvec & value(void) const noexcept { return this->value_; }
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
    /** @brief Calculate gradient from data in CPU parallel section.
     *  @param train_data Data to train the model.
     *  @param thread_idx Index of the current thread calculating the gradient.
     *  @param n_threads Number of threads calculating the gradient.
     *  @param cache_mem Cache memory foreach thread to calculate the gradient, should be at least
     *  ``std::uint64_t[n_threads*ndim]``.
     */
    MERLIN_EXPORTS void calc_by_cpu(const array::Array & train_data, std::uint64_t thread_idx, std::uint64_t n_threads,
                                    std::uint64_t * cache_mem) noexcept;
#ifdef __NVCC__
    /** @brief Calculate gradient from data in GPU parallel section.
     *  @param train_data Data to train the model.
     *  @param thread_idx Index of the current thread calculating the gradient.
     *  @param n_threads Number of threads calculating the gradient.
     *  @param cache_mem Cache memory foreach thread to calculate the gradient, should be at least
     *  ``std::uint64_t[n_threads*ndim]``.
     */
    __cudevice__ void calc_by_gpu(const array::Parcel & train_data, std::uint64_t thread_idx, std::uint64_t n_threads,
                                  std::uint64_t * cache_mem) noexcept;
#endif  // __NVCC__
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
