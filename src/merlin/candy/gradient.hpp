// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_GRADIENT_HPP_
#define MERLIN_CANDY_GRADIENT_HPP_

#include <string>       // std::string
#include <type_traits>  // std::add_pointer

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Gradient
#include "merlin/candy/model.hpp"        // merlin::candy::Model
#include "merlin/config.hpp"             // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::Index, merlin::DoubleVec

namespace merlin {

namespace candy {

/** @brief Function type calculating gradient.*/
using GradientCalc =
    std::add_pointer<double(const candy::Model &, const array::NdData &, std::uint64_t, Index &) noexcept>::type;

/** @brief Function calculating gradient.
 *  @param model Model on which the gradient is calculated.
 *  @param train_data Data to train the model loss function.
 *  @param gradient Vector storing the resulted gradient.
 *  @param metric Metric for calculating the gradient.
 *  @param thread_idx Index of the current thread calculating the gradient.
 *  @param n_threads Number of threads calculating the gradient.
 *  @param index_mem Cache memory storing index foreach thread.
 */
__cuhostdev__ void calc_gradient(const candy::Model & model, const array::NdData & train_data, DoubleVec & gradient,
                                 unsigned int metric, std::uint64_t thread_idx, std::uint64_t n_threads,
                                 Index & index_mem) noexcept;

}  // namespace candy

/** @brief %Model gradient.*/
class candy::Gradient {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Gradient(void) {}
    /** @brief Assignment from initialized members.*/
    __cuhostdev__ Gradient(double * data, std::uint64_t num_params, candy::TrainMetric train_metric) :
    value_(data, num_params, true), train_metric_(train_metric) {}
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to gradient vector.*/
    __cuhostdev__ DoubleVec & value(void) noexcept { return this->value_; }
    /** @brief Get constant reference to gradient vector.*/
    __cuhostdev__ const DoubleVec & value(void) const noexcept { return this->value_; }
    /// @}

    /// @name Calculation
    /// @{
    /** @brief Calculate gradient from data in CPU parallel section.
     *  @param model Canonical model.
     *  @param train_data Data to train the model.
     *  @param index_mem Cache memory storing index foreach thread.
     */
    MERLIN_EXPORTS void calc_by_cpu(candy::Model & model, const array::Array & train_data, Index & index_mem) noexcept;
#ifdef __NVCC__
    /** @brief Calculate gradient from data in GPU parallel section.
     *  @param model Canonical model.
     *  @param train_data Data to train the model.
     *  @param thread_idx Index of the current thread calculating the gradient.
     *  @param n_threads Number of threads calculating the gradient.
     *  @param index_mem Cache memory storing index foreach thread.
     */
    __cudevice__ void calc_by_gpu(candy::Model & model, const array::Parcel & train_data, std::uint64_t thread_idx,
                                  std::uint64_t n_threads, Index & index_mem) noexcept;
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
    __cuhostdev__ ~Gradient(void);
    /// @}

  protected:
    /** @brief %Gradient of canonical model with respect to a data.*/
    DoubleVec value_;
    /** @brief Training metric.*/
    candy::TrainMetric train_metric_ = candy::TrainMetric::RelativeSquare;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_GRADIENT_HPP_
