// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_TRAIN_CPU_TRAINER_HPP_
#define MERLIN_CANDY_TRAIN_CPU_TRAINER_HPP_

#include <cstddef>  // nullptr
#include <map>      // std::map
#include <string>   // std::string
#include <vector>   // std::vector

#include "merlin/array/declaration.hpp"         // merlin::array::Array
#include "merlin/candy/trial_policy.hpp"        // merlin::candy::TrialPolicy
#include "merlin/candy/train/declaration.hpp"   // merlin::candy::train::CpuTrainer
#include "merlin/candy/train/trainer_base.hpp"  // merlin::candy::train::TrainerBase
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"              // merlin::Synchronizer

namespace merlin {

// Utility
// -------

namespace candy::train {

/** @brief Dry-run.*/
MERLIN_EXPORTS void run_dry_run(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                candy::Model * p_model, const array::Array ** p_data, candy::Optimizer * p_optimizer,
                                std::vector<std::tuple<std::uint64_t, double *, std::uint64_t *>> && tracker,
                                candy::TrainMetric metric, std::uint64_t n_threads, candy::TrialPolicy policy);

/** @brief Train a model until a given threshold is met.*/
MERLIN_EXPORTS void run_update_until(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                     candy::Model * p_model, const array::Array ** p_data,
                                     candy::Optimizer * p_optimizer, std::uint64_t size, candy::TrainMetric metric,
                                     std::uint64_t rep, double threshold, std::uint64_t n_threads, bool export_result,
                                     const std::string * p_export_fnames);

/** @brief Train a model for a fixed number of iterations.*/
MERLIN_EXPORTS void run_update_for(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                   candy::Model * p_model, const array::Array ** p_data, candy::Optimizer * p_optimizer,
                                   std::uint64_t size, candy::TrainMetric metric, std::uint64_t max_iter,
                                   std::uint64_t n_threads, bool export_result, const std::string * p_export_fnames);

/** @brief Reconstruct data.*/
MERLIN_EXPORTS void run_reconstruct(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                    candy::Model * p_model, std::vector<array::Array *> && p_rec_data,
                                    std::uint64_t size, std::uint64_t n_threads);

/** @brief Calculate error.*/
MERLIN_EXPORTS void run_get_error(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                  candy::Model * p_model, const array::Array ** p_data, std::uint64_t size,
                                  std::vector<std::array<double *, 2>> && p_error, std::uint64_t n_threads);

}  // namespace candy::train

// CpuTrainer
// ----------

/** @brief Trainer using CPU parallelism.*/
class candy::train::CpuTrainer : public candy::train::TrainerBase {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CpuTrainer(void) = default;
    /** @brief Constructor from the total number of elements.*/
    MERLIN_EXPORTS CpuTrainer(std::uint64_t capacity, Synchronizer & synch);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    CpuTrainer(const candy::train::CpuTrainer & src) = delete;
    /** @brief Copy assignment.*/
    candy::train::CpuTrainer & operator=(const candy::train::CpuTrainer & src) = delete;
    /** @brief Move constructor.*/
    CpuTrainer(candy::train::CpuTrainer && src) : candy::train::TrainerBase(std::move(src)) {
        std::swap(this->p_data_, src.p_data_);
    }
    /** @brief Move assignment.*/
    MERLIN_EXPORTS candy::train::CpuTrainer & operator=(candy::train::CpuTrainer && src);
    /// @}

    /// @name Add elements
    /// @{
    /** @brief Add a model to trainer.
     *  @details Add or modify the model assigned to an ID.
     *  @param name Name (ID) assigned to the model.
     *  @param model Model to add.
     */
    MERLIN_EXPORTS void set_model(const std::string & name, const candy::Model & model);
    /** @brief Add a optimizer to trainer.
     *  @details Add or modify the optimizer assigned to an ID.
     *  @param name Name (ID) assigned to the model.
     *  @param optmz Optimizer to add.
     */
    MERLIN_EXPORTS void set_optmz(const std::string & name, const candy::Optimizer & optmz);
    /** @brief Assign data to an ID.
     *  @details Assign data to train to an ID.
     *  @note The class does not take ownership of the data. Remember to keep the data alive until the class finished
     *  its job.
     *  @param name Name (ID) assigned to the model.
     *  @param data Data to add.
     */
    MERLIN_EXPORTS void set_data(const std::string & name, const array::Array & data);
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get reference to a model.*/
    MERLIN_EXPORTS candy::Model & get_model(const std::string & name);
    /** @brief Get reference to an optimizer.*/
    MERLIN_EXPORTS candy::Optimizer & get_optmz(const std::string & name);
    /// @}

    /// @name Training
    /// @{
    /** @brief Dry-run.
     *  @details Perform the gradient descent algorithm for a given number of iterations without updating the model. The
     *  iterative process will automatically stop when the RMSE after the update is larger than before.
     *  @param tracking_map Map to vectors storing the errors per iteration for each case and the number of iterations
     *  performed before halting the run. The size of each vector must be bigger or equal to ``policy.sum()``.
     *  @param policy Number of steps for each stage of the dry-run.
     *  @param n_threads Number of parallel threads.
     *  @param metric Training metric for the model.
     */
    MERLIN_EXPORTS void dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                                candy::TrialPolicy policy = candy::TrialPolicy(), std::uint64_t n_threads = 1,
                                candy::TrainMetric metric = candy::TrainMetric::RelativeSquare);
    /** @brief Update the CP model according to the gradient using CPU parallelism until a specified threshold is met.
     *  @details Update CP model for a certain number of iterations, and check if the relative error after the training
     *  process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
     *  again and check.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param n_threads Number of parallel threads.
     *  @param metric Training metric for the model.
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training.
     */
    MERLIN_EXPORTS void update_until(std::uint64_t rep, double threshold, std::uint64_t n_threads = 1,
                                     candy::TrainMetric metric = candy::TrainMetric::RelativeSquare,
                                     bool export_result = true);
    /** @brief Update CP model according to gradient using CPU for a given number of iterations.
     *  @details Update CP model for a certain number of iterations.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param max_iter Max number of iterations.
     *  @param n_threads Number of parallel threads.
     *  @param metric Training metric for the model.
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training.
     */
    MERLIN_EXPORTS void update_for(std::uint64_t max_iter, std::uint64_t n_threads = 1,
                                   candy::TrainMetric metric = candy::TrainMetric::RelativeSquare,
                                   bool export_result = true);
    /// @}

    /// @name Reconstruction
    /// @{
    /** @brief Reconstruct a whole multi-dimensional data from the model using CPU parallelism.
     *  @param rec_data_map Map to pointers to reconstructed data.
     *  @param n_threads Number of parallel threads for reconstruction.
     */
    MERLIN_EXPORTS void reconstruct(const std::map<std::string, array::Array *> & rec_data_map,
                                    std::uint64_t n_threads = 1);
    /** @brief Get the RMSE and RMAE error with respect to the training data.
     *  @param error_map Map of pointers to RMSE and RMAE for each case.
     *  @param n_threads Number of parallel threads.
     */
    MERLIN_EXPORTS void get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                                  std::uint64_t n_threads = 1);
    /// @}

    /// @name Export models
    /// @{
    /** @brief Export all models to output directory.
     *  @note This operation is synchronized with the calling CPU thread. Only invoke it after all training processes
     *  are finished to avoid data race.
     */
    MERLIN_EXPORTS void export_models(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~CpuTrainer(void);
    /// @}

  protected:
    /** @brief Pointer to data of each model.*/
    std::vector<const array::Array *> p_data_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAIN_CPU_TRAINER_HPP_
