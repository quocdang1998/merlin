// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_TRAIN_GPU_TRAINER_HPP_
#define MERLIN_CANDY_TRAIN_GPU_TRAINER_HPP_

#include <cstddef>  // nullptr
#include <vector>   // std::vector
#include <utility>  // std::move, std::swap

#include "merlin/array/declaration.hpp"         // merlin::array::Parcel
#include "merlin/candy/trial_policy.hpp"        // merlin::candy::TrialPolicy
#include "merlin/candy/train/declaration.hpp"   // merlin::candy::train::GpuTrainer
#include "merlin/candy/train/trainer_base.hpp"  // merlin::candy::train::TrainerBase
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"              // merlin::Synchronizer

namespace merlin {

// Kernels
// -------

namespace candy::train {

/** @brief Launch CUDA kernel dry-running.*/
void launch_dry_run(candy::Model * p_model, candy::Optimizer * p_optmz, const array::Parcel * p_data,
                    std::uint64_t * p_cases, double * p_error, std::uint64_t * p_count, std::uint64_t size,
                    candy::TrialPolicy policy, candy::TrainMetric metric, std::uint64_t block_size,
                    std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);

/** @brief Launch CUDA kernel training a model until a given threshold is met.*/
void launch_update_until(candy::Model * p_model, candy::Optimizer * p_optmz, const array::Parcel * p_data,
                         std::uint64_t size, std::uint64_t rep, double threshold, std::uint64_t block_size,
                         candy::TrainMetric metric, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);

/** @brief Launch CUDA kernel training a model for a fixed number of iterations.*/
void launch_update_for(candy::Model * p_model, candy::Optimizer * p_optmz, const array::Parcel * p_data,
                       std::uint64_t size, std::uint64_t max_iter, std::uint64_t block_size, candy::TrainMetric metric,
                       std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);

/** @brief Launch CUDA kernel reconstructing data.*/
void launch_reconstruct(candy::Model * p_model, array::Parcel * p_data, std::uint64_t size, std::uint64_t block_size,
                        std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);

/** @brief Launch CUDA kernel calculating error.*/
void launch_get_error(candy::Model * p_model, array::Parcel * p_data, double * p_error, std::uint64_t size,
                      std::uint64_t block_size, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr);

}  // namespace candy::train

// GpuTrainer
// ----------

/** @brief Trainer using GPU parallelism.*/
class candy::train::GpuTrainer : public candy::train::TrainerBase {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    GpuTrainer(void) = default;
    /** @brief Constructor from the total number of elements.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS GpuTrainer(std::uint64_t capacity, Synchronizer & synch);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    GpuTrainer(const candy::train::GpuTrainer & src) = delete;
    /** @brief Copy assignment.*/
    candy::train::GpuTrainer & operator=(const candy::train::GpuTrainer & src) = delete;
    /** @brief Move constructor.*/
    GpuTrainer(candy::train::GpuTrainer && src) : candy::train::TrainerBase(std::move(src)) {
        std::swap(this->p_model_vectors_, src.p_model_vectors_);
        std::swap(this->p_optimizer_dynamic_, src.p_optimizer_dynamic_);
        std::swap(this->p_data_, src.p_data_);
        std::swap(this->shared_mem_size_, src.shared_mem_size_);
    }
    /** @brief Move assignment.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS candy::train::GpuTrainer & operator=(candy::train::GpuTrainer && src);
    /// @}

    /// @name Add elements
    /// @{
    /** @brief Add a model to trainer.
     *  @details Add or modify the model assigned to an ID.
     *  @warning This function will lock the mutex.
     *  @param name Name (ID) assigned to the model.
     *  @param model Model to add.
     */
    MERLIN_EXPORTS void set_model(const std::string & name, const candy::Model & model);
    /** @brief Add a optimizer to trainer.
     *  @details Add or modify the optimizer assigned to an ID.
     *  @warning This function will lock the mutex.
     *  @param name Name (ID) assigned to the model.
     *  @param optmz Optimizer to add.
     */
    MERLIN_EXPORTS void set_optmz(const std::string & name, const candy::Optimizer & optmz);
    /** @brief Assign data to an ID.
     *  @details Assign data to train to an ID.
     *  @note The class does not take ownership of the data. Remember to keep the data alive until the class finished
     *  its job.
     *  @warning This function will lock the mutex.
     *  @param name Name (ID) assigned to the model.
     *  @param data Data to add.
     */
    MERLIN_EXPORTS void set_data(const std::string & name, const array::Parcel & data);
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get copy to a model.
     *  @note This function is synchronized with the calling CPU thread. Only invoke it after all training processes are
     *  finished to avoid data race.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS candy::Model get_model(const std::string & name);
    /// @}

    /// @name Training
    /// @{
    /** @brief Dry-run.
     *  @details Perform the gradient descent algorithm for a given number of iterations without updating the model. The
     *  iterative process will automatically stop when the RMSE after the update is larger than before.
     *  @warning This function will lock the mutex.
     *  @param tracking_map Map to vectors storing the errors per iteration for each case and the number of iterations
     *  performed before halting the run. The size of each vector must be bigger or equal to ``policy.sum()``.
     *  @param policy Number of steps for each stage of the dry-run.
     *  @param block_size Number of parallel CUDA threads per block.
     *  @param metric Training metric for the model.
     */
    MERLIN_EXPORTS void dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                                candy::TrialPolicy policy = candy::TrialPolicy(), std::uint64_t block_size = 32,
                                candy::TrainMetric metric = candy::TrainMetric::RelativeSquare);
    /** @brief Update the CP model according to the gradient using GPU parallelism until a specified threshold is met.
     *  @details Update CP model for a certain number of iterations, and check if the relative error after the training
     *  process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
     *  again and check.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @warning This function will lock the mutex.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param block_size Number of parallel CUDA threads per block.
     *  @param metric Training metric for the model.
     *  @param export_result Must be set to ``false``.
     */
    MERLIN_EXPORTS void update_until(std::uint64_t rep, double threshold, std::uint64_t block_size = 32,
                                     candy::TrainMetric metric = candy::TrainMetric::RelativeSquare,
                                     bool export_result = false);
    /** @brief Update CP model according to gradient using GPU for a given number of iterations.
     *  @details Update CP model for a certain number of iterations.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @warning This function will lock the mutex.
     *  @param max_iter Max number of iterations.
     *  @param block_size Number of parallel CUDA threads per block.
     *  @param metric Training metric for the model.
     *  @param export_result Must be set to ``false``.
     */
    MERLIN_EXPORTS void update_for(std::uint64_t max_iter, std::uint64_t block_size = 32,
                                   candy::TrainMetric metric = candy::TrainMetric::RelativeSquare,
                                   bool export_result = false);
    /// @}

    /// @name Reconstruction
    /// @{
    /** @brief Reconstruct a whole multi-dimensional data from the model using GPU parallelism.
     *  @param rec_data_map Map to pointers to reconstructed data.
     *  @param block_size Number of parallel CUDA threads per block.
     *   @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void reconstruct(const std::map<std::string, array::Parcel *> & rec_data_map,
                                    std::uint64_t block_size = 32);
    /** @brief Get the RMSE and RMAE error with respect to the training data.
     *  @param error_map Map of pointers to RMSE and RMAE for each case.
     *  @param block_size Number of parallel CUDA threads per block.
     *   @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                                  std::uint64_t block_size = 32);
    /// @}

    /// @name Export models
    /// @{
    /** @brief Export all models to output directory.
     *  @note This operation is synchronized with the calling CPU thread. Only invoke it after all training processes
     *  are finished to avoid data race.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void export_models(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.
     *   @warning This function will lock the mutex.
     */
    ~GpuTrainer(void) { this->free_memory(); }
    /// @}

  protected:
    /** @brief Pointers to parameter vectors of each model on GPUs.*/
    std::vector<double *> p_model_vectors_;
    /** @brief Pointers to dynamic data of each optimizer on GPUs.*/
    std::vector<double *> p_optimizer_dynamic_;
    /** @brief Pointers to data of each model.*/
    array::Parcel * p_data_ = nullptr;
    /** @brief Max shared memory size required.*/
    std::array<std::uint64_t, 3> shared_mem_size_{0, 0, 0};

  private:
    /** @brief Free data.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void free_memory(void);
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAIN_GPU_TRAINER_HPP_
