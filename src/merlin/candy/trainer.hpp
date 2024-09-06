#ifndef MERLIN_CANDY_TRAINER_HPP_
#define MERLIN_CANDY_TRAINER_HPP_

#include <cstddef>  // nullptr
#include <utility>  // std::exchange, std::swap

#include "merlin/array/declaration.hpp"         // merlin::array::NdData
#include "merlin/candy/declaration.hpp"         // merlin::candy::Trainer, merlin::candy::TrainMetric
#include "merlin/candy/trial_policy.hpp"        // merlin::candy::TrialPolicy
#include "merlin/candy/train/trainer_base.hpp"  // merlin::candy::train::TrainerBase
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"              // merlin::Synchronizer

namespace merlin {

// Trainer
// -------

/** @brief Launch a train process on CP model asynchronously.*/
class candy::Trainer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Trainer(void) = default;
    /** @brief Constructor from the total number of elements.*/
    MERLIN_EXPORTS Trainer(std::uint64_t capacity, Synchronizer & synch);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Trainer(const candy::Trainer & src) = delete;
    /** @brief Copy assignment.*/
    candy::Trainer & operator=(const candy::Trainer & src) = delete;
    /** @brief Move constructor.*/
    Trainer(candy::Trainer && src) { this->p_core_ = std::exchange(src.p_core_, nullptr); }
    /** @brief Move assignment.*/
    candy::Trainer & operator=(candy::Trainer && src) {
        std::swap(this->p_core_, src.p_core_);
        return *this;
    }
    /// @}

    /// @name Add elements
    /// @{
    /** @brief Check if the trainer is filled.*/
    bool is_full(void) { return this->p_core_->is_full(); }
    /** @brief Add a model to trainer.
     *  @details Add or modify the model assigned to an ID.
     *  @param name Name (ID) assigned to the model.
     *  @param model Model to add.
     */
    inline void set_model(const std::string & name, const candy::Model & model) {
        this->p_core_->set_model(name, model);
    }
    /** @brief Add a optimizer to trainer.
     *  @details Add or modify the optimizer assigned to an ID.
     *  @param name Name (ID) assigned to the optimizer.
     *  @param optmz Optimizer to add.
     */
    inline void set_optmz(const std::string & name, const candy::Optimizer & optmz) {
        this->p_core_->set_optmz(name, optmz);
    }
    /** @brief Assign data to an ID.
     *  @details Assign data to train to an ID.
     *  @note The class does not take ownership of the data. Remember to keep the data alive until the class finished
     *  its job.
     *  @param name Name (ID) assigned to the model.
     *  @param data Data to add.
     */
    MERLIN_EXPORTS void set_data(const std::string & name, const array::NdData & data);
    /** @brief Add exported names to model.
     *  @details Add or modify the output filename of a model.
     *  @param name Name (ID) assigned to the targeted model.
     *  @param export_fname Filename to be exported.
     */
    inline void set_export_fname(const std::string & name, const std::string & export_fname) {
        this->p_core_->set_export_fname(name, export_fname);
    }
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get list of keys.*/
    inline std::vector<std::string> get_keys(void) { return this->p_core_->get_keys(); }
    /** @brief Query if the object data is instatialized on CPU or on GPU.*/
    MERLIN_EXPORTS bool on_gpu(void);
    /** @brief Get model shape and rank.*/
    inline const std::pair<Index, std::uint64_t> & get_model_shape(const std::string & name) {
        return this->p_core_->get_model_shape(name);
    }
    /** @brief Get copy to a model.
     *  @note This function is synchronized with the calling CPU thread. Only invoke it after all training processes are
     *  finished to avoid data race.
     */
    MERLIN_EXPORTS candy::Model get_model(const std::string & name);
    /** @brief Get reference to the synchronizer.*/
    inline Synchronizer & get_synch(void) { return this->p_core_->get_synch(); }
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
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     *  @param metric Training metric for the model.
     */
    inline void dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                        candy::TrialPolicy policy = candy::TrialPolicy(), std::uint64_t n_threads = 16,
                        candy::TrainMetric metric = candy::TrainMetric::RelativeSquare) {
        this->p_core_->dry_run(tracking_map, policy, n_threads, metric);
    }
    /** @brief Update the CP model according to the gradient until a specified threshold is met.
     *  @details Update CP model for a certain number of iterations, and check if the relative error after the training
     *  process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
     *  again and check.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     *  @param metric Training metric for the model.
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training (must be
     *  ``false`` in GPU configuration).
     */
    inline void update_until(std::uint64_t rep, double threshold, std::uint64_t n_threads = 16,
                             candy::TrainMetric metric = candy::TrainMetric::RelativeSquare,
                             bool export_result = false) {
        this->p_core_->update_until(rep, threshold, n_threads, metric, export_result);
    }
    /** @brief Update CP model according to gradient for a given number of iterations.
     *  @details Update CP model for a certain number of iterations.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param max_iter Max number of iterations.
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     *  @param metric Training metric for the model.
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training (must be
     *  ``false`` in GPU configuration).
     */
    inline void update_for(std::uint64_t max_iter, std::uint64_t n_threads = 16,
                           candy::TrainMetric metric = candy::TrainMetric::RelativeSquare, bool export_result = false) {
        this->p_core_->update_for(max_iter, n_threads, metric, export_result);
    }
    /// @}

    /// @name Reconstruction
    /// @{
    /** @brief Reconstruct a whole multi-dimensional data from the model.
     *  @param rec_data_map Map to pointers to reconstructed data.
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     */
    MERLIN_EXPORTS void reconstruct(const std::map<std::string, array::NdData *> & rec_data_map,
                                    std::uint64_t n_threads = 16);
    /** @brief Get the RMSE and RMAE error with respect to the training data.
     *  @param error_map Map of pointers to RMSE and RMAE for each case.
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     */
    inline void get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                          std::uint64_t n_threads = 16) {
        this->p_core_->get_error(error_map, n_threads);
    }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~Trainer(void);
    /// @}

  protected:
    /** @brief Pointer to the underlying object.*/
    candy::train::TrainerBase * p_core_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAINER_HPP_
