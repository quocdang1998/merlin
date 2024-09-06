// Copyright 2024 quocdang1998
#ifndef MERLIN_CANDY_TRAIN_TRAINER_BASE_HPP_
#define MERLIN_CANDY_TRAIN_TRAINER_BASE_HPP_

#include <algorithm>  // std::all_of
#include <array>      // std::array
#include <cstddef>    // nullptr
#include <cstdint>    // std::uint64_t
#include <map>        // std::map
#include <string>     // std::string
#include <utility>    // std::exchange, std::move, std::pair, std::swap
#include <vector>     // std::vector

#include "merlin/candy/declaration.hpp"        // merlin::candy::Model, merlin::candy::Optimizer
#include "merlin/candy/train/declaration.hpp"  // merlin::candy::train::TrainerBase
#include "merlin/config.hpp"                   // merlin::Index
#include "merlin/exports.hpp"                  // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"             // merlin::Synchronizer

namespace merlin {

// Helper classes and templates
// ----------------------------

namespace candy::train {

/** @brief Indicator map for ID.*/
using IndicatorMap = std::map<std::string, std::pair<std::uint64_t, std::array<bool, 3>>>;

/** @brief Check if two maps have the same set of keys.*/
template <typename Map1, typename Map2>
bool key_compare(const Map1 & map1, const Map2 & map2) {
    return std::all_of(map1.begin(), map1.end(), [&map2](Map1::const_reference x) { return map2.contains(x.first); });
}

}  // namespace candy::train

// TrainerBase
// -----------

/** @brief Base class for trainer.*/
class candy::train::TrainerBase {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    TrainerBase(void) = default;
    /** @brief Constructor from the capacity.*/
    MERLIN_EXPORTS TrainerBase(std::uint64_t capacity);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    TrainerBase(const candy::train::TrainerBase & src) = delete;
    /** @brief Copy assignment.*/
    candy::train::TrainerBase & operator=(const candy::train::TrainerBase & src) = delete;
    /** @brief Move constructor.*/
    TrainerBase(candy::train::TrainerBase && src) :
    map_(std::move(src.map_)), details_(std::move(src.details_)), export_fnames_(std::move(src.export_fnames_)) {
        std::swap(this->size_, src.size_);
        std::swap(this->capacity_, src.capacity_);
        std::swap(this->p_model_, src.p_model_);
        std::swap(this->p_optmz_, src.p_optmz_);
        std::swap(this->p_synch_, src.p_synch_);
    }
    /** @brief Move assignment.*/
    candy::train::TrainerBase & operator=(candy::train::TrainerBase && src) {
        this->map_ = std::move(src.map_);
        this->size_ = std::exchange(src.size_, 0);
        this->capacity_ = std::exchange(src.capacity_, 0);
        this->p_model_ = std::exchange(src.p_model_, nullptr);
        this->p_optmz_ = std::exchange(src.p_optmz_, nullptr);
        this->details_ = std::move(src.details_);
        this->export_fnames_ = std::move(src.export_fnames_);
        this->p_synch_ = std::exchange(src.p_synch_, nullptr);
        return *this;
    }
    /// @}

    /// @name Add elements
    /// @{
    /** @brief Check if the trainer is filled.*/
    bool is_full(void) { return this->size_ == this->capacity_; }
    /** @brief Add a model to trainer.
     *  @details Add or modify the model assigned to an ID.
     *  @param name Name (ID) assigned to the model.
     *  @param model Model to add.
     */
    virtual void set_model(const std::string & name, const candy::Model & model) = 0;
    /** @brief Add a optimizer to trainer.
     *  @details Add or modify the optimizer assigned to an ID.
     *  @param name Name (ID) assigned to the optimizer.
     *  @param optmz Optimizer to add.
     */
    virtual void set_optmz(const std::string & name, const candy::Optimizer & optmz) = 0;
    /** @brief Add exported names to model.
     *  @details Add or modify the output filename of a model.
     *  @param name Name (ID) assigned to the targeted model.
     *  @param export_fname Filename to be exported.
     */
    MERLIN_EXPORTS void set_export_fname(const std::string & name, const std::string & export_fname);
    /// @}

    /// @name Get elements
    /// @{
    /** @brief Get list of keys.*/
    MERLIN_EXPORTS std::vector<std::string> get_keys(void);
    /** @brief Get index corresponding to key.*/
    std::uint64_t get_index(const std::string & name) { return this->map_.at(name).first; }
    /** @brief Get model shape and rank.*/
    const std::pair<Index, std::uint64_t> & get_model_shape(const std::string & name) {
        return this->details_[this->get_index(name)];
    }
    /** @brief Get reference to the synchronizer.*/
    Synchronizer & get_synch(void) { return *(this->p_synch_); }
    /// @}

    /// @name Check
    /// @{
    /** @brief Check if each key is assigned to a model, an optimizer and a data.*/
    MERLIN_EXPORTS bool is_complete(void);
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
    virtual void dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                         candy::TrialPolicy policy, std::uint64_t n_threads, candy::TrainMetric metric) = 0;
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
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training.
     */
    virtual void update_until(std::uint64_t rep, double threshold, std::uint64_t n_threads, candy::TrainMetric metric,
                              bool export_result) = 0;
    /** @brief Update CP model according to gradient for a given number of iterations.
     *  @details Update CP model for a certain number of iterations.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param max_iter Max number of iterations.
     *  @param n_threads Number of OpenMP threads, or number of CUDA threads per block.
     *  @param metric Training metric for the model.
     *  @param export_result Flag indicate whether to serialize the model right at the end of the training.
     */
    virtual void update_for(std::uint64_t max_iter, std::uint64_t n_threads, candy::TrainMetric metric,
                            bool export_result) = 0;
    /// @}

    /// @name Reconstruction
    /// @{
    /** @brief Get the RMSE and RMAE error with respect to the training data.
     *  @param error_map Map of pointers to RMSE and RMAE for each case.
     *  @param n_threads Number of parallel threads.
     */
    virtual void get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                           std::uint64_t n_threads) = 0;
    /// @}

    /// @name Export models
    /// @{
    /** @brief Export all models to output directory.
     *  @note This operation is synchronized with the calling CPU thread. Only invoke it after all training processes
     *  are finished to avoid data race.
     */
    virtual void export_models(void) = 0;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    virtual ~TrainerBase(void) = default;
    /// @}

  protected:
    /** @brief IDs for each model.*/
    candy::train::IndicatorMap map_;
    /** @brief Number of instances.*/
    std::uint64_t size_ = 0;
    /** @brief Capacity of allocated memory.*/
    std::uint64_t capacity_ = 0;
    /** @brief Vector of models.*/
    candy::Model * p_model_ = nullptr;
    /** @brief Vector of optimizers for each model.*/
    candy::Optimizer * p_optmz_ = nullptr;
    /** @brief Vector recording model shape and rank.*/
    std::vector<std::pair<Index, std::uint64_t>> details_;
    /** @brief Vector of filenames of exported models.*/
    std::vector<std::string> export_fnames_;
    /** @brief Pointer to synchronizer.*/
    Synchronizer * p_synch_ = nullptr;

    /// @name Index manipulation
    /// @{
    /** @brief Get index corresponding to a given key. If the key does not exist, create the new object assigned to the
     *  new key.*/
    MERLIN_EXPORTS std::uint64_t get_index_or_create_key(const std::string & name);
    /** @brief Add model shape and total number of parameters to the detail array.*/
    MERLIN_EXPORTS void update_details(std::uint64_t index, const candy::Model & model);
    /** @brief Check if all models are assigned.*/
    MERLIN_EXPORTS void check_models(void);
    /** @brief Check if all models, optimizers and data are assigned.*/
    MERLIN_EXPORTS void check_complete(void);
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAIN_TRAINER_BASE_HPP_
