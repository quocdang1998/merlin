#ifndef MERLIN_CANDY_TRAINER_HPP_
#define MERLIN_CANDY_TRAINER_HPP_

#include <utility>  // std::swap, std::move

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Trainer
#include "merlin/candy/model.hpp"        // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"    // merlin::candy::Optimizer
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"       // merlin::ProcessorType, merlin::Synchronizer

namespace merlin {

// Utility
// -------

namespace candy {

/** @brief Train a model using CPU parallelism.*/
void train_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data,
                  candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric, std::uint64_t rep,
                  double threshold, std::uint64_t n_threads);

/** @brief Train a model using GPU parallelism.*/
void train_by_gpu(candy::Model * p_model, const array::Parcel * p_data, candy::Optimizer * p_optimizer,
                  candy::TrainMetric metric, std::uint64_t rep, std::uint64_t n_threads, double threshold,
                  std::uint64_t shared_mem_size, cuda::Stream & stream);

/** @brief Calculate error using CPU parallelism.*/
void error_by_cpu(std::future<void> && synch, candy::Model * p_model, const array::Array * p_data, double * p_rmse,
                  double * p_rmae, std::uint64_t n_threads);

/** @brief Calculate error using GPU parallelism.*/
void error_by_gpu(candy::Model * p_model, const array::Parcel * p_data, double * p_rmse, double * p_rmae,
                  std::uint64_t n_threads, std::uint64_t shared_mem_size, cuda::Stream & stream);

}  // namespace candy

// Trainer
// -------

/** @brief Launch a train process on Candecomp model asynchronously.*/
class candy::Trainer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Trainer(void) = default;
    /** @brief Constructor a trainer.
     *  @warning This function will lock the mutex in GPU mode.
     *  @param model Candecomp model.
     *  @param optimizer Gradient method to train the model.
     *  @param processor Type of processor to launch the training process.
     */
    MERLIN_EXPORTS Trainer(const candy::Model & model, const candy::Optimizer & optimizer,
                           ProcessorType processor = ProcessorType::Cpu);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Trainer(const candy::Trainer & src) = delete;
    /** @brief Copy assignment.*/
    candy::Trainer & operator=(const candy::Trainer & src) = delete;
    /** @brief Move constructor.*/
    Trainer(candy::Trainer && src) :
    model_(std::move(src.model_)), optmz_(std::move(src.optmz_)), synch_(std::move(src.synch_)) {
        this->cpu_grad_mem_ = std::exchange(src.cpu_grad_mem_, nullptr);
    }
    /** @brief Move assignment.*/
    candy::Trainer & operator=(candy::Trainer && src) {
        this->model_ = std::move(src.model_);
        this->optmz_ = std::move(src.optmz_);
        this->synch_ = std::move(src.synch_);
        std::swap(this->cpu_grad_mem_, src.cpu_grad_mem_);
        return *this;
    }
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get constant reference the current model.*/
    const candy::Model & model(void) const noexcept { return this->model_; };
    /** @brief Get constant reference the optimizer.*/
    const candy::Optimizer & optmz(void) const noexcept { return this->optmz_; };
    /** @brief Get GPU ID on which the training algorithm will be executed.*/
    constexpr unsigned int gpu_id(void) const noexcept {
        if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(this->synch_.synchronizer))) {
            return stream_ptr->get_gpu().id();
        }
        return static_cast<unsigned int>(-1);
    }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->synch_.synchronizer.index() == 1); }
    /// @}

    /// @name Train CP model based on gradient descent for a given threshold
    /// @{
    /** @brief Update CP model according to gradient on CPU.
     *  @details Update CP model for a certain number of iterations, and check if the relative error after the training
     *  process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
     *  again and check.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @param data Data to train the model.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param n_threads Number of parallel threads for training the model.
     *  @param metric Training metric for the model.
     */
    MERLIN_EXPORTS void update_cpu(const array::Array & data, std::uint64_t rep, double threshold,
                                   std::uint64_t n_threads = 1,
                                   candy::TrainMetric metric = candy::TrainMetric::RelativeSquare);
    /** @brief Update CP model according to gradient on GPU.
     *  @details Update CP model for a certain number of iterations, and check if the relative error after the training
     *  process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
     *  again and check.
     *
     *  This function is asynchronous. To get the model after trained, remember to synchronize the object first.
     *  @warning This function will lock the mutex.
     *  @param data Data to train the model.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param n_threads Number of parallel threads for training the model.
     *  @param metric Training metric for the model.
     */
    MERLIN_EXPORTS void update_gpu(const array::Parcel & data, std::uint64_t rep, double threshold,
                                   std::uint64_t n_threads = 32,
                                   candy::TrainMetric metric = candy::TrainMetric::RelativeSquare);
    /// @}

    /// @name Error metric
    /// @{
    /** @brief Get the RMSE and RMAE error with respect to a given dataset by CPU.
     *  @details Asynchronously calculate RMSE and RMAE, and store it into 2 variables.
     */
    MERLIN_EXPORTS void error_cpu(const array::Array & data, double & rmse, double & rmae, std::uint64_t n_threads = 1);
    /** @brief Get the RMSE and RMAE error with respect to a given dataset by GPU.
     *  @details Asynchronously calculate RMSE and RMAE, and store it into 2 variables.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void error_gpu(const array::Parcel & data, double & rmse, double & rmae,
                                  std::uint64_t n_threads = 32);
    /// @}

    /// @name Synchronization
    /// @{
    /** @brief Force the current CPU to wait until all asynchronous tasks have finished.*/
    void synchronize(void) { this->synch_.synchronize(); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.
     *  @warning This function will lock the mutex in GPU mode.
     */
    MERLIN_EXPORTS ~Trainer(void);
    /// @}

  private:
    /** @brief Candecomp model.*/
    candy::Model model_;
    /** @brief Optimizer.*/
    candy::Optimizer optmz_;
    /** @brief Synchronizer.*/
    Synchronizer synch_;

    /** @brief Memory for storing gradient (only for CPU launch).*/
    double * cpu_grad_mem_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAINER_HPP_
