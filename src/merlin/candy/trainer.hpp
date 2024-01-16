#ifndef MERLIN_CANDY_TRAINER_HPP_
#define MERLIN_CANDY_TRAINER_HPP_

#include <utility>  // std::exchange, std::move

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model, merlin::candy::Optimizer, merlin::candy::Trainer
#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Stream
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"       // merlin::ProcessorType, merlin::Synchronizer
#include "merlin/vector.hpp"             // merlin::intvec

namespace merlin {

// Utility
// -------

namespace candy {

/** @brief Allocate memory on GPU for the trainer.*/
void create_trainer_gpu_ptr(const candy::Model & cpu_model, const array::Array & cpu_data,
                            const candy::Optimizer & cpu_optimizer, candy::Model *& gpu_model,
                            array::NdData *& gpu_data, candy::Optimizer *& gpu_optimizer, array::Parcel *& parcel_data,
                            cuda::Stream & stream);

/** @brief Train a model using CPU parallelism.*/
void train_by_cpu(std::shared_future<void> synch, candy::Model * p_model, array::Array * p_data,
                  candy::Optimizer * p_optimizer, double * cpu_grad_mem, candy::TrainMetric metric, std::uint64_t rep,
                  double threshold, std::uint64_t n_threads, intvec * p_cache_mem);

/** @brief Train a model using GPU parallelism.*/
void train_by_gpu(candy::Model * p_model, array::Parcel * p_data, candy::Optimizer * p_optimizer,
                  candy::TrainMetric metric, std::uint64_t rep, std::uint64_t n_threads, std::uint64_t ndim,
                  double threshold, std::uint64_t shared_mem_size, cuda::Stream & stream);

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
     *  @param model Candecomp model.
     *  @param data Data to be fitted by the model.
     *  @param optimizer Gradient method to train the model.
     *  @param processor Type of processor to launch the training process.
     */
    MERLIN_EXPORTS Trainer(const candy::Model & model, array::Array && data, const candy::Optimizer & optimizer,
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
    ndim_(src.ndim_),
    synch_(std::move(src.synch_)),
    cpu_cache_mem_(src.cpu_cache_mem_),
    shared_mem_size_(src.shared_mem_size_) {
        this->p_model_ = std::exchange(src.p_model_, nullptr);
        this->p_data_ = std::exchange(src.p_data_, nullptr);
        this->p_optmz_ = std::exchange(src.p_optmz_, nullptr);
        this->cpu_grad_mem_ = std::exchange(src.cpu_grad_mem_, nullptr);
        this->p_parcel_ = std::exchange(src.p_parcel_, nullptr);
    }
    /** @brief Move assignment.*/
    candy::Trainer & operator=(candy::Trainer && src) {
        this->p_model_ = std::exchange(src.p_model_, nullptr);
        this->p_data_ = std::exchange(src.p_data_, nullptr);
        this->p_optmz_ = std::exchange(src.p_optmz_, nullptr);
        this->ndim_ = src.ndim_;
        this->synch_ = std::move(src.synch_);
        this->cpu_grad_mem_ = std::exchange(src.cpu_grad_mem_, nullptr);
        this->cpu_cache_mem_ = std::move(src.cpu_cache_mem_);
        this->shared_mem_size_ = src.shared_mem_size_;
        this->p_parcel_ = std::exchange(src.p_parcel_, nullptr);
        return *this;
    }
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get a copy to the current CP model.*/
    MERLIN_EXPORTS candy::Model get_model(void) const;
    /** @brief Get GPU ID on which the memory is allocated.*/
    constexpr unsigned int gpu_id(void) const noexcept {
        if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(this->synch_.synchronizer))) {
            return stream_ptr->get_gpu().id();
        }
        return static_cast<unsigned int>(-1);
    }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->synch_.synchronizer.index() == 1); }
    /// @}

    /// @name Train CP model based on gradient descent
    /// @{
    /** @brief Update CP model according to gradient.
     *  @param rep Number of times to repeat the gradient descent update in each step.
     *  @param threshold Threshold to stop the training process.
     *  @param n_threads Number of parallel threads for training the model.
     *  @param metric Training metric for the model.
     */
    MERLIN_EXPORTS void update(std::uint64_t rep, double threshold, std::uint64_t n_threads = 1,
                               candy::TrainMetric metric = candy::TrainMetric::RelativeSquare);
    /// @}

    /// @name Synchronization
    /// @{
    /** @brief Force the current CPU to wait until all asynchronous tasks have finished.*/
    void synchronize(void) { this->synch_.synchronize(); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Trainer(void);
    /// @}

  private:
    /** @brief Pointer to Candecomp model.*/
    candy::Model * p_model_ = nullptr;
    /** @brief Pointer to train data.*/
    array::NdData * p_data_ = nullptr;
    /** @brief Pointer to optimizer.*/
    candy::Optimizer * p_optmz_ = nullptr;

    /** @brief Number of dimension.*/
    std::uint64_t ndim_;
    /** @brief Synchronizer.*/
    Synchronizer synch_;

    /** @brief Memory for storing gradient (only for CPU launch).*/
    double * cpu_grad_mem_ = nullptr;
    /** @brief Cache memory for calculating the gradient.*/
    intvec cpu_cache_mem_;

    /** @brief Size of shared memory to reserve (only for GPU launch).*/
    std::uint64_t shared_mem_size_ = 0;
    /** @brief GPU data (only for GPU launch).*/
    array::Parcel * p_parcel_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAINER_HPP_
