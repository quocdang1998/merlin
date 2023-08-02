// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_LAUNCHER_HPP_
#define MERLIN_CANDY_LAUNCHER_HPP_

#include <cstdint>  // std::uint64_t
#include <future>   // std::future

#include "merlin/array/declaration.hpp"  // merlin::array::NdData, merlin::array::Array, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Launcher, merlin::candy::Model, merlin::candy::Optimizer
#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Stream
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::floatvec, merlin::intvec

namespace merlin {

/** @brief Class launching model training.*/
class candy::Launcher {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Launcher(void) = default;
    /** @brief Constructor from a model and CPU array.
     *  @param model Candecomp model.
     *  @param train_data Data to be fitted by the model.
     *  @param optimizer Gradient method to train the model.
     *  @param n_thread Number of CPU threads.
     */
    MERLIN_EXPORTS Launcher(candy::Model & model, const array::Array & train_data, candy::Optimizer & optimizer,
                            std::uint64_t n_thread = 1);
    /** @brief Constructor from GPU-pre-allocated pointers.
     *  @param p_model Pointer to candecomp model pre-allocated on GPU.
     *  @param p_train_data Pointer to train data pre-allocated on GPU.
     *  @param p_optimizer Pointer to optimizer pre-allocated on GPU.
     *  @param model_size Number of parameter of the model.
     *  @param ndim Number of dimension.
     *  @param share_mem_size Size of share memory of the model, training data and optimizer.
     *  @param block_size Number of threads in the CUDA execution block.
     */
    MERLIN_EXPORTS Launcher(candy::Model * p_model, const array::Parcel * p_train_data, candy::Optimizer * p_optimizer,
                            std::uint64_t model_size, std::uint64_t ndim, std::uint64_t share_mem_size,
                            std::uint64_t block_size = 1);
    /// @}

    /** @brief Check if the processor is a GPU.*/
    bool is_gpu(void) const noexcept { return this->processor_id_ != static_cast<std::uintptr_t>(-1); }

    /** @brief Launch asynchronously the gradient update.
     *  @param rep Number of times to update model parameter.
     */
    MERLIN_EXPORTS void launch_async(std::uint64_t rep = 1);
    /** @brief Synchronize the launch.
     *  @details Force CPU to wait until the launch has finished.
     */
    MERLIN_EXPORTS void synchronize(void);

    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Launcher(void);

  protected:
    /** @brief Pointer to canonical decomposition model.*/
    candy::Model * p_model_ = nullptr;
    /** @brief Pointer to data.*/
    const array::NdData * p_data_ = nullptr;
    /** @brief Optimization algorithm.*/
    candy::Optimizer * p_optimizer_;
    /** @brief Processor ID.
     *  @details Value ``-1`` indicates CPU processor. Positive values represent CUDA context.
     */
    std::uintptr_t processor_id_ = static_cast<std::uintptr_t>(-1);
    /** @brief Number of threads to use.*/
    std::uint64_t n_thread_ = 1;

  private:
    /** @brief Pointer to synchronizer .*/
    void * synchronizer_ = nullptr;
    /** @brief Number of parameters to train in the model.*/
    std::uint64_t model_size_ = 0;
    /** @brief Number of dimension of the model and train data.*/
    std::uint64_t ndim_ = 0;
    /** @brief Size of shared memory.*/
    std::uint64_t shared_mem_size_ = 0;
};

namespace candy {

/** @brief Convert contiguous index to ndim index with a dimension fixed.*/
__cuhostdev__ intvec contiguous_to_ndim_idx_1(std::uint64_t index, const intvec & shape, std::uint64_t skip_dim,
                                              std::uint64_t * data_ptr = nullptr);

/** @brief Launch asynchronously model fitting algorithm on CPU.*/
std::future<void> * cpu_async_launch(std::future<void> * current_job, candy::Model * p_model,
                                     const array::Array * p_train_data, candy::Optimizer * p_optimizer,
                                     std::uint64_t model_size, std::uint64_t n_thread, std::uint64_t rep);

/** @brief Launch asynchronously model fitting algorithm on GPU.*/
void gpu_asynch_launch(candy::Model * p_model, const array::Parcel * p_train_data, candy::Optimizer * p_optimizer,
                       std::uint64_t model_size, std::uint64_t ndim, std::uint64_t share_mem_size,
                       std::uint64_t block_size, std::uint64_t rep, cuda::Stream * stream_ptr);

/** @brief Push context and destroy the stream.*/
void destroy_stream_in_context(std::uintptr_t context_ptr, cuda::Stream *& stream_ptr);

}  // namespace candy

}  // namespace merlin

#endif  // MERLIN_CANDY_LAUNCHER_HPP_
