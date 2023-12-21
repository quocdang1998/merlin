#ifndef MERLIN_CANDY_TRAINER_HPP_
#define MERLIN_CANDY_TRAINER_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model, merlin::candy::Optimizer, merlin::candy::Trainer
#include "merlin/cuda/declaration.hpp"  // merlin::cuda::Stream
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/synchronizer.hpp"  // merlin::ProcessorType, merlin::Synchronizer

namespace merlin {

namespace candy {

/** @brief Allocate memory on GPU for the trainer.*/
void create_trainer_gpu_ptr(const candy::Model & cpu_model, const array::Array & cpu_data,
                            const candy::Optimizer & cpu_optimizer, candy::Model *& gpu_model,
                            array::NdData *& gpu_data, candy::Optimizer *& gpu_optimizer, array::Parcel *& parcel_data,
                            cuda::Stream & stream);

}  // namespace candy

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

    /// @name Get elements and attributes
    /// @{
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

    /** @brief Size of shared memory to reserve (only for GPU launch).*/
    std::uint64_t shared_mem_size_ = 0;
    /** @brief GPU data (only for GPU launch).*/
    array::Parcel * p_parcel_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_TRAINER_HPP_
