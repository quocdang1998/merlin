// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_LAUNCHER_HPP_
#define MERLIN_CANDY_LAUNCHER_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/array/declaration.hpp"  // merlin::array::NdData
#include "merlin/candy/declaration.hpp"  // merlin::candy::Launcher, merlin::candy::Model, merlin::candy::Optimizer
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::floatvec

namespace merlin {

/** @brief Class launching model training.*/
class candy::Launcher {
  public:
    /** @brief Default constructor.*/
    Launcher(void) =  default;
    /** @brief Constructor from a model and CPU array.*/
    MERLIN_EXPORTS Launcher(candy::Model & model, const array::Array & train_data, candy::Optimizer & optimizer,
                            std::uint64_t n_thread = 1);

    /** @brief Check if the processor is a GPU.*/
    bool is_gpu(void) const noexcept {return this->processor_id_ >= 0;}

    /** @brief Launch asynchronously the gradient update.
     *  @param rep Number of times to update model parameter.
     */
    void launch_async(std::uint64_t rep = 1);
    /** @brief Synchronize the launch.
     *  @details Force CPU to wait until the launch has finished.
     */
    void synchronize(void);

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
     *  @details Value ``-1`` indicates CPU processor. Positive values represent GPU ID.
     */
    std::int64_t processor_id_ = -1;
    /** @brief Number of threads to use.*/
    std::uint64_t n_thread_ = 1;

  private:
    /** @brief Pointer to synchronizer .*/
    void * synchronizer_ = nullptr;
    /** @brief Vector storing gradient for calculation on CPU.*/
    floatvec gradient_;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_LAUNCHER_HPP_
