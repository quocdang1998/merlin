// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::Vector

namespace merlin {

/** @brief Base class for optimizer of model.*/
class candy::Optimizer {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Optimizer(void) = default;
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Optimizer(const candy::Optimizer & src) = default;
    /** @brief Copy assignment.*/
    candy::Optimizer & operator=(const candy::Optimizer & src) = default;
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src) = default;
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src) = default;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model by gradient.
     *  @param model Candecomp model to be trained.
     *  @param gradient Vector storing gradient of each parameter.
     *  @param n_thread Number of CPU threads for execution.
     */
    virtual void update_cpu(candy::Model & model, floatvec & gradient, std::uint64_t n_thread = 1) noexcept {}
#ifdef __NVCC__
    /** @brief Update model by gradient value of current thread.*/
    __cudevice__ virtual void update_gpu(candy::Model * p_model, floatvec * p_gradient, void * share_ptr,
                                         std::uint64_t thread_idx, std::uint64_t block_size) noexcept {}
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Create an object on GPU by the GPU.
     *  @details Create object by GPU allow register v-table on the GPU, which is required for calling virtual
     *  functions. This function is synchronous.
     */
    virtual candy::Optimizer * new_gpu(void) const;
    /** @brief Destroy an object by GPU.*/
    static void delete_gpu(candy::Optimizer * p_optimizer);
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the object.*/
    virtual std::uint64_t sharedmem_size(void) const noexcept { return sizeof(candy::Optimizer); }
#ifdef __NVCC__
    /** @brief Copy necessary data to a pre-allocated memory region by a GPU block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param data_ptr Memory region where the data of the new created optimizer is stored.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ virtual void * copy_data_by_block(void * data_ptr, std::uint64_t thread_idx,
                                                   std::uint64_t block_size) {
        return data_ptr;
    }
    /** @brief Copy data to a pre-allocated memory region by a single GPU threads.
     *  @param data_ptr Pre-allocated pointer to memory region storing data.
     */
    __cudevice__ virtual void * copy_data_by_thread(void * data_ptr) { return data_ptr; }
#endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    virtual __cuhostdev__ ~Optimizer(void);
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
