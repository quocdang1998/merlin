// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include "merlin/candy/declaration.hpp"         // merlin::candy::Model
#include "merlin/candy/optmz/adagrad.hpp"       // merlin::candy::optmz::AdaGrad
#include "merlin/candy/optmz/adam.hpp"          // merlin::candy::optmz::Adam
#include "merlin/candy/optmz/grad_descent.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/cuda_interface.hpp"            // __cudevice__
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS

namespace merlin {

namespace candy {

/** @brief Type for static data of optimizer.*/
using OptmzStatic = std::variant<candy::optmz::GradDescent, candy::optmz::AdaGrad, candy::optmz::Adam>;

}  // namespace candy

/** @brief Algorithm for updating a model based on its gradient.*/
struct candy::Optimizer {
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Optimizer(void) = default;
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Optimizer(const candy::Optimizer & src);
    /** @brief Copy assignment.*/
    candy::Optimizer & operator=(const candy::Optimizer & src);
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src);
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src);
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                   std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a GPU parallel region.*/
    __cudevice__ void update_gpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                 std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate number of bytes to allocate on GPU.*/
    std::uint64_t cumalloc_size(void) const noexcept {
        return sizeof(candy::Optimizer) + sizeof(char) * this->dynamic_size;
    }
    /** @brief Copy the optimizer from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate additional number of bytes to allocate in CUDA shared memory for dynamic data.*/
    std::uint64_t sharedmem_size(void) const noexcept { return sizeof(candy::Optimizer); }
#ifdef __NVCC__
    /** @brief Copy object to pre-allocated memory region by current CUDA block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the object is copied to.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(candy::Optimizer * dest_ptr, void * dynamic_data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy object to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the object is copied to.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     */
    __cudevice__ void * copy_by_thread(candy::Optimizer * dest_ptr, void * dynamic_data_ptr) const;
#endif  // __NVCC__
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Optimizer(void);
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Static data for the algorithm (data resides on the stack memory).*/
    candy::OptmzStatic static_data;
    /** @brief Dynamic data for the algorithm (data resides on the heap memory and must be deallocated in destructor).*/
    char * dynamic_data = nullptr;
    /** @brief Size of dynamic memory.*/
    std::uint64_t dynamic_size = 0;
    /// @}
};

namespace candy {

/** @brief Create an optimizer with gradient descent algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_grad_descent(double learning_rate);

/** @brief Create an optimizer with adagrad algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_adagrad(double learning_rate, const candy::Model & model, double bias = 1.0e-8);

/** @brief Create an optimizer with adam algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_adam(double learning_rate, double beta_m, double beta_v,
                                            const candy::Model & model, double bias = 1.0e-8);

}  // namespace candy

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
