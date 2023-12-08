// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include <variant>  // std::variant

#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/candy/optmz/grad_descent.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS

namespace merlin {

namespace candy {

/** @brief Enum for optimization algorithms.*/
enum class OptAlgorithm : unsigned int {
    /** @brief Naive gradient descent algorithm.*/
    GdAlgo = 0x00,
};

/** @brief Type for static data of optimizer.*/
using OptmzStatic = std::variant<candy::optmz::GradDescent>;

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
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                   std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a CPU parallel region.*/
    __cudevice__ void update_gpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t thread_idx,
                                 std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate additional number of bytes to allocate for dynamic data.*/
    MERLIN_EXPORTS std::uint64_t cumalloc_size(void) const noexcept;
    /** @brief Copy the optimizer from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate additional number of bytes to allocate in CUDA shared memory for dynamic data.*/
    MERLIN_EXPORTS std::uint64_t sharedmem_size(void) const noexcept;
    // copy by block and copy by thread are wait for implementation
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
    /** @brief Optimization algorithm.*/
    candy::OptAlgorithm algorithm;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
