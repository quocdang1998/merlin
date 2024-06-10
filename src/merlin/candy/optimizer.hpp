// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_OPTIMIZER_HPP_
#define MERLIN_CANDY_OPTIMIZER_HPP_

#include <string>       // std::string
#include <type_traits>  // std::add_pointer
#include <utility>      // std::exchange
#include <variant>      // std::variant

#include "merlin/candy/declaration.hpp"         // merlin::candy::Model
#include "merlin/candy/optmz/adadelta.hpp"      // merlin::candy::optmz::AdaDelta
#include "merlin/candy/optmz/adagrad.hpp"       // merlin::candy::optmz::AdaGrad
#include "merlin/candy/optmz/adam.hpp"          // merlin::candy::optmz::Adam
#include "merlin/candy/optmz/grad_descent.hpp"  // merlin::candy::optmz::GradDescent
#include "merlin/candy/optmz/rmsprop.hpp"       // merlin::candy::optmz::RmsProp
#include "merlin/config.hpp"                    // __cudevice__
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS

namespace merlin {

namespace candy {

/** @brief Type for static data of optimizer.*/
using OptmzStatic = std::variant<candy::optmz::GradDescent, candy::optmz::AdaGrad, candy::optmz::Adam,
                                 candy::optmz::AdaDelta, candy::optmz::RmsProp>;

/** @brief Type of optimizing function.*/
using OptmzUpdater = std::add_pointer<void(void *, double *, candy::Model &, const candy::Gradient &, std::uint64_t,
                                           std::uint64_t, std::uint64_t) noexcept>::type;

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
    MERLIN_EXPORTS Optimizer(const candy::Optimizer & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS candy::Optimizer & operator=(const candy::Optimizer & src);
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src) :
    static_data(std::forward<candy::OptmzStatic>(src.static_data)), dynamic_size(src.dynamic_size) {
        this->dynamic_data = std::exchange(src.dynamic_data, nullptr);
    }
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src) {
        this->static_data = std::forward<candy::OptmzStatic>(src.static_data);
        this->dynamic_size = src.dynamic_size;
        this->dynamic_data = std::exchange(src.dynamic_data, nullptr);
        return *this;
    }
    /// @}

    /// @name Check compatibility with a model
    /// @{
    /** @brief Check compatibility with a model. Return ``false`` when incompatibility detected.*/
    MERLIN_EXPORTS bool is_compatible(const candy::Model & model) const;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model inside a CPU parallel region.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t time_step,
                                   std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;
#ifdef __NVCC__
    /** @brief Update model inside a GPU parallel region.*/
    __cudevice__ void update_gpu(candy::Model & model, const candy::Gradient & grad, std::uint64_t time_step,
                                 std::uint64_t thread_idx, std::uint64_t n_threads) noexcept;
#endif  // __NVCC__
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate number of bytes to allocate on GPU.*/
    std::uint64_t cumalloc_size(void) const noexcept {
        return sizeof(candy::Optimizer) + sizeof(double) * this->dynamic_size;
    }
    /** @brief Copy the optimizer from CPU to a pre-allocated memory on GPU.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param dynamic_data_ptr Pointer to a pre-allocated GPU memory storing dynamic data.
     *  @param stream_ptr Pointer to CUDA stream for asynchronous copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(candy::Optimizer * gpu_ptr, void * dynamic_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate additional number of bytes to allocate in CUDA shared memory for dynamic data.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
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
    /** @brief Copy data from GPU back to CPU.*/
    MERLIN_EXPORTS void * copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr = 0) noexcept;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
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
    double * dynamic_data = nullptr;
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

/** @brief Create an optimizer with adadelta algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_adadelta(double learning_rate, double rho, const candy::Model & model,
                                                double bias = 1.0e-8);

/** @brief Create an optimizer with rmsprop algorithm.*/
MERLIN_EXPORTS candy::Optimizer create_rmsprop(double learning_rate, double beta, const candy::Model & model,
                                               double bias = 1.0e-16);

}  // namespace candy

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
