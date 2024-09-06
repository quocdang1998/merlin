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
#include "merlin/config.hpp"                    // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"                   // MERLIN_EXPORTS

namespace merlin {

namespace candy {

/** @brief Type for static data of optimizer.*/
using OptmzStatic = std::variant<candy::optmz::GradDescent, candy::optmz::AdaGrad, candy::optmz::Adam,
                                 candy::optmz::AdaDelta, candy::optmz::RmsProp>;

/** @brief Type of optimizing function on CPU.*/
using OptmzUpdaterCpu =
    std::add_pointer<void(void *, double *, candy::Model &, const candy::Gradient &, std::uint64_t) noexcept>::type;

/** @brief Type of optimizing function on GPU.*/
using OptmzUpdaterGpu = std::add_pointer<void(void *, double *, candy::Model &, const candy::Gradient &, std::uint64_t,
                                              std::uint64_t, std::uint64_t) noexcept>::type;

}  // namespace candy

/** @brief Algorithm for updating a model based on its gradient.*/
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
    MERLIN_EXPORTS Optimizer(const candy::Optimizer & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS candy::Optimizer & operator=(const candy::Optimizer & src);
    /** @brief Move constructor.*/
    Optimizer(candy::Optimizer && src) :
    static_data_(std::forward<candy::OptmzStatic>(src.static_data_)), dynamic_size_(src.dynamic_size_) {
        this->dynamic_data_ = std::exchange(src.dynamic_data_, nullptr);
    }
    /** @brief Move assignment.*/
    candy::Optimizer & operator=(candy::Optimizer && src) {
        this->static_data_ = std::forward<candy::OptmzStatic>(src.static_data_);
        this->dynamic_size_ = src.dynamic_size_;
        this->dynamic_data_ = std::exchange(src.dynamic_data_, nullptr);
        return *this;
    }
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Get reference to static data.*/
    __cuhostdev__ constexpr candy::OptmzStatic & static_data(void) noexcept { return this->static_data_; }
    /** @brief Get constant reference to static data.*/
    __cuhostdev__ constexpr const candy::OptmzStatic & static_data(void) const noexcept { return this->static_data_; }
    /** @brief Get pointer to optimizer dynamic data.*/
    __cuhostdev__ constexpr double * dynamic_data(void) noexcept { return this->dynamic_data_; }
    /** @brief Get pointer to (constant) optimizer dynamic data.*/
    __cuhostdev__ constexpr const double * dynamic_data(void) const noexcept { return this->dynamic_data_; }
    /** @brief Allocate dynamic data.*/
    void allocate_data(std::uint64_t size);
    /// @}

    /// @name Check compatibility with a model
    /// @{
    /** @brief Check compatibility with a model. Return ``false`` when incompatibility detected.*/
    MERLIN_EXPORTS bool is_compatible(std::uint64_t num_params) const;
    /// @}

    /// @name Update model by gradient
    /// @{
    /** @brief Update model using CPU.*/
    MERLIN_EXPORTS void update_cpu(candy::Model & model, const candy::Gradient & grad,
                                   std::uint64_t time_step) noexcept;
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
        return sizeof(candy::Optimizer) + sizeof(double) * this->dynamic_size_;
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

  protected:
    /** @brief Static data for the algorithm (data resides on the stack memory).*/
    candy::OptmzStatic static_data_;
    /** @brief Dynamic data for the algorithm (data resides on the heap memory and must be deallocated in destructor).*/
    double * dynamic_data_ = nullptr;
    /** @brief Size of dynamic memory.*/
    std::uint64_t dynamic_size_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_OPTIMIZER_HPP_
