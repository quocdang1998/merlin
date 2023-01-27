// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_CONTEXT_HPP_
#define MERLIN_CUDA_CONTEXT_HPP_

#include <cstdint>  // std::uintptr_t
#include <map>  // std::map
#include <mutex>  // std::mutex
#include <string>  // std::string
#include <utility>  // std::exchange, std::pair
#include <vector>  // std::vector

#include "merlin/cuda/declaration.hpp"  // merlin::cuda::Context, merlin::cuda::Device
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE

namespace merlin {

/** @brief Abstract class representing a CUDA context.
 *  @details CUDA associated to each CPU process a stack of context, each of which is bounded to a GPU. All CUDA
 *  operations are performed inside the context at the top of the stack.
 */
class MERLIN_EXPORTS cuda::Context {
  public:
    /** @brief Parameter controlling how the CPU process schedules tasks when waiting for results from the GPU.*/
    enum class Flags : unsigned int {
        /** Automatic schedule based on the number of context and number of logical process.*/
        AutoSchedule = 0x00,
        /** Actively spins when waiting for results from the GPU.*/
        SpinSchedule = 0x01,
        /** Yield the CPU process when waiting for results from the GPU.*/
        YieldSchedule = 0x02,
        /** Block CPU process until synchronization.*/
        BlockSyncSchedule = 0x04
    };

    /// @name Constructor
    /// @{
    /** @brief Construct a context referencing to the current context.*/
    Context(void) = default;
    /** @brief Construct a context assigned to a GPU and attached to the current CPU process.*/
    Context(const cuda::Device & gpu, cuda::Context::Flags flag = cuda::Context::Flags::AutoSchedule);
    /** @brief Construct a context from context pointer.
     *  @note The context will be destroyed after the instance is deleted.
     */
    Context(std::uintptr_t context_ptr);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Context(const cuda::Context & src) : context_(src.context_) {
        cuda::Context::mutex_.lock();
        cuda::Context::attribute_[src.context_].reference_count += 1;
        cuda::Context::mutex_.unlock();
    }
    /** @brief Copy assignment (deleted).*/
    cuda::Context & operator=(const cuda::Context & src) {
        this->context_ = src.context_;
        cuda::Context::mutex_.lock();
        cuda::Context::attribute_[src.context_].reference_count += 1;
        cuda::Context::mutex_.unlock();
        return *this;
    }
    /** @brief Move constructor.*/
    Context(cuda::Context && src) {
        this->context_ = std::exchange(src.context_, 0);
    }
    /** @brief Move assignment.*/
    cuda::Context & operator=(cuda::Context && src) {
        this->context_ = std::exchange(src.context_, 0);
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get pointer to Context object.*/
    constexpr std::uintptr_t get_context_ptr(void) const noexcept {return this->context_;}
    /** @brief Get number of instances representing the context.*/
    std::uint64_t get_reference_count(void) const noexcept {
        return cuda::Context::attribute_[this->context_].reference_count;
    }
    bool is_primary(void) const noexcept {return cuda::Context::attribute_[this->context_].is_primary;}
    /// @}

    /// @name Manipulation of the context stack
    /// @{
    /** @brief Check if the context is the top of context stack.*/
    bool is_current(void);
    /** @brief Push the context to the stack owned by the current CPU process.*/
    void push_current(void);
    /** @brief Pop the context out of the stack of the current CPU process.*/
    cuda::Context & pop_current(void);
    /// @}

    /// @name Query current context
    /// @{
    /** @brief Get current context.*/
    static cuda::Context get_current(void);
    /** @brief Get GPU attached to current context.*/
    static cuda::Device get_gpu_of_current_context(void);
    /** @brief Get flag of the current context.*/
    static cuda::Context::Flags get_flag_of_current_context(void);
    /** @brief Block for a context's tasks to complete.*/
    static void synchronize(void);
    /// @}

    /// @name Primary context
    /// @{
    friend cuda::Context create_primary_context(const cuda::Device & gpu, cuda::Context::Flags flag);
    /// @}

    /// @name Default context
    /// @{
    friend cuda::Context initialize_context(void);
    /// @}

    /// @name Comparison
    /// @{
    /** @brief Identical comparison operator.*/
    friend bool constexpr operator==(const cuda::Context & ctx_1, const cuda::Context & ctx_2) noexcept {
        return ctx_1.context_ == ctx_2.context_;
    }
    /** @brief Different comparison operator.*/
    friend bool constexpr operator!=(const cuda::Context & ctx_1, const cuda::Context & ctx_2) noexcept {
        return ctx_1.context_ != ctx_2.context_;
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string str(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Context(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUctx_st`` object.*/
    std::uintptr_t context_ = 0;
    /** @brief Mutex lock for updating static attributes.*/
    static std::mutex mutex_;

  private:
    /** @brief Attributes of the context.*/
    struct Attribute {
        /** @brief Reference count of the current context.
         *  @details Number of instances representing the same context.
         */
        std::uint64_t reference_count;
        /** @brief The current context is a primary context.
         *  @details Context is shared between CUDA runtime API and CUDA driver API.
         */
        bool is_primary = false;
        /** @brief GPU of the current context.
         *  @details GPU device binded to the primary context.
         */
        int gpu;
    };
    /** @brief Map from context pointers to their attributes.*/
    static std::map<std::uintptr_t, cuda::Context::Attribute> attribute_;
};

namespace cuda {

/** @brief Create a primary context.
 *  @param gpu GPU to which the primary context is binded.
 *  @param flag Setting flag to apply to the primary context.
 */
MERLIN_EXPORTS cuda::Context create_primary_context(const cuda::Device & gpu,
                                                    cuda::Context::Flags flag = cuda::Context::Flags::AutoSchedule);

/** @brief Initialize default context.
 *  Initialize a default regular context if not initialized.
 */
MERLIN_EXPORTS cuda::Context initialize_context(void);

/** @brief Default context.*/
MERLIN_EXPORTS extern cuda::Context default_context;

}  // namespace cuda

}  // namespace merlin

#endif  // MERLIN_CUDA_CONTEXT_HPP_
