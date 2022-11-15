// Copyright 2022 quocdang1998
#ifndef MERLIN_DEVICE_CONTEXT_HPP_
#define MERLIN_DEVICE_CONTEXT_HPP_

#include <cstdint>  // std::uintptr_t
#include <utility>  // std::exchange, std::pair
#include <vector>  // std::vector

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/device/gpu_query.hpp"  // merlin::device::Device
#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE

namespace merlin::device {

/** @brief Abstract class representing a CUDA context.
 *  @details CUDA associated to each CPU process a stack of context, each of which is bounded to a GPU. All CUDA operations
 *  are performed inside the context at the top of the stack.
 */
class MERLIN_EXPORTS Context {
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
    Context(void) {}
    /** @brief Construct a context assigned to a GPU and attached to the current CPU process.*/
    Context(const Device & gpu, Flags flag = Flags::AutoSchedule);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Context(const Context & src) = delete;
    /** @brief Copy assignment (deleted).*/
    Context & operator=(const Context & src) = delete;
    /** @brief Move constructor.*/
    Context(Context && src) {
        if (src.context_ == 0) {
            FAILURE(cuda_runtime_error, "Original context is a primary context.\n");
        }
        this->context_ = std::exchange(src.context_, 0);
        this->device_ = std::exchange(src.device_, 0);
    }
    /** @brief Move assignment.*/
    Context & operator=(Context && src) {
        if (src.context_ == 0) {
            FAILURE(cuda_runtime_error, "Original context is a primary context.\n");
        }
        this->context_ = std::exchange(src.context_, 0);
        this->device_ = std::exchange(src.device_, 0);
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get GPU bounded to the context.*/
    Device get_gpu(void) const {return this->device_;}
    /** @brief Check if the context is attached to any CPU process.*/
    bool is_attached(void) const {return this->attached_;}
    /// @}

    /// @name Manipulation of the context stack
    /// @{
    /** @brief Push the context to the stack owned by the current CPU process.*/
    void push_current(void);
    /** @brief Pop the context out of the stack of the current CPU process.*/
    Context & pop_current(void);
    /** @brief Check if the context is the top of context stack.*/
    bool is_current(void);
    /** @brief Set current context at the top of the stack.*/
    void set_current(void);
    /// @}

    /// @name Interaction with primary context.
    /// @{
    /** @brief List of primary contexts of each GPU.*/
    static std::vector<Context> primary_contexts;
    /** @brief Create list of primary contexts at initialization.*/
    static void create_primary_context_list(void);
    /** @brief Create primary context instance assigned to a GPU.*/
    static Context create_primary_context(const Device & gpu);
    /** @brief Get primary context instance corresponding to a GPU.*/
    static Context & get_primary_context(const Device & gpu) {return Context::primary_contexts[gpu.id()];}
    /** @brief Get state of the primary context.
     *  @returns Active state (``false`` means inactive) and setting flag of the primary context associated with the GPU.
     */
    static std::pair<bool, Flags> get_primary_ctx_state(const Device & gpu);
    /** @brief Set flag for primary context.*/
    static void set_flag_primary_context(const Device & gpu, Flags flag);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Context(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUctx_st`` object.*/
    std::uintptr_t context_ = 0;
    /** @brief GPU associated to the CUDA context.*/
    Device device_;
    /** @brief Context attached to CPU process.*/
    bool attached_ = true;
};

}  // namespace merlin::device

#endif  // MERLIN_DEVICE_CONTEXT_HPP_
