// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_CONTEXT_HPP_
#define MERLIN_CUDA_CONTEXT_HPP_

#include <cstdint>  // std::uintptr_t
#include <map>  // std::map
#include <mutex>  // std::mutex
#include <utility>  // std::exchange, std::pair
#include <vector>  // std::vector

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/cuda/gpu_query.hpp"  // merlin::device::Device
#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE

namespace merlin::cuda {

/** @brief Abstract class representing a CUDA context.
 *  @details CUDA associated to each CPU process a stack of context, each of which is bounded to a GPU. All CUDA
 *  operations are performed inside the context at the top of the stack.
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
    Context(void) = default;
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
    }
    /** @brief Move assignment.*/
    Context & operator=(Context && src) {
        if (src.context_ == 0) {
            FAILURE(cuda_runtime_error, "Original context is a primary context.\n");
        }
        this->context_ = std::exchange(src.context_, 0);
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get GPU bounded to the context.*/
    Device get_gpu(void) const {return Context::shared_attributes_[this->context_].device;}
    /** @brief Check if the context is attached to any CPU process.*/
    bool is_attached(void) const {return Context::shared_attributes_[this->context_].attached;}
    /// @}

    /// @name Manipulation of the context stack
    /// @{
    /** @brief Push the context to the stack owned by the current CPU process.*/
    void push_current(void);
    /** @brief Pop the context out of the stack of the current CPU process.*/
    Context & pop_current(void);
    /** @brief Get current context.*/
    static Context get_current(void);
    /** @brief Check if the context is the top of context stack.*/
    bool is_current(void);
    /** @brief Set current context at the top of the stack.*/
    void set_current(void);
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
    static std::mutex m_;
    /** @brief Attributes shared between contextes instances.*/
    struct SharedAttribures {
        unsigned int reference_count;
        bool attached;
        Device device;
    };
    /** @brief Attributes of Context instances.*/
    static std::map<std::uintptr_t, SharedAttribures> shared_attributes_;
};

}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_CONTEXT_HPP_
