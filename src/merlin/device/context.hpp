// Copyright 2022 quocdang1998
#ifndef MERLIN_DEVICE_STREAM_HPP_
#define MERLIN_DEVICE_STREAM_HPP_

#include <cstdint>  // std::uintptr_t
#include <utility>  // std::exchange

#include "merlin/exports.hpp"  // 
#include "merlin/device/gpu_query.hpp"  // merlin::device::Device

namespace merlin::device {

/** @brief Abstract class representing a CUDA context.
 *  @details CUDA associated to each CPU process a stack of context, each of which is bounded to a GPU. All CUDA operations
 *  are performed inside the context at the top of the stack.
 */
class MERLIN_EXPORTS Context {
  public:
    enum class Flags : unsigned int {
        AutoSchedule = 0x00,
        SpinSchedule = 0x01,
        YieldSchedule = 0x02,
        BlockSyncSchedule = 0x04
    };

    /// @name Constructor
    /// @{
    /** @brief Construct a context referencing to the current context.*/
    Context(void);
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
        this->context_ = std::exchange(src.context_, NULL);
        this->device_ = std::exchange(src.device_, NULL);
    }
    /** @brief Move assignment.*/
    Context & operator=(Context && src) {
        this->context_ = std::exchange(src.context_, NULL);
        this->device_ = std::exchange(src.device_, NULL);
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get GPU bounded to the context.*/
    Device get_gpu(void) const {return this->device_;}
    /** @brief Check if the context is attached to any CPU process.
     *  @note If the current object is a reference and
     */
    bool is_attached(void) const {return this->attached_;}
    /** @brief Check if the object is referenced to any other context.*/
    bool is_reference(void) const {return this->reference_;}
    /// @}

    /// @name Context stack manipulation
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

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Context(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUctx_st`` object.*/
    std::uintptr_t context_;
    /** @brief GPU associated to the CUDA context.*/
    Device device_;
    /** @brief Context attached to any CPU.*/
    bool attached_ = true;
    /** @brief Reference to an already existing `CUctx_st` object.*/
    bool reference_ = false;
};

}  // namespace merlin::device

#endif  // MERLIN_DEVICE_STREAM_HPP_
