// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_EVENT_HPP_
#define MERLIN_CUDA_EVENT_HPP_

#include <string>  // std::string
#include <utility>  // std::exchange

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/cuda/enum_wrapper.hpp"  // merlin::cuda::EventCategory
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief CUDA event.
 *  @details Milestone marking events in the CUDA stream.
 */
class cuda::Event {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor from flag.
     *  @details Construct a CUDA event with a specific flag.
     *  @param category Event flag.
     */
    MERLIN_EXPORTS Event(unsigned int category = cuda::EventCategory::DefaultEvent);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Event(const cuda::Event & src) = delete;
    /** @brief Copy assignment (deleted).*/
    cuda::Event & operator=(const cuda::Event & src) = delete;
    /** @brief Move constructor.*/
    Event(cuda::Event && src) {
        this->event_ = std::exchange(src.event_, 0);
        this->device_ = src.device_;
        this->context_ = src.context_;
    }
    /** @brief Move assignment.*/
    cuda::Event & operator=(cuda::Event && src) {
        this->event_ = std::exchange(src.event_, 0);
        this->device_ = src.device_;
        this->context_ = src.context_;
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get event pointer.*/
    constexpr std::uintptr_t get_event_ptr(void) const noexcept {return this->event_;}
    /** @brief Get setting flag of the event.*/
    constexpr unsigned int category(void) const {return this->category_;}
    /** @brief Get context associated to event.*/
    constexpr const cuda::Context & get_context(void) const noexcept {return this->context_;};
    /** @brief Get GPU.*/
    constexpr const cuda::Device & get_gpu(void) const noexcept {return this->device_;}
    /// @}

    /// @name Query
    /// @{
    /** @brief Query the status of all work currently captured by event.
     *  @details ``true`` if all captured work has been completed.
     */
    MERLIN_EXPORTS bool is_complete(void) const;
    /** @brief Check validity of GPU and context.
     * @details Check if the current CUDA context and active GPU is valid for the event.
     */
    MERLIN_EXPORTS void check_cuda_context(void) const;
    /// @}

    /// @name Operations
    /// @{
    /** @brief Synchronize the event.
     *  @details Block the CPU process until the event occurs.
     */
    MERLIN_EXPORTS void synchronize(void) const;

    MERLIN_EXPORTS friend float operator-(const cuda::Event & ev_1, const cuda::Event & ev_2);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Event(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUevent_st`` object.*/
    std::uintptr_t event_ = 0;
    /** @brief Creation flag of the event.*/
    unsigned int category_;
    /** @brief GPU associated to the event.*/
    cuda::Device device_;
    /** @brief Context associated to the event.*/
    cuda::Context context_;
};

namespace cuda {

/** @brief Calculate elapsed time (in millisecond) between 2 events.*/
MERLIN_EXPORTS float operator-(const cuda::Event & ev_1, const cuda::Event & ev_2);

}  // namespace cuda

}  // namespace merlin

#endif  // MERLIN_CUDA_EVENT_HPP_
