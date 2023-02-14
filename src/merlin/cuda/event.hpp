// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_EVENT_HPP_
#define MERLIN_CUDA_EVENT_HPP_

#include <string>  // std::string
#include <utility>  // std::exchange

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

/** @brief CUDA event.
 *  @details Milestone marking events in the CUDA stream.
 */
class MERLIN_EXPORTS cuda::Event {
  public:
    /** @brief Parameter describing the purpose of the event.*/
    enum class Category : unsigned int {
        /** Default event.*/
        Default = 0x00,
        /** Event meant to be synchronize with CPU (process on CPU blocked until the event occurs).*/
        BlockingSync = 0x01,
        /** Event not recording time data.*/
        DisableTiming = 0x02,
        /** Event might be used in an interprocess communication.*/
        EventInterprocess = 0x04
    };

    /// @name Constructor
    /// @{
    /** @brief Constructor from flag.
     *  @details Construct a CUDA event with a specific flag.
     *  @param category %Event flag.
     */
    Event(cuda::Event::Category category = cuda::Event::Category::Default);
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
    constexpr cuda::Event::Category category(void) const {return this->category_;}
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
    bool is_complete(void) const;
    /** @brief Check validity of GPU and context.
     * @details Check if the current CUDA context and active GPU is valid for the event.
     */
    void check_cuda_context(void) const;
    /// @}

    /// @name Operations
    /// @{
    /** @brief Synchronize the event.
     *  @details Block the CPU process until the event occurs.
     */
    void synchronize(void) const;

    MERLIN_EXPORTS friend float operator-(const cuda::Event & ev_1, const cuda::Event & ev_2);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Event(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUstream_st`` object.*/
    std::uintptr_t event_ = 0;
    /** @brief Creation flag of the event.*/
    Category category_;
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
