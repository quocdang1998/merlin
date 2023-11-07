// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_STREAM_HPP_
#define MERLIN_CUDA_STREAM_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::uintptr_t
#include <string>   // std::string
#include <utility>  // std::exchange

#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Event
#include "merlin/cuda/device.hpp"        // merlin::cuda::Device
#include "merlin/cuda/enum_wrapper.hpp"  // merlin::cuda::StreamSetting, merlin::cuda::EventWaitFlag

namespace merlin::cuda {}  // namespace merlin::cuda

namespace merlin {

/** @brief CUDA stream of tasks.*/
class cuda::Stream {
  public:
#ifdef __NVCC__
    /** @brief Type of CPU callback function.*/
    typedef cudaStreamCallback_t CudaStreamCallback;
#else
    typedef void (*CudaStreamCallback)(std::uintptr_t, int, void *);
#endif  // __NVCC__

    /// @name Constructor
    /// @{
    /** @brief Default constructor (the null stream).*/
    MERLIN_EXPORTS Stream(void);
    /** @brief Constructor from Setting and priority.
     *  @details Construct a CUDA stream from its setting and priority in the current context.
     *  @param setting %Stream creation flag.
     *  @param priority %Stream task priority (lower number means higher priority).
     */
    MERLIN_EXPORTS Stream(cuda::StreamSetting setting, int priority = 0);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor (deleted).*/
    Stream(const cuda::Stream & src) = delete;
    /** @brief Copy assignment (deleted).*/
    cuda::Stream & operator=(const cuda::Stream & src) = delete;
    /** @brief Move constructor.*/
    Stream(cuda::Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        this->device_ = src.device_;
    }
    /** @brief Move assignment.*/
    cuda::Stream & operator=(cuda::Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        this->device_ = src.device_;
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get stream pointer.*/
    constexpr std::uintptr_t get_stream_ptr(void) const noexcept { return this->stream_; }
    /** @brief Get setting flag of the stream.*/
    MERLIN_EXPORTS cuda::StreamSetting setting(void) const;
    /** @brief Get priority of the stream.*/
    MERLIN_EXPORTS int priority(void) const;
    /** @brief Get GPU.*/
    constexpr const cuda::Device & get_gpu(void) const noexcept { return this->device_; }
    /// @}

    /// @name Query
    /// @{
    /** @brief Query for completion status.
     *  @details ``true`` if all operations in the stream have completed.
     */
    MERLIN_EXPORTS bool is_complete(void) const;
    /** @brief Check if the stream is being captured.*/
    MERLIN_EXPORTS bool is_capturing(void) const;
    /** @brief Check validity of GPU and context.
     * @details Check if the current CUDA context and active GPU is valid for the stream.
     */
    MERLIN_EXPORTS void check_cuda_context(void) const;
    /// @}

    /// @name Operations
    /// @{
    /** @brief Launch CPU function with a certain arguments.
     *  @param func Function to be launched, prototype ``void func(void *)``.
     *  @param arg Argument to be provided to the function.
     *  @details Example:
     *  @code {.cu}
     *  // arguments
     *  void str_callback(::cudaStream_t stream, ::cudaError_t status, void * data) {
     *      int & a = *(static_cast<int *>(data));
     *      std::printf("Callback argument: %d\n", a);
     *  }
     *  int data = 1;
     *
     *  // use
     *  merlin::cuda::Stream s();
     *  s.add_callback(str_callback, &data);
     *
     *  // result
     *  Callback argument: 1
     *  @endcode
     */
    MERLIN_EXPORTS void add_callback(cuda::Stream::CudaStreamCallback func, void * arg) const;
    /** @brief Record (register) an event on CUDA stream.
     *  @param event CUDA event to be recorded.
     */
    MERLIN_EXPORTS void record_event(const cuda::Event & event) const;
    /** @brief Make CUDA stream wait on an event.
     *  @param event CUDA event to be synchronized.
     *  @param flag Flag of the event wait.
     */
    MERLIN_EXPORTS void wait_event(const cuda::Event & event,
                                   cuda::EventWaitFlag flag = cuda::EventWaitFlag::Default) const;
    /** @brief Synchronize the stream.
     *  @details Pause the CPU process until all operations on the stream has finished.
     */
    MERLIN_EXPORTS void synchronize(void) const;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Stream(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUstream_st`` object.*/
    std::uintptr_t stream_ = 0;
    /** @brief GPU associated to the stream.*/
    cuda::Device device_;
};

namespace cuda {

/** @brief Capture mode.*/
enum class MERLIN_EXPORTS StreamCaptureMode : unsigned int {
    /** Global capture mode.*/
    Global = 0x0,
    /** Thread local capture mode.*/
    ThreadLocal = 0x1,
    /** Relaxed capture.*/
    Relaxed = 0x2
};

/** @brief Capturing stream for CUDA graph.
 *  @details When a CUDA stream is in capture mode, execution orders will not be launch. It is enqueued in an execution
 *  CUDA graph to be retrieved later.
 */
MERLIN_EXPORTS void begin_capture_stream(const cuda::Stream & stream,
                                         StreamCaptureMode mode = StreamCaptureMode::Global);

/** @brief End capturing a stream and returning a graph.
 *  @details Stop CUDA stream capture mode and return a CUDA graph representing enqueued works.
 */
MERLIN_EXPORTS cuda::Graph end_capture_stream(const cuda::Stream & stream);

}  // namespace cuda

}  // namespace merlin

#endif  // MERLIN_CUDA_STREAM_HPP_
