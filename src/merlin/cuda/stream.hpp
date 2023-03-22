// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_STREAM_HPP_
#define MERLIN_CUDA_STREAM_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::uintptr_t
#include <string>  // std::string
#include <utility>  // std::exchange

#include "merlin/cuda/declaration.hpp"  // merlin::cuda::Event
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device

namespace merlin::cuda {
#ifdef __NVCC__
/** @brief Type of CPU callback function.*/
typedef cudaStreamCallback_t CudaStreamCallback;
#else
typedef void(* CudaStreamCallback)(std::uintptr_t, int, void *);
#endif  // __NVCC__
}  // namespace merlin::cuda

namespace merlin {

/** @brief CUDA stream of tasks.*/
class cuda::Stream {
  public:
    /** @brief Parameter controlling the behavior of the stream.*/
    enum class MERLIN_EXPORTS Setting : unsigned int {
        /** Default stream creation flag (synchonized with the null stream).*/
        Default = 0x00,
        /** Works may run concurrently with null stream.*/
        NonBlocking = 0x01
    };

    /// @name Constructor
    /// @{
    /** @brief Default constructor (the null stream).*/
    MERLIN_EXPORTS Stream(void);
    /** @brief Constructor from Setting and priority.
     *  @details Construct a CUDA stream from its setting and priority in the current context.
     *  @param setting %Stream creation flag.
     *  @param priority %Stream task priority (lower number means higher priority).
     */
    MERLIN_EXPORTS Stream(cuda::Stream::Setting setting, int priority = 0);
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
    constexpr std::uintptr_t get_stream_ptr(void) const noexcept {return this->stream_;}
    /** @brief Get setting flag of the stream.*/
    MERLIN_EXPORTS cuda::Stream::Setting setting(void) const;
    /** @brief Get priority of the stream.*/
    MERLIN_EXPORTS int priority(void) const;
    /** @brief Get context associated to stream.*/
    MERLIN_EXPORTS cuda::Context get_context(void) const;
    /** @brief Get GPU.*/
    constexpr const cuda::Device & get_gpu(void) const noexcept {return this->device_;}
    /// @}

    /// @name Query
    /// @{
    /** @brief Query for completion status.
     *  @details ``true`` if all operations in the stream have completed.
     */
    MERLIN_EXPORTS bool is_complete(void) const;
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
    MERLIN_EXPORTS void add_callback(cuda::CudaStreamCallback func, void * arg);
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

/** @brief Record (register) an event on CUDA stream.
 *  @param event CUDA event to be recorded.
 *  @param stream CUDA stream on which the event is recorded (default value is the null stream).
 */
MERLIN_EXPORTS void record_event(const cuda::Event & event, const cuda::Stream & stream = cuda::Stream());

}  // namespace cuda

}  // namespace merlin

#endif  // MERLIN_CUDA_STREAM_HPP_
