// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_STREAM_HPP_
#define MERLIN_CUDA_STREAM_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::uintptr_t
#include <utility>  // std::exchange

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context

namespace merlin::cuda {
#ifdef __NVCC__
/** @brief Type of CPU callback function.*/
typedef cudaHostFn_t CudaStreamCallback;
#else
typedef void(* CudaStreamCallback)(void *);
#endif  // __NVCC__
}  // namespace merlin::cuda

namespace merlin {

class MERLIN_EXPORTS cuda::Stream {
  public:
    enum class Setting : unsigned int {
        /** @brief Default stream creation flag (synchonized with the null stream).*/
        Default = 0x00,
        /** @brief Works may run concurrently with null stream.*/
        NonBlocking = 0x01
    };

    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Stream(void) = default;
    /** @brief Constructor from Setting and priority.*/
    Stream(cuda::Context & context, cuda::Stream::Setting setting = cuda::Stream::Setting::Default, int priority = 0);
    /// @}

    /// @name Copy and Move
    /// @{
    Stream(const cuda::Stream & src) = delete;
    cuda::Stream & operator=(const cuda::Stream & src) = delete;
    Stream(cuda::Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        this->context_ = std::exchange(src.context_, nullptr);
    }
    cuda::Stream & operator=(cuda::Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        this->context_ = std::exchange(src.context_, nullptr);
        return *this;
    }
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get stream pointer.*/
    std::uintptr_t stream(void) const {return this->stream_;}
    /** @brief Get flag.*/
    Setting setting(void);
    /** @brief Get priority.*/
    int priority(void);
    /** @brief Query for completion status.
     *  @details ``true`` if all operations in the stream have completed.
     */
    bool is_complete(void);
    /** @brief Get GPU.*/
    cuda::Device get_gpu(void) const {
        if (this->context_ == nullptr) {
            return cuda::Device::get_current_gpu();
        }
        return this->context_->get_gpu();
    }
    /// @}

    /// @name Operations
    /// @{
    void launch_cpu_function(cuda::CudaStreamCallback func, void * arg);
    /** @brief Synchronize the stream.*/
    void synchronize(void);
    /// @}

    /// @name Destructor
    /// @{
    ~Stream(void);
    /// @}

  protected:
    /** @brief Pointer to ``CUstream_st`` object.*/
    std::uintptr_t stream_ = 0;
    /** @brief Pointer to context containing the stream.*/
    cuda::Context * context_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_CUDA_STREAM_HPP_
