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

class MERLIN_EXPORTS Stream {
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
    Stream(Context & context, Setting setting = Setting::Default, int priority = 0);
    /// @}

    /// @name Copy and Move
    /// @{
    Stream(const Stream & src) = delete;
    Stream & operator=(const Stream & src) = delete;
    Stream(Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        this->context_ = std::exchange(src.context_, nullptr);
    }
    Stream & operator=(Stream && src) {
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
    /// @}

    /// @name Operations
    /// @{
    void launch_cpu_function(CudaStreamCallback func, void * arg);
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
    Context * context_ = nullptr;
};

}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_STREAM_HPP_
