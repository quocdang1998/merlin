// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_STREAM_HPP_
#define MERLIN_CUDA_STREAM_HPP_

#include <cstdint>  // std::uint64_t
#include <utility>  // std::exchange

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context

namespace merlin::cuda {

class MERLIN_EXPORTS Stream {
  public:
    Stream(void) = default;
    Stream()

    Stream(const Stream & src) = delete;
    Stream & operator=(const Stream & src) = delete;
    Stream(Stream && src) {this->stream_ = std::exchange(src.stream_, 0);}
    Stream & operator=(Stream && src) {
        this->stream_ = std::exchange(src.stream_, 0);
        return *this;
    }

    ~Stream(void);

  protected:
    /** @brief Pointer to ``CUstream_st`` object.*/
    std::uint64_t stream_ = 0;
    /** @brief Context containing the stream.*/
    merlin::cuda::Context context_;
};

}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_STREAM_HPP_
