// Copyright 2023 quocdang1998
#include "merlin/candy/launcher.hpp"

#include <cuda.h>  // ::CUContext

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/context.hpp"  // merlin::cuda::Context
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/env.hpp"  // merlin::Environment

namespace merlin {

// Constructor from a model and array on GPU
candy::Launcher::Launcher(candy::Model * p_model, const array::Parcel * p_train_data, candy::Optimizer * p_optimizer,
                          std::uint64_t model_size, std::uint64_t share_mem_size, std::uint64_t block_size) :
p_model_(p_model), p_data_(p_train_data), p_optimizer_(p_optimizer), model_size_(model_size),
shared_mem_size_(share_mem_size), n_thread_(block_size) {
    Environment::mutex.lock();
    // set processor ID as current GPU
    this->processor_id_ = cuda::Context::get_current().get_context_ptr();
    // create CUDA stream of the current launcher
    cuda::Stream * p_stream = new cuda::Stream(cuda::StreamSetting::NonBlocking);
    this->synchronizer_ = reinterpret_cast<void *>(p_stream);
    Environment::mutex.unlock();
}

// Destructor
candy::Launcher::~Launcher(void) {
    Environment::mutex.lock();
    // push current context if Launcher is GPU type
    ::CUcontext ctx;
    cuda::Stream * stream_ptr;
    if (this->is_gpu()) {
        ctx = reinterpret_cast<::CUcontext>(this->processor_id_);
        ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxPushCurrent(ctx));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Push context to current stack failed with message \"%s\".\n",
                    ::cudaGetErrorString(err_));
        }
        stream_ptr = reinterpret_cast<cuda::Stream *>(this->synchronizer_);
    }
    // deallocate synchronizer
    if (this->synchronizer_ != nullptr) {
        if (this->is_gpu()) {
            delete stream_ptr;
        } else {
            std::future<void> * p_future = reinterpret_cast<std::future<void> *>(this->synchronizer_);
            delete p_future;
        }
    }
    // pop the current context
    if (this->is_gpu()) {
        ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxPopCurrent(&ctx));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Pop current context out of the stack failed with message \"%s\".\n",
                    ::cudaGetErrorString(err_));
        }
    }
    Environment::mutex.unlock();
}

}  // namespace merlin
