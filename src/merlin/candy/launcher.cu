// Copyright 2023 quocdang1998
#include "merlin/candy/launcher.hpp"

#include <cuda.h>  // ::CUContext

#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/context.hpp"     // merlin::cuda::Context
#include "merlin/cuda/stream.hpp"      // merlin::cuda::Stream
#include "merlin/env.hpp"              // merlin::Environment

namespace merlin {

// Constructor from a model and array on GPU
candy::Launcher::Launcher(candy::Model * p_model, const array::Parcel * p_train_data, candy::Optimizer * p_optimizer,
                          std::uint64_t model_size, std::uint64_t ndim, std::uint64_t share_mem_size,
                          std::uint64_t block_size) :
p_model_(p_model), p_data_(p_train_data), p_optimizer_(p_optimizer), model_size_(model_size), ndim_(ndim),
n_thread_(block_size) {
    Environment::mutex.lock();
    // set processor ID as current GPU
    this->processor_id_ = cuda::Context::get_current().get_context_ptr();
    // create CUDA stream of the current launcher
    cuda::Stream * p_stream = new cuda::Stream(cuda::StreamSetting::NonBlocking);
    this->synchronizer_ = reinterpret_cast<void *>(p_stream);
    Environment::mutex.unlock();
    // add share memory size for calculation
    this->shared_mem_size_ = share_mem_size + model_size * sizeof(double) + block_size * ndim * sizeof(std::uint64_t);
}

// Push context and destroy the stream
void candy::destroy_stream_in_context(std::uintptr_t context_ptr, cuda::Stream *& stream_ptr) {
    // get current context
    bool lock_success = Environment::mutex.try_lock();
    cuda::Context context(context_ptr);
    // delete the stream
    context.push_current();
    delete stream_ptr;
    stream_ptr = nullptr;
    context.pop_current();
    // release context
    context.assign(0);
    if (lock_success) {
        Environment::mutex.unlock();
    }
}

}  // namespace merlin
