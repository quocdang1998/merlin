// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/device.hpp"      // merlin::cuda::CtxGuard
#include "merlin/cuda/memory.hpp"      // merlin::cuda::Memory
#include "merlin/cuda/stream.hpp"      // merlin::cuda::Stream
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Update CP model according to gradient on GPU
void candy::Trainer::update_gpu(const array::Parcel & data, std::uint64_t rep, double threshold,
                                std::uint64_t n_threads, candy::TrainMetric metric) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + this->optmz_.sharedmem_size() + data.sharedmem_size();
    shared_mem += sizeof(double) * this->model_.num_params();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, this->optmz_);
    candy::train_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), metric, rep, n_threads, threshold, shared_mem,
                        stream);
    this->model_.copy_from_gpu(reinterpret_cast<double *>(mem.get<0>() + 1), stream.get_stream_ptr());
}

// Get the RMSE and RMAE error with respect to a given dataset by GPU
void candy::Trainer::error_gpu(const array::Parcel & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + data.sharedmem_size();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->synch_.synchronizer);
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, rmse, rmae);
    candy::error_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), mem.get<3>(), n_threads, shared_mem, stream);
    ::cudaMemcpyAsync(&rmse, mem.get<2>(), sizeof(double), ::cudaMemcpyDeviceToHost, cuda_stream);
    ::cudaMemcpyAsync(&rmae, mem.get<3>(), sizeof(double), ::cudaMemcpyDeviceToHost, cuda_stream);
}

}  // namespace merlin
