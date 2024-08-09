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
void candy::Trainer::update_until(const array::Parcel & data, std::uint64_t rep, double threshold,
                                  std::uint64_t n_threads, candy::TrainMetric metric, bool export_result) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + this->optmz_.sharedmem_size() + data.sharedmem_size();
    shared_mem += sizeof(double) * this->model_.num_params();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, this->optmz_);
    candy::train_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), metric, rep, n_threads, threshold, shared_mem,
                        stream);
    this->model_.copy_from_gpu(reinterpret_cast<double *>(mem.get<0>() + 1), stream.get_stream_ptr());
    this->optmz_.copy_from_gpu(reinterpret_cast<double *>(mem.get<2>() + 1), stream.get_stream_ptr());
    // save model to a file
    if (export_result) {
        stream.add_callback(candy::save_model, this->model_, this->fname_);
    }
}

// Update CP model according to gradient using GPU
void candy::Trainer::update_for(const array::Parcel & data, std::uint64_t max_iter, std::uint64_t n_threads,
                                candy::TrainMetric metric, bool export_result) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + this->optmz_.sharedmem_size() + data.sharedmem_size();
    shared_mem += sizeof(double) * this->model_.num_params();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, this->optmz_);
    candy::process_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), max_iter, n_threads, metric, shared_mem, stream);
    this->model_.copy_from_gpu(reinterpret_cast<double *>(mem.get<0>() + 1), stream.get_stream_ptr());
    this->optmz_.copy_from_gpu(reinterpret_cast<double *>(mem.get<2>() + 1), stream.get_stream_ptr());
    // save model to a file
    if (export_result) {
        stream.add_callback(candy::save_model, this->model_, this->fname_);
    }
}

// Get the RMSE and RMAE error with respect to a given dataset by GPU
void candy::Trainer::get_error(const array::Parcel & data, double & rmse, double & rmae, std::uint64_t n_threads) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + data.sharedmem_size();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, rmse, rmae);
    candy::error_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), mem.get<3>(), n_threads, shared_mem, stream);
    ::cudaMemcpyAsync(&rmse, mem.get<2>(), sizeof(double), ::cudaMemcpyDeviceToHost, cuda_stream);
    ::cudaMemcpyAsync(&rmae, mem.get<3>(), sizeof(double), ::cudaMemcpyDeviceToHost, cuda_stream);
}

// Dry-run using GPU
void candy::Trainer::dry_run(const array::Parcel & data, DoubleVec & error, std::uint64_t & actual_iter,
                             candy::TrialPolicy policy, std::uint64_t n_threads, candy::TrainMetric metric) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // check error length
    if (error.size() < policy.sum()) {
        Fatal<std::invalid_argument>("Size of error must be greater or equal to max_iter.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + this->optmz_.sharedmem_size() + data.sharedmem_size();
    shared_mem += sizeof(double) * this->model_.num_params();
    // copy memory to GPU
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, data, this->optmz_, error, actual_iter);
    // dry run
    double * error_gpu = reinterpret_cast<double *>(mem.get<3>() + 1);
    candy::dryrun_by_gpu(mem.get<0>(), mem.get<1>(), mem.get<2>(), metric, n_threads, error_gpu, mem.get<4>(), policy,
                         shared_mem, stream);
    // clone data back to CPU
    error.copy_from_gpu(error_gpu, stream.get_stream_ptr());
    ::cudaMemcpyAsync(&actual_iter, mem.get<4>(), sizeof(std::uint64_t), ::cudaMemcpyDeviceToHost,
                      reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
}

// Reconstruct a whole multi-dimensional data from the model using GPU parallelism
void candy::Trainer::reconstruct(array::Parcel & destination, std::uint64_t n_threads) {
    // check if trainer is on GPU
    if (!(this->on_gpu())) {
        Fatal<std::invalid_argument>("The current object is allocated on CPU.\n");
    }
    // check shape
    if (!this->model_.check_compatible_shape(destination.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // calculate shared memory
    std::uint64_t shared_mem = this->model_.sharedmem_size() + destination.sharedmem_size();
    // copy memory to GPU and launch kernel
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    cuda::Memory mem(stream.get_stream_ptr(), this->model_, destination);
    candy::reconstruct_by_gpu(mem.get<0>(), mem.get<1>(), n_threads, shared_mem, stream);
}

}  // namespace merlin
