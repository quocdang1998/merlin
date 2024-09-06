// Copyright 2024 quocdang1998
#include "merlin/candy/train/gpu_trainer.hpp"

#include <algorithm>  // std::max
#include <cinttypes>  // PRIu64
#include <numeric>    // std::accumulate

#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/assume.hpp"           // merlin::assume
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/memory.hpp"      // merlin::cuda::Memory
#include "merlin/logger.hpp"           // merlin::cuda_runtime_error, merlin::Fatal, merlin::Warning
#include "merlin/vector.hpp"           // merlin::DoubleVec, merlin::UIntVec

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Throw CUDA error
static inline void check_cuda_error(::cudaError_t error, const std::string & step_name) {
    if (error != 0) {
        Fatal<cuda_runtime_error>("%s failed with error \"%s\"", step_name.c_str(), ::cudaGetErrorString(error));
    }
}

// Transfer vector of data to GPU
static inline array::Parcel * transfer_parcels(std::vector<array::Parcel *> & p_rec_data, cuda::Stream & stream) {
    // check if all data are initialized on the same GPU as the stream
    for (std::uint64_t index = 0; index < p_rec_data.size(); index++) {
        if (p_rec_data[index]->device() != stream.get_gpu()) {
            Fatal<std::runtime_error>("Array %" PRIu64 " is allocated on a diffrent GPU than the stream.\n", index);
        }
    }
    // allocate array for reconstructed data on GPU
    array::Parcel * p_data;
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    ::cudaError_t err;
    err = ::cudaMallocAsync(&p_data, sizeof(array::Parcel) * p_rec_data.size(), stream_ptr);
    check_cuda_error(err, "Malloc for data");
    // copy data to GPU
    for (std::uint64_t index = 0; index < p_rec_data.size(); index++) {
        p_rec_data[index]->copy_to_gpu(p_data + index, nullptr, stream.get_stream_ptr());
    }
    return p_data;
}

// ---------------------------------------------------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from the total number of elements
candy::train::GpuTrainer::GpuTrainer(std::uint64_t capacity, Synchronizer & synch) :
candy::train::TrainerBase(capacity), p_model_vectors_(capacity, nullptr), p_optimizer_dynamic_(capacity, nullptr) {
    // check if the synchronizer is on CPU
    if (const std::future<void> * future_ptr = std::get_if<std::future<void>>(&(synch.core))) {
        Fatal<std::runtime_error>("The synchronizer is allocated on CPU.");
    }
    this->p_synch_ = &synch;
    cuda::Stream & stream = std::get<cuda::Stream>(synch.core);
    cuda::CtxGuard guard(stream.get_gpu());
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    // allocate memory
    assume(capacity < 131072);
    this->capacity_ = capacity;
    ::cudaError_t err;
    err = ::cudaMallocAsync(&(this->p_model_), sizeof(candy::Model) * capacity, stream_ptr);
    check_cuda_error(err, "Malloc for models");
    err = ::cudaMallocAsync(&(this->p_optmz_), sizeof(candy::Optimizer) * capacity, stream_ptr);
    check_cuda_error(err, "Malloc for optimizers");
    err = ::cudaMallocAsync(&(this->p_data_), sizeof(array::Parcel) * capacity, stream_ptr);
    check_cuda_error(err, "Malloc for data");
}

// Add a model to trainer
void candy::train::GpuTrainer::set_model(const std::string & name, const candy::Model & model) {
    // get index
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[0] = true;
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    ::cudaError_t error;
    // de-allocate memory if another object has already present
    if (this->p_model_vectors_[index] != nullptr) {
        error = ::cudaFreeAsync(this->p_model_vectors_[index], stream_ptr);
        check_cuda_error(error, "Free parameter vector for model");
    }
    // allocate memory for new model
    error = ::cudaMallocAsync(&(this->p_model_vectors_[index]), model.cumalloc_size() - sizeof(candy::Model),
                              stream_ptr);
    check_cuda_error(error, "Malloc parameter vector for model");
    // copy to GPU
    model.copy_to_gpu(this->p_model_ + index, this->p_model_vectors_[index], stream.get_stream_ptr());
    // save extra details
    this->update_details(index, model);
    this->shared_mem_size_[0] = std::max(this->shared_mem_size_[0], model.sharedmem_size());
    this->shared_mem_size_[2] = std::max(this->shared_mem_size_[2], sizeof(double) * model.num_params());
}

// Add a optimizer to trainer
void candy::train::GpuTrainer::set_optmz(const std::string & name, const candy::Optimizer & optmz) {
    // get index
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[1] = true;
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    ::cudaError_t error;
    // de-allocate memory if another object has already present
    if (this->p_optimizer_dynamic_[index] != nullptr) {
        error = ::cudaFreeAsync(this->p_optimizer_dynamic_[index], stream_ptr);
        check_cuda_error(error, "Free dynamic data for optimizer");
    }
    // allocate memory for new model
    error = ::cudaMallocAsync(&(this->p_optimizer_dynamic_[index]), optmz.cumalloc_size() - sizeof(candy::Optimizer),
                              stream_ptr);
    check_cuda_error(error, "Malloc dynamic data for optimizer");
    // copy to GPU
    optmz.copy_to_gpu(this->p_optmz_ + index, this->p_optimizer_dynamic_[index], stream.get_stream_ptr());
    // save extra details
    this->shared_mem_size_[1] = std::max(this->shared_mem_size_[1], optmz.sharedmem_size());
}

// Add data to trainer
void candy::train::GpuTrainer::set_data(const std::string & name, const array::Parcel & data) {
    // get index
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[2] = true;
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // copy to GPU
    data.copy_to_gpu(this->p_data_ + index, nullptr);
}

// Get copy to a model
candy::Model candy::train::GpuTrainer::get_model(const std::string & name) {
    // check if model is initialized
    const std::pair<std::uint64_t, std::array<bool, 3>> & status = this->map_.at(name);
    if (!status.second[0]) {
        Fatal<std::runtime_error>("No model assigned to key \"%s\"", name.c_str());
    }
    // get index
    std::uint64_t index = status.first;
    candy::Model result(this->details_[index].first, this->details_[index].second);
    // change GPU
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // copy back to CPU
    result.copy_from_gpu(this->p_model_vectors_[index], 0);
    return result;
}

// Dry-run
void candy::train::GpuTrainer::dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                                       candy::TrialPolicy policy, std::uint64_t block_size, candy::TrainMetric metric) {
    // check if all elements are initialized
    this->check_complete();
    if (!candy::train::key_compare(tracking_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the tracking map to presents in the objects.\n");
    }
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // copy details to GPUs
    UIntVec cases(tracking_map.size());
    for (std::uint64_t i = 0; const auto & [name, pointers] : tracking_map) {
        cases[i++] = this->map_.at(name).first;
    }
    std::uint64_t error_size = policy.sum();
    DoubleVec errors(tracking_map.size() * error_size);
    UIntVec count(tracking_map.size());
    cuda::Memory mem(stream.get_stream_ptr(), cases, errors, count);
    // launch
    std::uint64_t shared_mem_size = std::accumulate(this->shared_mem_size_.begin(), this->shared_mem_size_.end(),
                                                    sizeof(array::Parcel));
    std::uint64_t * p_cases = reinterpret_cast<std::uint64_t *>(mem.get<0>() + 1);
    double * p_error = reinterpret_cast<double *>(mem.get<1>() + 1);
    std::uint64_t * p_count = reinterpret_cast<std::uint64_t *>(mem.get<2>() + 1);
    candy::train::launch_dry_run(this->p_model_, this->p_optmz_, this->p_data_, p_cases, p_error, p_count,
                                 tracking_map.size(), policy, metric, block_size, shared_mem_size,
                                 stream.get_stream_ptr());
    // copy to CPU
    ::cudaError_t error;
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    for (std::uint64_t i = 0; const auto & [name, pointers] : tracking_map) {
        error = ::cudaMemcpyAsync(pointers.first, p_error + i * error_size, error_size * sizeof(double),
                                  ::cudaMemcpyDeviceToHost, stream_ptr);
        check_cuda_error(error, "Copy errors");
        error = ::cudaMemcpyAsync(pointers.second, p_count + i, sizeof(std::uint64_t), ::cudaMemcpyDeviceToHost,
                                  stream_ptr);
        check_cuda_error(error, "Copy count");
        i++;
    }
}

// Update the CP model according to the gradient using GPU parallelism until a specified threshold is met
void candy::train::GpuTrainer::update_until(std::uint64_t rep, double threshold, std::uint64_t block_size,
                                            candy::TrainMetric metric, bool export_result) {
    // warning if export_result is True
    if (export_result) {
        Warning("Cannot export trained models asynchronously. "
                "Use \"export_models\" after synchronization of the stream instead.\n");
    }
    // check if all elements are initialized
    this->check_complete();
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // launch
    std::uint64_t shared_mem_size = std::accumulate(this->shared_mem_size_.begin(), this->shared_mem_size_.end(),
                                                    sizeof(array::Parcel));
    candy::train::launch_update_until(this->p_model_, this->p_optmz_, this->p_data_, this->size_, rep, threshold,
                                      block_size, metric, shared_mem_size, stream.get_stream_ptr());
}

// Update CP model according to gradient using GPU for a given number of iterations
void candy::train::GpuTrainer::update_for(std::uint64_t max_iter, std::uint64_t block_size, candy::TrainMetric metric,
                                          bool export_result) {
    // warning if export_result is True
    if (export_result) {
        Warning("Cannot export trained models asynchronously. "
                "Use \"export_models\" after synchronization of the stream instead.\n");
    }
    // check if all elements are initialized
    this->check_complete();
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // launch
    std::uint64_t shared_mem_size = std::accumulate(this->shared_mem_size_.begin(), this->shared_mem_size_.end(),
                                                    sizeof(array::Parcel));
    candy::train::launch_update_for(this->p_model_, this->p_optmz_, this->p_data_, this->size_, max_iter, block_size,
                                    metric, shared_mem_size, stream.get_stream_ptr());
}

// Reconstruct a whole multi-dimensional data from the model using GPU parallelism
void candy::train::GpuTrainer::reconstruct(const std::map<std::string, array::Parcel *> & rec_data_map,
                                           std::uint64_t block_size) {
    // check models
    this->check_models();
    if ((this->map_.size() != rec_data_map.size()) || !candy::train::key_compare(rec_data_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the reconstructed map to be the same as the objects.\n");
    }
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // copy pointers
    std::vector<array::Parcel *> p_rec_data(this->map_.size(), nullptr);
    for (auto & [name, rec_data] : rec_data_map) {
        p_rec_data[this->map_.at(name).first] = rec_data;
    }
    // transfer vector to GPU
    array::Parcel * gpu_data = transfer_parcels(p_rec_data, stream);
    std::uint64_t shared_mem_size = this->shared_mem_size_[0] + sizeof(array::Parcel);
    candy::train::launch_reconstruct(this->p_model_, gpu_data, this->size_, block_size, shared_mem_size,
                                     stream.get_stream_ptr());
    // free data
    ::cudaError_t error;
    error = ::cudaFreeAsync(gpu_data, reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
    check_cuda_error(error, "Free memory of reconstructed data");
}

// Get the RMSE and RMAE error with respect to the training data
void candy::train::GpuTrainer::get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                                         std::uint64_t block_size) {
    // check argument
    this->check_models();
    if ((this->map_.size() != error_map.size()) || !candy::train::key_compare(error_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the error map to be the same as the objects.\n");
    }
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // allocate memory for error on GPU
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    ::cudaError_t error;
    double * p_error_gpu;
    error = ::cudaMallocAsync(&p_error_gpu, 2 * error_map.size() * sizeof(double), stream_ptr);
    check_cuda_error(error, "Malloc data for errors");
    // launch
    std::uint64_t shared_mem_size = this->shared_mem_size_[0] + sizeof(array::Parcel);
    candy::train::launch_get_error(this->p_model_, this->p_data_, p_error_gpu, this->size_, block_size, shared_mem_size,
                                   stream.get_stream_ptr());
    // copy back the result to CPU
    for (auto & [name, errors_cpu] : error_map) {
        std::uint64_t index = this->map_.at(name).first;
        error = cudaMemcpyAsync(errors_cpu[0], p_error_gpu + 2 * index, sizeof(double), ::cudaMemcpyDeviceToHost,
                                stream_ptr);
        check_cuda_error(error, "Copy RMSE");
        error = cudaMemcpyAsync(errors_cpu[1], p_error_gpu + 2 * index + 1, sizeof(double), ::cudaMemcpyDeviceToHost,
                                stream_ptr);
        check_cuda_error(error, "Copy RMAE");
    }
    // free memory
    error = ::cudaFreeAsync(p_error_gpu, stream_ptr);
    check_cuda_error(error, "Free memory of error vector on GPU");
}

// Export all models to output directory
void candy::train::GpuTrainer::export_models(void) {
    // check models
    this->check_models();
    // chnage current GPU
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    // export each model
    for (std::uint64_t i_case = 0; i_case < this->size_; i_case++) {
        if (this->export_fnames_[i_case].empty()) {
            continue;
        }
        candy::Model copied_model(this->details_[i_case].first, this->details_[i_case].second);
        copied_model.copy_from_gpu(this->p_model_vectors_[i_case], 0);
        copied_model.save(this->export_fnames_[i_case], true);
    }
}

// Free data
void candy::train::GpuTrainer::free_memory(void) {
    // get stream
    if (this->p_synch_ == nullptr || this->capacity_ == 0) {
        return;
    }
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    ::cudaStream_t stream_ptr = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    cuda::CtxGuard guard(stream.get_gpu());
    // free memory on GPU
    ::cudaFreeAsync(this->p_model_, stream_ptr);
    ::cudaFreeAsync(this->p_optmz_, stream_ptr);
    ::cudaFreeAsync(this->p_data_, stream_ptr);
    for (std::uint64_t i_case = 0; i_case < this->size_; i_case++) {
        if (double * model_vectors = this->p_model_vectors_[i_case]; model_vectors != nullptr) {
            ::cudaFreeAsync(model_vectors, stream_ptr);
        }
        if (double * optimizer_dynamic = this->p_optimizer_dynamic_[i_case]; optimizer_dynamic != nullptr) {
            ::cudaFreeAsync(optimizer_dynamic, stream_ptr);
        }
    }
    // reset to default
    this->capacity_ = 0;
    this->p_model_ = nullptr;
    this->p_optmz_ = nullptr;
    this->p_data_ = nullptr;
    this->p_model_vectors_.resize(0);
    this->p_optimizer_dynamic_.resize(0);
    this->shared_mem_size_.fill(0);
}

}  // namespace merlin
