// Copyright 2024 quocdang1998
#include "merlin/candy/train/gpu_trainer.hpp"

#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/assume.hpp"           // merlin::assume
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/cuda/device.hpp"      // merlin::cuda::CtxGuard
#include "merlin/cuda/stream.hpp"      // merlin::cuda::Stream
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/memory.hpp"           // merlin::mem_alloc_device, merlin::mem_free_device

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from the total number of elements
candy::train::GpuTrainer::GpuTrainer(std::uint64_t capacity, Synchronizer & synch) :
candy::train::TrainerBase(capacity), p_model_vectors_(capacity, nullptr), p_optimizer_dynamic_(capacity, nullptr) {
    // check if the synchronizer is on CPU
    if (const std::future<void> * future_ptr = std::get_if<std::future<void>>(&(synch.core))) {
        Fatal<std::runtime_error>("The synchronizer is allocated on CPU.\n");
    }
    this->p_synch_ = &synch;
    cuda::Stream & stream = std::get<cuda::Stream>(synch.core);
    cuda::CtxGuard guard(stream.get_gpu());
    // allocate memory
    assume(capacity < 131072);
    this->capacity_ = capacity;
    std::uintptr_t stream_ptr = stream.get_stream_ptr();
    mem_alloc_device(reinterpret_cast<void**>(&(this->p_model_)), sizeof(candy::Model) * capacity, stream_ptr);
    mem_alloc_device(reinterpret_cast<void**>(&(this->p_optmz_)), sizeof(candy::Optimizer) * capacity, stream_ptr);
    mem_alloc_device(reinterpret_cast<void**>(&(this->p_data_)), sizeof(array::Parcel) * capacity, stream_ptr);
}

// Move assignment
candy::train::GpuTrainer & candy::train::GpuTrainer::operator=(candy::train::GpuTrainer && src) {
    this->free_memory();
    this->candy::train::TrainerBase::operator=(std::forward<candy::train::GpuTrainer>(src));
    this->p_model_vectors_ = std::move(src.p_model_vectors_);
    this->p_optimizer_dynamic_ = std::move(src.p_optimizer_dynamic_);
    this->p_data_ = std::exchange(src.p_data_, nullptr);
    this->shared_mem_size_ = std::move(src.shared_mem_size_);
    return *this;
}

// Add a model to trainer
void candy::train::GpuTrainer::set_model(const std::string & name, const candy::Model & model) {
    // get index
    std::uint64_t index = this->get_index_or_create_key(name);
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    std::uintptr_t stream_ptr = stream.get_stream_ptr();
    cuda::CtxGuard guard(stream.get_gpu());
    // de-allocate memory if another object has already present
    if (this->p_model_vectors_[index] != nullptr) {
        mem_free_device(this->p_model_vectors_[index], stream_ptr);
    }
    // allocate memory for new model
    mem_alloc_device(reinterpret_cast<void **>(&(this->p_model_vectors_[index])),
                     model.cumalloc_size() - sizeof(candy::Model), stream_ptr);
    // copy to GPU
    model.copy_to_gpu(this->p_model_ + index, this->p_model_vectors_[index], stream_ptr);
    // save extra details
    this->map_.at(name).second[0] = true;
    this->update_details(index, model);
    this->shared_mem_size_[0] = std::max(this->shared_mem_size_[0], model.sharedmem_size());
    this->shared_mem_size_[2] = std::max(this->shared_mem_size_[2], sizeof(double) * model.num_params());
}

// Add a optimizer to trainer
void candy::train::GpuTrainer::set_optmz(const std::string & name, const candy::Optimizer & optmz) {
    // get index
    std::uint64_t index = this->get_index_or_create_key(name);
    // get stream
    cuda::Stream & stream = std::get<cuda::Stream>(this->p_synch_->core);
    cuda::CtxGuard guard(stream.get_gpu());
    std::uintptr_t stream_ptr = stream.get_stream_ptr();
    // de-allocate memory if another object has already present
    if (this->p_optimizer_dynamic_[index] != nullptr) {
        mem_free_device(this->p_optimizer_dynamic_[index], stream_ptr);
    }
    // allocate memory for new model
    mem_alloc_device(reinterpret_cast<void **>(&(this->p_optimizer_dynamic_[index])),
                     optmz.cumalloc_size() - sizeof(candy::Optimizer), stream_ptr);
    // copy to GPU
    optmz.copy_to_gpu(this->p_optmz_ + index, this->p_optimizer_dynamic_[index], stream.get_stream_ptr());
    // save extra details
    this->map_.at(name).second[1] = true;
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

#ifndef __MERLIN_CUDA__

// Get copy to a model
candy::Model candy::train::GpuTrainer::get_model(const std::string & name) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
    return candy::Model();
}

// Dry-run
void candy::train::GpuTrainer::dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                                       candy::TrialPolicy policy, std::uint64_t block_size, candy::TrainMetric metric) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Update the CP model according to the gradient using GPU parallelism until a specified threshold is met
void candy::train::GpuTrainer::update_until(std::uint64_t rep, double threshold, std::uint64_t block_size,
                                            candy::TrainMetric metric, bool export_result) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Update CP model according to gradient using GPU for a given number of iterations
void candy::train::GpuTrainer::update_for(std::uint64_t max_iter, std::uint64_t block_size, candy::TrainMetric metric,
                                          bool export_result) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Reconstruct a whole multi-dimensional data from the model using GPU parallelism
void candy::train::GpuTrainer::reconstruct(const std::map<std::string, array::Parcel *> & rec_data_map,
                                           std::uint64_t block_size) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Get the RMSE and RMAE error with respect to the training data
void candy::train::GpuTrainer::get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                                         std::uint64_t block_size) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Export all models to output directory
void candy::train::GpuTrainer::export_models(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Free data
void candy::train::GpuTrainer::free_memory(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
