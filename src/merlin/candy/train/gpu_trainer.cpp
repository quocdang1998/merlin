// Copyright 2024 quocdang1998
#include "merlin/candy/train/gpu_trainer.hpp"

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // merlin::cuda_compile_error, merlin::Fatal


namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------------------------------------------------

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

#ifndef __MERLIN_CUDA__

// Constructor from the total number of elements
candy::train::GpuTrainer::GpuTrainer(std::uint64_t capacity, Synchronizer & synch) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Add a model to trainer
void candy::train::GpuTrainer::set_model(const std::string & name, const candy::Model & model) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Add a optimizer to trainer
void candy::train::GpuTrainer::set_optmz(const std::string & name, const candy::Optimizer & optmz) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

// Add data to trainer
void candy::train::GpuTrainer::set_data(const std::string & name, const array::Parcel & data) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use GpuTrainer.\n");
}

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
