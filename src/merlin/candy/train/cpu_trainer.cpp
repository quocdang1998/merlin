// Copyright 2024 quocdang1998
#include "merlin/candy/train/cpu_trainer.hpp"

#include <algorithm>    // std::for_each
#include <future>       // std::async, std::future
#include <numeric>      // std::iota
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple
#include <utility>      // std::move

#include <omp.h>  // ::omp_get_thread_num

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/assume.hpp"           // merlin::assume
#include "merlin/candy/gradient.hpp"   // merlin::candy::Gradient
#include "merlin/candy/loss.hpp"       // merlin::candy::rmae_cpu, merlin::candy::rmse_cpu
#include "merlin/candy/model.hpp"      // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/utils.hpp"            // merlin::is_finite, merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Dry-run
void candy::train::run_dry_run(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                               candy::Model * p_model, const array::Array ** p_data, candy::Optimizer * p_optimizer,
                               std::vector<std::tuple<std::uint64_t, double *, std::uint64_t *>> && tracker,
                               candy::TrainMetric metric, std::uint64_t n_threads, candy::TrialPolicy policy) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // reduce the task
    n_threads = std::min(n_threads, tracker.size());
    // parallel dry-run for each model
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        std::uint64_t normal_count;
        for (std::uint64_t i_case = thread_idx; i_case < tracker.size(); i_case += n_threads) {
            std::uint64_t & count = *(std::get<2>(tracker[i_case]));
            count = 1;
            double * error = std::get<1>(tracker[i_case]);
            // clone elements
            std::uint64_t index = std::get<0>(tracker[i_case]);
            candy::Model model(p_model[index]);
            candy::Optimizer optimizer(p_optimizer[index]);
            const array::Array & data = *(p_data[index]);
            // initialize gradient
            DoubleVec grad_mem(model.num_params());
            candy::Gradient gradient(grad_mem.data(), model.num_params(), metric);
            // calculate error before training
            Index index_mem;
            index_mem.fill(0);
            candy::rmse_cpu(&model, &data, error[0], normal_count, index_mem);
            // discarded phase
            std::uint64_t start = 0;
            std::uint64_t loops = policy.discarded();
            for (std::uint64_t iter = 1; iter < loops; iter++) {
                gradient.calc_by_cpu(model, data, index_mem);
                optimizer.update_cpu(model, gradient, start + iter);
                candy::rmse_cpu(&model, &data, error[start + iter], normal_count, index_mem);
                bool break_condition = !is_finite(error[start + iter]);
                if (break_condition) {
                    break;
                }
                count += 1;
            }
            loops = (count == start + loops) ? (policy.strict()) : 0;
            start += policy.discarded();
            // strictly descent phase
            for (std::uint64_t iter = 0; iter < loops; iter++) {
                gradient.calc_by_cpu(model, data, index_mem);
                optimizer.update_cpu(model, gradient, start + iter);
                candy::rmse_cpu(&model, &data, error[start + iter], normal_count, index_mem);
                bool break_condition = !is_finite(error[start + iter]) ||
                                       (error[start + iter] >= strict_max_ratio * error[start + iter - 1]);
                if (break_condition) {
                    break;
                }
                count += 1;
            }
            loops = (count == start + loops) ? (policy.loose()) : 0;
            start += policy.strict();
            // loose phase
            for (std::uint64_t iter = 0; iter < loops; iter++) {
                gradient.calc_by_cpu(model, data, index_mem);
                optimizer.update_cpu(model, gradient, start + iter);
                candy::rmse_cpu(&model, &data, error[start + iter], normal_count, index_mem);
                bool break_condition = !is_finite(error[start + iter]) ||
                                       (error[start + iter] >= loose_max_ratio * error[start + iter - 1]);
                if (break_condition) {
                    break;
                }
                count += 1;
            }
        }
    }
}

// Train a model until a given threshold is met
void candy::train::run_update_until(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                    candy::Model * p_model, const array::Array ** p_data,
                                    candy::Optimizer * p_optimizer, std::uint64_t size, candy::TrainMetric metric,
                                    std::uint64_t rep, double threshold, std::uint64_t n_threads, bool export_result,
                                    const std::string * p_export_fnames) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // create map from index to ID
    std::vector<std::string_view> names(size);
    for (const auto & [name, value] : *p_index) {
        names[value.first] = name;
    }
    // reduce the task
    n_threads = std::min(n_threads, size);
    // parallel training for each model
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        for (std::uint64_t i_case = thread_idx; i_case < size; i_case += n_threads) {
            // get elements
            candy::Model & model = p_model[i_case];
            const array::Array & data = *(p_data[i_case]);
            candy::Optimizer & optimizer = p_optimizer[i_case];
            // initialize gradient
            DoubleVec grad_mem(model.num_params());
            candy::Gradient gradient(grad_mem.data(), model.num_params(), metric);
            // calculate based on error
            double priori_error = 0.0;
            double posteriori_error;
            std::uint64_t normal_count;
            bool go_on = true;
            // calculate error before training
            Index index_mem;
            index_mem.fill(0);
            candy::rmse_cpu(&model, &data, posteriori_error, normal_count, index_mem);
            // training loop
            std::uint64_t step = 1;
            do {
                // move error to register
                priori_error = posteriori_error;
                // gradient descent loop
                for (std::uint64_t iter = 0; iter < rep; iter++) {
                    gradient.calc_by_cpu(model, data, index_mem);
                    optimizer.update_cpu(model, gradient, step + iter);
                }
                // measure relative error
                candy::rmse_cpu(&model, &data, posteriori_error, normal_count, index_mem);
                double rel_err = std::abs(priori_error - posteriori_error) / posteriori_error;
                // decide whether to continue
                go_on = (is_finite(posteriori_error)) ? (rel_err > threshold) : false;
                step += rep;
            } while (go_on);
            // save model to file
            if (export_result && !p_export_fnames[i_case].empty()) {
                model.save(p_export_fnames[i_case], true);
            }
        }
    }
}

// Train a model for a fixed number of iterations
void candy::train::run_update_for(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                  candy::Model * p_model, const array::Array ** p_data, candy::Optimizer * p_optimizer,
                                  std::uint64_t size, candy::TrainMetric metric, std::uint64_t max_iter,
                                  std::uint64_t n_threads, bool export_result, const std::string * p_export_fnames) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // create map from index to ID
    std::vector<std::string_view> names(size);
    for (const auto & [name, value] : *p_index) {
        names[value.first] = name;
    }
    // reduce the task
    n_threads = std::min(n_threads, size);
    // parallel dry-run for each model
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        for (std::uint64_t i_case = thread_idx; i_case < size; i_case += n_threads) {
            // get elements
            candy::Model & model = p_model[i_case];
            const array::Array & data = *(p_data[i_case]);
            candy::Optimizer & optimizer = p_optimizer[i_case];
            // initialize gradient
            DoubleVec grad_mem(model.num_params());
            candy::Gradient gradient(grad_mem.data(), model.num_params(), metric);
            // initialize index
            Index index_mem;
            index_mem.fill(0);
            // training loop
            for (std::uint64_t iter = 1; iter <= max_iter; iter++) {
                gradient.calc_by_cpu(model, data, index_mem);
                optimizer.update_cpu(model, gradient, iter);
            }
            // save model to file
            if (export_result && !p_export_fnames[i_case].empty()) {
                model.save(p_export_fnames[i_case], true);
            }
        }
    }
}

// Reconstruct data
void candy::train::run_reconstruct(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                   candy::Model * p_model, std::vector<array::Array *> && p_rec_data,
                                   std::uint64_t size, std::uint64_t n_threads) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // reduce the task
    n_threads = std::min(n_threads, size);
    // parallel reconstructing for each model
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        for (std::uint64_t i_case = thread_idx; i_case < size; i_case += n_threads) {
            // get elements
            candy::Model & model = p_model[i_case];
            array::Array & rec_data = *(p_rec_data[i_case]);
            // initialize index
            Index index;
            index.fill(0);
            // reconstruct
            for (std::uint64_t c_index = 0; c_index < rec_data.size(); c_index++) {
                contiguous_to_ndim_idx(c_index, rec_data.shape().data(), rec_data.ndim(), index.data());
                rec_data[index] = model.eval(index);
            }
        }
    }
}

// Calculate error
void candy::train::run_get_error(std::future<void> && synch, const candy::train::IndicatorMap * p_index,
                                 candy::Model * p_model, const array::Array ** p_data, std::uint64_t size,
                                 std::vector<std::array<double *, 2>> && p_error, std::uint64_t n_threads) {
    // finish old job
    if (synch.valid()) {
        synch.get();
    }
    // reduce the task
    n_threads = std::min(n_threads, size);
    // parallel reconstructing for each model
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        std::uint64_t normal_count;
        for (std::uint64_t i_case = thread_idx; i_case < size; i_case += n_threads) {
            // get elements
            candy::Model & model = p_model[i_case];
            const array::Array & data = *(p_data[i_case]);
            // initialize index
            Index index;
            index.fill(0);
            // calculate error
            std::uint64_t thread_idx = ::omp_get_thread_num();
            candy::rmse_cpu(&model, &data, *(p_error[i_case][0]), normal_count, index);
            candy::rmae_cpu(&model, &data, *(p_error[i_case][1]), normal_count, index);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CpuTrainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from the total number of elements
candy::train::CpuTrainer::CpuTrainer(std::uint64_t capacity, Synchronizer & synch) :
candy::train::TrainerBase(capacity), p_data_(capacity, nullptr) {
    // check if the synchronizer is on CPU
    if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(synch.core))) {
        Fatal<std::runtime_error>("The synchronizer is allocated on GPU.\n");
    }
    this->p_synch_ = &synch;
    // allocate memory
    assume(capacity < 131072);
    this->capacity_ = capacity;
    this->p_model_ = new candy::Model[capacity];
    this->p_optmz_ = new candy::Optimizer[capacity];
    // assign default
    for (std::uint64_t i_case = 0; i_case < capacity; i_case++) {
        new (this->p_model_ + i_case) candy::Model();
        new (this->p_optmz_ + i_case) candy::Optimizer();
    }
}

// Move assignment
candy::train::CpuTrainer & candy::train::CpuTrainer::operator=(candy::train::CpuTrainer && src) {
    if (this->capacity_ != 0) {
        delete[] this->p_model_;
        delete[] this->p_optmz_;
    }
    this->candy::train::TrainerBase::operator=(std::forward<candy::train::CpuTrainer>(src));
    this->p_data_ = std::move(src.p_data_);
    return *this;
}

// Add a model to trainer
void candy::train::CpuTrainer::set_model(const std::string & name, const candy::Model & model) {
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[0] = true;
    this->p_model_[index] = model;
    this->update_details(index, model);
}

// Add a optimizer to trainer
void candy::train::CpuTrainer::set_optmz(const std::string & name, const candy::Optimizer & optmz) {
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[1] = true;
    this->p_optmz_[index] = optmz;
}

// Add data to trainer
void candy::train::CpuTrainer::set_data(const std::string & name, const array::Array & data) {
    std::uint64_t index = this->get_index_or_create_key(name);
    this->map_.at(name).second[2] = true;
    this->p_data_[index] = &data;
}

// Get reference to a model
candy::Model & candy::train::CpuTrainer::get_model(const std::string & name) {
    const std::pair<std::uint64_t, std::array<bool, 3>> & status = this->map_.at(name);
    if (!status.second[0]) {
        Fatal<std::runtime_error>("No model assigned to key \"%s\".\n", name.c_str());
    }
    return this->p_model_[status.first];
}

// Get reference to an optimizer
candy::Optimizer & candy::train::CpuTrainer::get_optmz(const std::string & name) {
    const std::pair<std::uint64_t, std::array<bool, 3>> & status = this->map_.at(name);
    if (!status.second[1]) {
        Fatal<std::runtime_error>("No optimizer assigned to key \"%s\".\n", name.c_str());
    }
    return this->p_optmz_[status.first];
}

// Dry-run
void candy::train::CpuTrainer::dry_run(const std::map<std::string, std::pair<double *, std::uint64_t *>> & tracking_map,
                                       candy::TrialPolicy policy, std::uint64_t n_threads, candy::TrainMetric metric) {
    // check argument
    this->check_complete();
    if (!candy::train::key_compare(tracking_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the tracking map to presents in the objects.\n");
    }
    // map the tracking to a vector
    std::vector<std::tuple<std::uint64_t, double *, std::uint64_t *>> tracker;
    tracker.reserve(tracking_map.size());
    for (const auto & [name, pointers] : tracking_map) {
        std::uint64_t i_case = this->map_.at(name).first;
        tracker.emplace_back(i_case, pointers.first, pointers.second);
    }
    // launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train::run_dry_run, std::move(current_sync),
                                            &(this->map_), this->p_model_, this->p_data_.data(), this->p_optmz_,
                                            std::move(tracker), metric, n_threads, policy);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Update the CP model according to the gradient using CPU parallelism until a specified threshold is met
void candy::train::CpuTrainer::update_until(std::uint64_t rep, double threshold, std::uint64_t n_threads,
                                            candy::TrainMetric metric, bool export_result) {
    this->check_complete();
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train::run_update_until, std::move(current_sync),
                                            &(this->map_), this->p_model_, this->p_data_.data(), this->p_optmz_,
                                            this->size_, metric, rep, threshold, n_threads, export_result,
                                            this->export_fnames_.data());
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Update CP model according to gradient using CPU for a given number of iterations.
void candy::train::CpuTrainer::update_for(std::uint64_t max_iter, std::uint64_t n_threads, candy::TrainMetric metric,
                                          bool export_result) {
    this->check_complete();
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train::run_update_for, std::move(current_sync),
                                            &(this->map_), this->p_model_, this->p_data_.data(), this->p_optmz_,
                                            this->size_, metric, max_iter, n_threads, export_result,
                                            this->export_fnames_.data());
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Reconstruct a whole multi-dimensional data from the model using CPU parallelism
void candy::train::CpuTrainer::reconstruct(const std::map<std::string, array::Array *> & rec_data_map,
                                           std::uint64_t n_threads) {
    // check argument
    this->check_models();
    if ((this->map_.size() != rec_data_map.size()) || !candy::train::key_compare(rec_data_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the reconstructed map to be the same as the objects.\n");
    }
    // copy pointers
    std::vector<array::Array *> p_rec_data(this->map_.size(), nullptr);
    for (auto & [name, rec_data] : rec_data_map) {
        p_rec_data[this->map_.at(name).first] = rec_data;
    }
    // check shape
    for (std::uint64_t i_case = 0; i_case < this->size_; i_case++) {
        if (!this->p_model_[i_case].check_compatible_shape(p_rec_data[i_case]->shape())) {
            Fatal<std::runtime_error>("Model at index %" PRIu64 " is not compatible with destination array.\n", i_case);
        }
    }
    // launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train::run_reconstruct, std::move(current_sync),
                                            &(this->map_), this->p_model_, std::move(p_rec_data), this->size_,
                                            n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Get the RMSE and RMAE error with respect to the training data
void candy::train::CpuTrainer::get_error(const std::map<std::string, std::array<double *, 2>> & error_map,
                                         std::uint64_t n_threads) {
    // check argument
    this->check_models();
    if ((this->map_.size() != error_map.size()) || !candy::train::key_compare(error_map, this->map_)) {
        Fatal<std::runtime_error>("Expected the keys in the error map to be the same as the objects.\n");
    }
    // copy pointers
    std::vector<std::array<double *, 2>> p_error(this->map_.size(), {nullptr, nullptr});
    for (auto & [name, error_array] : error_map) {
        p_error[this->map_.at(name).first] = error_array;
    }
    // launch
    std::future<void> & current_sync = std::get<std::future<void>>(this->p_synch_->core);
    std::future<void> new_sync = std::async(std::launch::async, candy::train::run_get_error, std::move(current_sync),
                                            &(this->map_), this->p_model_, this->p_data_.data(), this->size_,
                                            std::move(p_error), n_threads);
    *(this->p_synch_) = Synchronizer(std::move(new_sync));
}

// Export all models to output directory
void candy::train::CpuTrainer::export_models(void) {
    this->check_models();
    for (std::uint64_t i_case = 0; i_case < this->size_; i_case++) {
        if (this->export_fnames_[i_case].empty()) {
            continue;
        }
        this->p_model_[i_case].save(this->export_fnames_[i_case], true);
    }
}

// Default destructor
candy::train::CpuTrainer::~CpuTrainer(void) {
    if (this->capacity_ != 0) {
        delete[] this->p_model_;
        delete[] this->p_optmz_;
    }
}

}  // namespace merlin
