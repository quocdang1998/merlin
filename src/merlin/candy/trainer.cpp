// Copyright 2023 quocdang1998
#include "merlin/candy/trainer.hpp"

#include "merlin/array/array.hpp"              // merlin::array::Array
#include "merlin/array/nddata.hpp"             // merlin::array::NdData
#include "merlin/array/parcel.hpp"             // merlin::array::Parcel
#include "merlin/candy/model.hpp"              // merlin::candy::Model
#include "merlin/candy/train/cpu_trainer.hpp"  // merlin::candy::train::CpuTrainer
#include "merlin/candy/train/gpu_trainer.hpp"  // merlin::candy::train::GpuTrainer
#include "merlin/logger.hpp"                   // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

template <class ArrayType>
std::map<std::string, ArrayType *> convert_map(const std::map<std::string, array::NdData *> & map) {
    std::map<std::string, ArrayType *> converted_map;
    for (auto & [name, p_data] : map) {
        ArrayType * p_converted = dynamic_cast<ArrayType *>(p_data);
        if (p_converted == nullptr) {
            Fatal<std::runtime_error>("Wrong array type provided.\n");
        }
        converted_map[name] = p_converted;
    }
    return converted_map;
}

// ---------------------------------------------------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from the total number of elements
candy::Trainer::Trainer(std::uint64_t capacity, Synchronizer & synch) {
    switch (synch.core.index()) {
        case 0 : {
            this->p_core_ = new candy::train::CpuTrainer(capacity, synch);
            break;
        }
        case 1 : {
            this->p_core_ = new candy::train::GpuTrainer(capacity, synch);
            break;
        }
    }
}

// Assign data to an ID
void candy::Trainer::set_data(const std::string & name, const array::NdData & data) {
    bool on_gpu = this->on_gpu();
    if (const array::Array * p_data = dynamic_cast<const array::Array *>(&data); p_data != nullptr) {
        if (on_gpu) {
            Fatal<std::runtime_error>("Provided CPU array to GPU initialized object.\n");
        }
        static_cast<candy::train::CpuTrainer *>(this->p_core_)->set_data(name, *p_data);
    } else if (const array::Parcel * p_data = dynamic_cast<const array::Parcel *>(&data); p_data != nullptr) {
        if (!on_gpu) {
            Fatal<std::runtime_error>("Provided GPU array to CPU initialized object.\n");
        }
        static_cast<candy::train::GpuTrainer *>(this->p_core_)->set_data(name, *p_data);
    } else {
        Fatal<std::runtime_error>("Array other than on CPU or GPU are not supported.\n");
    }
}

// Query if the object data is instantiated on CPU or on GPU
bool candy::Trainer::on_gpu(void) {
    if (this->p_core_ == nullptr) {
        Fatal<std::runtime_error>("Trainer not initialized.\n");
    }
    if (candy::train::GpuTrainer * p_core = dynamic_cast<candy::train::GpuTrainer *>(this->p_core_);
        p_core != nullptr) {
        return true;
    }
    return false;
}

// Get copy to a model
candy::Model candy::Trainer::get_model(const std::string & name) {
    bool on_gpu = this->on_gpu();
    candy::Model target_model;
    if (!on_gpu) {
        target_model = static_cast<candy::train::CpuTrainer *>(this->p_core_)->get_model(name);
    } else {
        target_model = static_cast<candy::train::GpuTrainer *>(this->p_core_)->get_model(name);
    }
    return target_model;
}

// Reconstruct a whole multi-dimensional data from the model using CPU parallelism
void candy::Trainer::reconstruct(const std::map<std::string, array::NdData *> & rec_data_map, std::uint64_t n_threads) {
    bool on_gpu = this->on_gpu();
    if (!on_gpu) {
        std::map<std::string, array::Array *> cpu_rec_data_map = convert_map<array::Array>(rec_data_map);
        static_cast<candy::train::CpuTrainer *>(this->p_core_)->reconstruct(cpu_rec_data_map, n_threads);
    } else {
        std::map<std::string, array::Parcel *> gpu_rec_data_map = convert_map<array::Parcel>(rec_data_map);
        static_cast<candy::train::GpuTrainer *>(this->p_core_)->reconstruct(gpu_rec_data_map, n_threads);
    }
}

// Destructor
candy::Trainer::~Trainer(void) {
    if (this->p_core_ != nullptr) {
        delete this->p_core_;
    }
}

}  // namespace merlin
