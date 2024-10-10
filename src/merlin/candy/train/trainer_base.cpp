// Copyright 2024 quocdang1998
#include "merlin/candy/train/trainer_base.hpp"

#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/logger.hpp"       // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// TrainerBase
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from the capacity
candy::train::TrainerBase::TrainerBase(std::uint64_t capacity) :
details_(capacity), export_fnames_(capacity, std::string()) {
    for (std::uint64_t i_case = 0; i_case < capacity; i_case++) {
        this->details_[i_case].first = Index();
        this->details_[i_case].second = 0;
    }
}

// Add exported names to model
void candy::train::TrainerBase::set_export_fname(const std::string & name, const std::string & export_fname) {
    std::uint64_t index = this->get_index_or_create_key(name);
    this->export_fnames_.at(index) = export_fname;
}

// Get list of keys
std::vector<std::string> candy::train::TrainerBase::get_keys(void) {
    std::vector<std::string> keys;
    keys.resize(this->size_);
    for (const auto & [key, value] : this->map_) {
        keys[value.first] = key;
    }
    return keys;
}

// Check if each key is assigned to a model, an optimizer and a data
bool candy::train::TrainerBase::is_complete(void) {
    auto check_lambda = [](candy::train::IndicatorMap::const_reference item) {
        return std::all_of(item.second.second.begin(), item.second.second.end(), [](bool x) { return x; });
    };
    return std::all_of(this->map_.begin(), this->map_.end(), check_lambda);
}

// Get index corresponding to a given key
std::uint64_t candy::train::TrainerBase::get_index_or_create_key(const std::string & name) {
    std::uint64_t index;
    if (this->map_.contains(name)) {
        index = this->map_.at(name).first;
    } else {
        if (this->is_full()) {
            Fatal<std::runtime_error>("Maximum capacity reached.\n");
        }
        std::pair<std::uint64_t, std::array<bool, 3>> map_value(this->size_, {false, false, false});
        this->map_.insert(std::make_pair(name, map_value));
        index = this->size_;
        this->size_ += 1;
    }
    return index;
}

// Add model shape and total number of parameters to the detail array
void candy::train::TrainerBase::update_details(std::uint64_t index, const candy::Model & model) {
    this->details_[index].first = model.shape();
    this->details_[index].second = model.rank();
}

// Check if all models are assigned
void candy::train::TrainerBase::check_models(void) {
    for (const auto & [name, map_value] : this->map_) {
        if (!map_value.second[0]) {
            Fatal<std::runtime_error>("Model at key %s is not assigned.\n", name.c_str());
        }
    }
}

// Check if all models, optimizers and data are assigned
void candy::train::TrainerBase::check_complete(void) {
    for (const auto & [name, map_value] : this->map_) {
        if (!std::all_of(map_value.second.begin(), map_value.second.end(), [](bool x) { return x; })) {
            Fatal<std::runtime_error>("Key %s is incomplet.\n", name.c_str());
        }
    }
}

}  // namespace merlin
