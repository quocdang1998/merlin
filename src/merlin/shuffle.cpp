// Copyright 2023 quocdang1998
#include "merlin/shuffle.hpp"

#include <algorithm>  // std::shuffle
#include <cstdint>    // std::uint64_t
#include <sstream>    // std::ostringstream

#include "merlin/env.hpp"  // merlin::Environment

namespace merlin {

// Random engine
std::mt19937_64 & Shuffle::random_generator_ = Environment::random_generator;

// Create index vector
static intvec initialize_index_vector(std::uint64_t size) {
    intvec result(size);
    for (std::uint64_t index = 0; index < size; index++) {
        result[index] = index;
    }
    return result;
}

// Constructor from shape array
Shuffle::Shuffle(const intvec & shape) : shuffled_index_(shape.size()) {
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        this->shuffled_index_[i_dim] = initialize_index_vector(shape[i_dim]);
        std::shuffle(this->shuffled_index_[i_dim].begin(), this->shuffled_index_[i_dim].end(), this->random_generator_);
    }
}

// Set random seed
void Shuffle::set_random_seed(std::uint64_t seed) { Shuffle::random_generator_.seed(seed); }

// Get shuffled index
intvec Shuffle::operator[](const intvec original_index) const noexcept {
    intvec result(this->ndim());
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        result[i_dim] = this->shuffled_index_[i_dim][original_index[i_dim]];
    }
    return result;
}

// inverse permutation
Shuffle Shuffle::inverse(void) const {
    Shuffle result(*this);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        intvec & inverse_result = result.shuffled_index_[i_dim];
        const intvec src = this->shuffled_index_[i_dim];
        for (std::uint64_t index = 0; index < src.size(); index++) {
            inverse_result[src[index]] = index;
        }
    }
    return result;
}

// String representation
std::string Shuffle::str(void) const {
    std::ostringstream output;
    output << "<Shuffle(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (i_dim != 0) {
            output << " ";
        }
        output << this->shuffled_index_[i_dim].str();
    }
    output << ")>";
    return output.str();
}

}  // namespace merlin
