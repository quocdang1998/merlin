// Copyright 2023 quocdang1998
#include "merlin/permutation.hpp"

#include <algorithm>  // std::fill_n, std::shuffle
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream
#include <vector>   // std::vector

#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Constructor of a random permutation given its range
Permutation::Permutation(std::uint64_t range) : index_(range) {
    std::iota(this->index_.begin(), this->index_.end(), 0);
    std::shuffle(this->index_.begin(), this->index_.end(), Environment::random_generator);
}

// Constructor from permutation index
Permutation::Permutation(const UIntVec & index) : index_(index.size()) {
    // check if it is a valid permutation
    std::vector<bool> tracker(index.size());
    std::fill_n(tracker.begin(), tracker.size(), false);
    for (const std::uint64_t i : index) {
        if (i >= index.size()) {
            FAILURE(std::invalid_argument, "Index out of range.\n");
        }
        if (tracker[i]) {
            FAILURE(std::invalid_argument, "Duplicate element.\n");
        }
        tracker[i] = true;
    }
    // copy permutation index
    for (std::uint64_t i = 0; i < index.size(); i++) {
        this->index_[i] = index[i];
    }
}

// String representation
std::string Permutation::str(void) const {
    std::ostringstream os;
    os << "<Permutation(";
    for (std::uint64_t i = 0; i < this->size(); i++) {
        os << ((i > 0) ? " " : "");
        os << i;
    }
    os << ")>";
    return os.str();
}

}  // namespace merlin
