// Copyright 2023 quocdang1998
#include "merlin/permutation.hpp"

#include <algorithm>  // std::fill_n
#include <numeric>    // std::iota
#include <sstream>    // std::ostringstream
#include <vector>     // std::vector

#include "merlin/logger.hpp"  // merlin::Fatal

namespace merlin {

// Constructor of an identity permutation given its range
Permutation::Permutation(std::uint64_t range) : index_(range) {
    std::iota(this->index_.begin(), this->index_.end(), 0);
}

// Constructor from permutation index
Permutation::Permutation(const UIntVec & index) : index_(index.size()) {
    // check if it is a valid permutation
    std::vector<bool> tracker(index.size());
    std::fill_n(tracker.begin(), tracker.size(), false);
    for (const std::uint64_t & i : index) {
        if (i >= index.size()) {
            Fatal<std::invalid_argument>("Index out of range.\n");
        }
        if (tracker[i]) {
            Fatal<std::invalid_argument>("Duplicate element.\n");
        }
        tracker[i] = true;
    }
    this->index_ = IntVec(index.begin(), index.end());
}

// Calculate the inverse permutation
Permutation Permutation::inv(void) const {
    Permutation inverse;
    inverse.index_ = IntVec(this->size());
    for (std::uint64_t i = 0; i < this->size(); i++) {
        inverse.index_[this->index_[i]] = i;
    }
    return inverse;
}

// String representation
std::string Permutation::str(void) const {
    std::ostringstream os;
    os << "<Permutation(";
    for (std::uint64_t i = 0; i < this->size(); i++) {
        os << ((i > 0) ? " " : "");
        os << this->index_[i];
    }
    os << ")>";
    return os.str();
}

}  // namespace merlin
