// Copyright 2024 quocdang1998
#include "merlin/candy/trial_policy.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// TrialPolicy
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from attributes
candy::TrialPolicy::TrialPolicy(std::uint64_t discarded, std::uint64_t strict, std::uint64_t loose) :
discarded_(discarded), strict_(strict), loose_(loose) {
    if (discarded < 1) {
        Fatal<std::invalid_argument>("Discarded must be greater than 1 (always neglect the initial error).");
    }
    this->sum_ = discarded + strict + loose;
}

// String representation
std::string candy::TrialPolicy::str(void) const {
    std::ostringstream os;
    os << "<TrialPolicy(";
    os << "discarded=" << this->discarded_ << ", ";
    os << "strict=" << this->strict_ << ", ";
    os << "loose=" << this->loose_;
    os << ")>";
    return os.str();
}

}  // namespace merlin
