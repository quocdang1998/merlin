// Copyright 2022 quocdang1998
#include "merlin/slice.hpp"

#include <sstream>  // std::ostringstream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------------------------------------------------

std::string Slice::str(void) const {
    std::ostringstream os;
    os << "<Slice object: " << this->start_ << ", " << this->stop_ << ", " << this->step_ << ">";
    return os.str();
}

}  // namespace merlin
