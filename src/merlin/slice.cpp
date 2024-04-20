// Copyright 2022 quocdang1998
#include "merlin/slice.hpp"

#include <cinttypes>  // PRIu64
#include <sstream>    // std::ostringstream

#include "merlin/logger.hpp"  // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------------------------------------------------

// Check validity of input values
void Slice::check_validity(void) const {
    if (this->step_ == 0) {
        Fatal<std::invalid_argument>("Slice step cannot be zero.\n");
    }
    if (this->stop_ < this->start_) {
        Fatal<std::invalid_argument>("Stop index %" PRIu64 " must be greater than start index %" PRIu64 ".\n",
                                     this->stop_, this->start_);
    }
}

// Constructor from initializer list
Slice::Slice(std::initializer_list<std::uint64_t> list) {
    const std::uint64_t * list_data = list.begin();
    switch (list.size()) {
        case 0 : {  // empty = get all element
            break;
        }
        case 1 : {  // 1 element = get 1 element
            this->start_ = list_data[0];
            this->stop_ = list_data[0] + 1;
            break;
        }
        case 2 : {  // 2 element = {start, stop}
            this->start_ = list_data[0];
            this->stop_ = list_data[1];
            this->check_validity();
            break;
        }
        case 3 : {
            this->start_ = list_data[0];
            this->stop_ = list_data[1];
            this->step_ = list_data[2];
            this->check_validity();
            break;
        }
        default : {
            Fatal<std::invalid_argument>("Expected intializer list with size at most 3, got %u.\n",
                                         static_cast<unsigned int>(list.size()));
            break;
        }
    }
}

std::string Slice::str(void) const {
    std::ostringstream os;
    os << "<Slice object: " << this->start_ << ", " << this->stop_ << ", " << this->step_ << ">";
    return os.str();
}

}  // namespace merlin
