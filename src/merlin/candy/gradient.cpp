// Copyright 2023 quocdang1998
#include "merlin/candy/gradient.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------------------------------------------------

// Calculate gradient from data in parallel section
void candy::Gradient::calc(const array::Array & train_data, std::uint64_t thread_idx, std::uint64_t n_threads,
                           intvec & index_mem, floatvec & parallel_mem) noexcept {
    candy::relquare_grad(*(this->model_ptr_), train_data, this->value_, thread_idx, n_threads, index_mem, parallel_mem);
}

// String representation
std::string candy::Gradient::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Gradient(";
    out_stream << this->value_.str();
    out_stream << ")>";
    return out_stream.str();
}

// Destructor
candy::Gradient::~Gradient(void) {}

}  // namespace merlin
