// Copyright 2023 quocdang1998
#include "merlin/candy/gradient.hpp"

#include <array>    // std::array
#include <sstream>  // std::ostringstream

#include "merlin/array/array.hpp"  // merlin::array::Array

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------------------------------------------------

// Calculate gradient from data in CPU parallel section
void candy::Gradient::calc_by_cpu(candy::Model & model, const array::Array & train_data, std::uint64_t thread_idx,
                                  std::uint64_t n_threads, std::uint64_t * cache_mem) noexcept {
    static std::array<candy::GradientCalc, 2> grad_methods = {candy::rlsquare_grad, candy::absquare_grad};
    unsigned int metric = static_cast<unsigned int>(this->train_metric_);
    grad_methods[metric](model, train_data, this->value_, thread_idx, n_threads, cache_mem);
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
