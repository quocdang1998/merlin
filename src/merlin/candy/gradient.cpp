// Copyright 2023 quocdang1998
#include "merlin/candy/gradient.hpp"

#include <array>    // std::array
#include <sstream>  // std::ostringstream

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// TrainMetric
// ---------------------------------------------------------------------------------------------------------------------

std::ostream & operator<<(std::ostream & out, const candy::TrainMetric & metric) {
    switch (metric) {
        case candy::TrainMetric::RelativeSquare : {
            out << "RelativeSquare";
            break;
        }
        case candy::TrainMetric::AbsoluteSquare : {
            out << "AbsoluteSquare";
            break;
        }
        default : {
            out << "Undefined";
            break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------------------------------------------------

// Calculate gradient from data in CPU parallel section
void candy::Gradient::calc_by_cpu(candy::Model & model, const array::Array & train_data, std::uint64_t thread_idx,
                                  std::uint64_t n_threads, Index & index_mem) noexcept {
    static std::array<candy::GradientCalc, 2> grad_methods = {candy::rlsquare_grad, candy::absquare_grad};
    unsigned int metric = static_cast<unsigned int>(this->train_metric_);
    grad_methods[metric](model, train_data, this->value_, thread_idx, n_threads, index_mem);
    _Pragma("omp barrier");
}

// String representation
std::string candy::Gradient::str(void) const {
    if (this->value_.data() == nullptr) {
        return std::string("<Invalid Gradient>");
    }
    std::ostringstream out_stream;
    out_stream << "<Gradient(value=<";
    out_stream << this->value_.str();
    out_stream << ">, metric=<";
    out_stream << this->train_metric_;
    out_stream << ">)>";
    return out_stream.str();
}

}  // namespace merlin
