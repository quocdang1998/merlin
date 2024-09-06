// Copyright 2023 quocdang1998
#include "merlin/candy/gradient.hpp"

#include <array>    // std::array
#include <sstream>  // std::ostringstream

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
void candy::Gradient::calc_by_cpu(candy::Model & model, const array::Array & train_data, Index & index_mem) noexcept {
    candy::calc_gradient(model, train_data, this->value_, static_cast<unsigned int>(this->train_metric_), 0, 1,
                         index_mem);
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
