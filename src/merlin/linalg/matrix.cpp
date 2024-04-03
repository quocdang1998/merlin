// Copyright 2023 quocdang1998
#include "merlin/linalg/matrix.hpp"

#include <new>      // std::align_val_t
#include <sstream>  // std::ostringstream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------------------------------------------------

// Create an empty matrix from shape
linalg::Matrix::Matrix(const std::array<std::uint64_t, 2> & shape) : shape_(shape) {
    this->ld_ = shape[0];
    this->data_ = new (std::align_val_t(32)) double[shape[0] * shape[1]];
}

// String representation
std::string linalg::Matrix::str(void) const {
    std::ostringstream os;
    os << "<Matrix(";
    for (std::uint64_t i_row = 0; i_row < this->nrow(); i_row++) {
        if (i_row != 0) {
            os << ' ';
        }
        os << '<';
        for (std::uint64_t i_col = 0; i_col < this->ncol(); i_col++) {
            if (i_col != 0) {
                os << ' ';
            }
            os << this->cget(i_row, i_col);
        }
        os << '>';
    }
    os << ")>";
    return os.str();
}

// Default destructor
linalg::Matrix::~Matrix(void) {
    if (this->data_ != nullptr) {
        delete[] this->data_;
    }
}

}  // namespace merlin
