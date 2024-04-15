// Copyright 2023 quocdang1998
#include "merlin/linalg/matrix.hpp"

#include <new>      // std::align_val_t
#include <sstream>  // std::ostringstream

#include "merlin/logger.hpp"  // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------------------------------------------------

// Create an empty matrix from shape
linalg::Matrix::Matrix(std::uint64_t nrow, std::uint64_t ncol) : shape_({nrow, ncol}) {
    if ((nrow == 0) || (ncol == 0)) {
        Fatal<std::invalid_argument>("Number of rows or number of columns equals 0.\n");
    }
    this->ld_ = nrow;
    this->data_ = static_cast<double *>(::operator new[](sizeof(double) * nrow * ncol, std::align_val_t(32)));
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
        ::operator delete[](this->data_, sizeof(double) * this->shape_[0] * this->shape_[1], std::align_val_t(32));
    }
}

}  // namespace merlin
