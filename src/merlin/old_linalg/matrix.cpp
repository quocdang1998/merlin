// Copyright 2023 quocdang1998
#include "merlin/old_linalg/matrix.hpp"

#include <cstring>  // std::memcpy
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

// Copy constructor
linalg::Matrix::Matrix(const linalg::Matrix & src) : shape_(src.shape_), ld_(src.ld_) {
    std::uint64_t data_size = sizeof(double) * src.shape_[0] * src.shape_[1];
    this->data_ = static_cast<double *>(::operator new[](data_size, std::align_val_t(32)));
    std::memcpy(this->data_, src.data_, data_size);
}

// Copy assignment
linalg::Matrix & linalg::Matrix::operator=(const linalg::Matrix & src) {
    if (this->data_ != nullptr) {
        ::operator delete[](this->data_, sizeof(double) * this->shape_[0] * this->shape_[1], std::align_val_t(32));
    }
    std::uint64_t data_size = sizeof(double) * src.shape_[0] * src.shape_[1];
    this->data_ = static_cast<double *>(::operator new[](data_size, std::align_val_t(32)));
    std::memcpy(this->data_, src.data_, data_size);
    return *this;
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
