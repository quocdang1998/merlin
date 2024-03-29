// Copyright 2023 quocdang1998
#include "merlin/linalg/matrix.hpp"

#include <sstream>  // std::ostringstream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------------------------------------------------

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

}  // namespace merlin
