// Copyright 2024 quocdang1998
#ifndef MERLIN_LINALG_QRP_DECOMP_HPP_
#define MERLIN_LINALG_QRP_DECOMP_HPP_

#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix, merlin::linalg::QRPDecomp
#include "merlin/permutation.hpp"  // merlin::Permutation
#include "merlin/vector.hpp"  // merlin::DoubleVec

namespace merlin {

/** @brief QR decomposition with column pivot.*/
class linalg::QRPDecomp {
  public:


  protected:
    /** @brief Core matrix.*/
    linalg::Matrix core_;
    /** @brief Diagonal elements.*/
    DoubleVec diag_;
    /** @brief Permutation matrix.*/
    Permutation permut_;
};

}  // namespace merlin

#endif  // MERLIN_LINALG_QRP_DECOMP_HPP_
