// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLANT_HPP_
#define MERLIN_SPLINT_INTERPOLANT_HPP_

#include "merlin/array/nddata.hpp"  // merlin::array::NdData

namespace merlin {

class splint::Interpolant {
  public:
    /** @brief Default constructor.*/
    Interpolant(void) = default;

  private:
    /** @brief Pointer to coefficient array.*/
    array::NdData * coefficient = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_INTERPOLANT_HPP_
