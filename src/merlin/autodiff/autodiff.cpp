// Copyright 2022 quocdang1998
#include "merlin/autodiff.hpp"

namespace merlin {

Constant::Constant(float value) {
    this->data_ = Tensor(value);
}

}  // namespace merlin
