#include "merlin/utils.hpp"

namespace merlin {

#if defined(__LIBMERLINCUDA_STATIC__) || defined(__MERLIN_FORCE_STATIC__)

// -------------------------------------------------------------------------------------------------------------------------
// Miscellaneous utils
// -------------------------------------------------------------------------------------------------------------------------

// Inner product
__cuhostdev__ unsigned long int inner_prod(const intvec & v1, const intvec & v2) {
    // check size of 2 vectors
    #ifndef __CUDA_ARCH__
    if (v1.size() != v2.size()) {
        FAILURE(std::invalid_argument, "Size of v1 (%d) and size of v2 (%d) are not equal.\n", v1.size(), v2.size());
    }
    #endif  // __CUDA_ARCH__
    // calculate inner product
    unsigned long int inner_product = 0;
    for (int i = 0; i < v1.size(); i++) {
        inner_product += v1[i] * v2[i];
    }
    return inner_product;
}

// -------------------------------------------------------------------------------------------------------------------------
// NdData tools
// -------------------------------------------------------------------------------------------------------------------------

// Convert n-dimensional index to C-contiguous index
__cuhostdev__ unsigned long int ndim_to_contiguous_idx(const intvec & index, const intvec & shape) {
    return inner_prod(index, shape);
}

// Convert C-contiguous index to n-dimensional index
__cuhostdev__ intvec contiguous_to_ndim_idx(unsigned long int index, const intvec & shape) {
    // calculate index vector
    intvec index_(shape.size());
    unsigned long int cum_prod;
    for (int i = shape.size()-1; i >= 0; i--) {
        if (i == shape.size()-1) {
            cum_prod = 1;
        } else {
            cum_prod = cum_prod * shape[i+1];
        }
        index_[i] = (index / cum_prod) % shape[i];
    }
    return index_;
}

#endif  // __LIBMERLINCUDA_STATIC__ || __MERLIN_FORCE_STATIC__

} // namespace merlin
