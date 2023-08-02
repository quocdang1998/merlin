// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_INNER_PROD_TPP_
#define MERLIN_LINALG_INNER_PROD_TPP_

#include <cmath>  // std::sqrt

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Inner product between 2 vectors
// ---------------------------------------------------------------------------------------------------------------------

// Inner product between 2 vectors
template <typename T>
__cuhostdev__ T linalg::inner_product(const Vector<T> & a, const Vector<T> & b) noexcept {
    T result = T();
    for (std::uint64_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Calculate norm of a vector
template <typename T>
__cuhostdev__ T linalg::norm(const Vector<T> & a) noexcept {
    return std::sqrt(linalg::inner_product(a, a));
}

// Normalize a vector
template <typename T>
__cuhostdev__ void linalg::normalize(Vector<T> & a) noexcept {
    T norm = linalg::norm(a);
    for (std::uint64_t i = 0; i < a.size(); i++) {
        a[i] /= norm;
    }
}

}  // namespace merlin

#endif  // MERLIN_LINALG_INNER_PROD_TPP_
