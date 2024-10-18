// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_NONE_HPP_
#define MERLIN_SIMD_ARCH_SIMD_NONE_HPP_

#include <algorithm>  // std::max
#include <cmath>      // std::fma
#include <limits>     // std::numeric_limits

#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Pack of 1 double
template <>
struct simd::arch::SimdGeneric<1> {
    // Constructors
    SimdGeneric(void) = default;
    SimdGeneric(double c) { this->core = c; }
    SimdGeneric(const double * src) { this->core = *src; }
    SimdGeneric(double data) : core{data} {}

    // Copy and move
    SimdGeneric(const simd::arch::SimdGeneric<1> & src) = delete;
    simd::arch::SimdGeneric<1> & operator=(const simd::arch::SimdGeneric<1> & src) = delete;
    SimdGeneric(simd::arch::SimdGeneric<1> && src) : core{src.core} {}
    simd::arch::SimdGeneric<1> & operator=(simd::arch::SimdGeneric<1> && src) {
        this->core = src.core;
        return *this;
    }

    // Load and store
    void load(const double * src) { this->core = *src; }
    void store(double * dest) { *dest = this->core; }

    // Arithmetic operators
    friend simd::arch::SimdGeneric<1> operator+(const simd::arch::SimdGeneric<1> & a,
                                                const simd::arch::SimdGeneric<1> & b) {
        return simd::arch::SimdGeneric<1>(a.core + b.core);
    }
    friend simd::arch::SimdGeneric<1> operator-(const simd::arch::SimdGeneric<1> & a,
                                                const simd::arch::SimdGeneric<1> & b) {
        return simd::arch::SimdGeneric<1>(a.core - b.core);
    }
    friend simd::arch::SimdGeneric<1> operator*(const simd::arch::SimdGeneric<1> & a,
                                                const simd::arch::SimdGeneric<1> & b) {
        return simd::arch::SimdGeneric<1>(a.core * b.core);
    }
    friend simd::arch::SimdGeneric<1> operator/(const simd::arch::SimdGeneric<1> & a,
                                                const simd::arch::SimdGeneric<1> & b) {
        double valid_divisor = std::max(b, simd::arch::SimdGeneric<1>::zero);
        return simd::arch::SimdGeneric<1>(a.core / valid_divisor);
    }

    // Arithmetic assignment operators
    simd::arch::SimdGeneric<1> & operator+=(const simd::arch::SimdGeneric<1> & other) {
        this->core += other.core;
        return *this;
    }
    simd::arch::SimdGeneric<1> & operator-=(const simd::arch::SimdGeneric<1> & other) {
        this->core -= other.core;
        return *this;
    }
    simd::arch::SimdGeneric<1> & operator*=(const simd::arch::SimdGeneric<1> & other) {
        this->core *= other.core;
        return *this;
    }
    simd::arch::SimdGeneric<1> & operator/=(const simd::arch::SimdGeneric<1> & other) {
        this->core /= std::max(other.core, simd::arch::SimdGeneric<1>::zero);
        return *this;
    }

    // Fused Add-Multiplication
    void fma(const simd::arch::SimdGeneric<1> & x, const simd::arch::SimdGeneric<1> & y) {
        this->core = std::fma(x.core, y.core, this->core);
    }
    void fms(const simd::arch::SimdGeneric<1> & x, const simd::arch::SimdGeneric<1> & y) {
        this->core = std::fma(x.core, -1.0 * y.core, this->core);
    }

    // Core
    double core;
    // Smallest divisor to avoid division by zero
    static inline const double zero = std::numeric_limits<double>::min();
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_NONE_HPP_
