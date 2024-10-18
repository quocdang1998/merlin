// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_NEON64_HPP_
#define MERLIN_SIMD_ARCH_SIMD_NEON64_HPP_

#include <limits>  // std::numeric_limits

#include <arm_neon.h>  // ::float64x2_t

#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Pack of 2 double
template <>
struct simd::arch::SimdGeneric<3> {
    // Constructors
    SimdGeneric(void) = default;
    SimdGeneric(double c) { this->core = ::vdupq_n_f64(c); }
    SimdGeneric(const double * src) { this->core = ::vld1q_f64(src); }
    SimdGeneric(::float64x2_t data) : core{data} {}

    // Copy and move
    SimdGeneric(const simd::arch::SimdGeneric<3> & src) = delete;
    simd::arch::SimdGeneric<3> & operator=(const simd::arch::SimdGeneric<3> & src) = delete;
    SimdGeneric(simd::arch::SimdGeneric<3> && src) : core{src.core} {}
    simd::arch::SimdGeneric<3> & operator=(simd::arch::SimdGeneric<3> && src) {
        this->core = src.core;
        return *this;
    }

    // Load and store
    void load(const double * src) { this->core = ::vld1q_f64(src); }
    void store(double * dest) { ::vst1q_f64(dest, this->core); }

    // Arithmetic operators
    friend simd::arch::SimdGeneric<3> operator+(const simd::arch::SimdGeneric<3> & a,
                                                const simd::arch::SimdGeneric<3> & b) {
        return simd::arch::SimdGeneric<3>(::vaddq_f64(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<3> operator-(const simd::arch::SimdGeneric<3> & a,
                                                const simd::arch::SimdGeneric<3> & b) {
        return simd::arch::SimdGeneric<3>(::vsubq_f64(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<3> operator*(const simd::arch::SimdGeneric<3> & a,
                                                const simd::arch::SimdGeneric<3> & b) {
        return simd::arch::SimdGeneric<3>(::vmulq_f64(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<3> operator/(const simd::arch::SimdGeneric<3> & a,
                                                const simd::arch::SimdGeneric<3> & b) {
        ::float64x2_t valid_divisor = ::vmaxq_f64(b.core, simd::arch::SimdGeneric<3>::zero);
        return simd::arch::SimdGeneric<3>(::vdivq_f64(a.core, valid_divisor));
    }

    // Arithmetic assignment operators
    simd::arch::SimdGeneric<3> & operator+=(const simd::arch::SimdGeneric<3> & other) {
        this->core = ::vaddq_f64(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<3> & operator-=(const simd::arch::SimdGeneric<3> & other) {
        this->core = ::vsubq_f64(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<3> & operator*=(const simd::arch::SimdGeneric<3> & other) {
        this->core = ::vmulq_f64(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<3> & operator/=(const simd::arch::SimdGeneric<3> & other) {
        this->core = ::vdivq_f64(this->core, ::vmaxq_f64(other.core, simd::arch::SimdGeneric<3>::zero));
        return *this;
    }

    // Fused Add-Multiplication
    void fma(const simd::arch::SimdGeneric<3> & x, const simd::arch::SimdGeneric<3> & y) {
        this->core = ::vfmaq_f64(x.core, y.core, this->core);
    }
    void fms(const simd::arch::SimdGeneric<3> & x, const simd::arch::SimdGeneric<3> & y) {
        this->core = ::vfmsq_f64(x.core, y.core, this->core);
    }

    // Core
    ::float64x2_t core;
    // Smallest divisor to avoid division by zero
    static inline const ::float64x2_t zero = ::vdupq_n_f64(std::numeric_limits<double>::min());
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_NEON64_HPP_
