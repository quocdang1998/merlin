// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_SSE2_HPP_
#define MERLIN_SIMD_ARCH_SIMD_SSE2_HPP_

#include <limits>  // std::numeric_limits

#include <emmintrin.h>  // ::__m128d

#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Pack of 2 double
template <>
struct simd::arch::SimdGeneric<2> {
    // Constructors
    SimdGeneric(void) = default;
    SimdGeneric(double c) { this->core = ::_mm_set1_pd(c); }
    SimdGeneric(const double * src) { this->core = ::_mm_load_pd(src); }
    SimdGeneric(::__m128d data) : core{data} {}

    // Copy and move
    SimdGeneric(const simd::arch::SimdGeneric<2> & src) = delete;
    simd::arch::SimdGeneric<2> & operator=(const simd::arch::SimdGeneric<2> & src) = delete;
    SimdGeneric(simd::arch::SimdGeneric<2> && src) : core{src.core} {}
    simd::arch::SimdGeneric<2> & operator=(simd::arch::SimdGeneric<2> && src) {
        this->core = src.core;
        return *this;
    }

    // Load and store
    void load(const double * src) { this->core = ::_mm_load_pd(src); }
    void store(double * dest) { ::_mm_store_pd(dest, this->core); }

    // Arithmetic operators
    friend simd::arch::SimdGeneric<2> operator+(const simd::arch::SimdGeneric<2> & a,
                                                const simd::arch::SimdGeneric<2> & b) {
        return simd::arch::SimdGeneric<2>(::_mm_add_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<2> operator-(const simd::arch::SimdGeneric<2> & a,
                                                const simd::arch::SimdGeneric<2> & b) {
        return simd::arch::SimdGeneric<2>(::_mm_sub_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<2> operator*(const simd::arch::SimdGeneric<2> & a,
                                                const simd::arch::SimdGeneric<2> & b) {
        return simd::arch::SimdGeneric<2>(::_mm_mul_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<2> operator/(const simd::arch::SimdGeneric<2> & a,
                                                const simd::arch::SimdGeneric<2> & b) {
        ::__m128d valid_divisor = ::_mm_max_pd(b.core, simd::arch::SimdGeneric<2>::zero);
        return simd::arch::SimdGeneric<2>(::_mm_div_pd(a.core, valid_divisor));
    }

    // Arithmetic assignment operators
    simd::arch::SimdGeneric<2> & operator+=(const simd::arch::SimdGeneric<2> & other) {
        this->core = ::_mm_add_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<2> & operator-=(const simd::arch::SimdGeneric<2> & other) {
        this->core = ::_mm_sub_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<2> & operator*=(const simd::arch::SimdGeneric<2> & other) {
        this->core = ::_mm_mul_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<2> & operator/=(const simd::arch::SimdGeneric<2> & other) {
        this->core = ::_mm_div_pd(this->core, ::_mm_max_pd(other.core, simd::arch::SimdGeneric<2>::zero));
        return *this;
    }

    // Fused Add-Multiplication
    void fma(const simd::arch::SimdGeneric<2> & x, const simd::arch::SimdGeneric<2> & y) {
#if defined(__FMA__) || defined(__MERLIN_FMA_PATCH__)
        this->core = ::_mm_fmadd_pd(x.core, y.core, this->core);
#else
        this->core = ::_mm_add_pd(::_mm_mul_pd(x.core, y.core), this->core);
#endif  // __FMA__  || __MERLIN_FMA_PATCH__
    }
    void fms(const simd::arch::SimdGeneric<2> & x, const simd::arch::SimdGeneric<2> & y) {
#if defined(__FMA__) || defined(__MERLIN_FMA_PATCH__)
        this->core = ::_mm_fmsub_pd(x.core, y.core, this->core);
#else
        this->core = ::_mm_sub_pd(::_mm_mul_pd(x.core, y.core), this->core);
#endif  // __FMA__  || __MERLIN_FMA_PATCH__
    }

    // Core
    ::__m128d core;
    // Smallest divisor to avoid division by zero
    static inline const ::__m128d zero = ::_mm_set1_pd(std::numeric_limits<double>::min());
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_SSE2_HPP_
