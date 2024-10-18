// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_AVX_HPP_
#define MERLIN_SIMD_ARCH_SIMD_AVX_HPP_

#include <limits>  // std::numeric_limits

#include <immintrin.h>  // ::__m256d

#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Pack of 4 double
template <>
struct simd::arch::SimdGeneric<4> {
    // Constructors
    SimdGeneric(void) = default;
    SimdGeneric(double c) { this->core = ::_mm256_set1_pd(c); }
    SimdGeneric(const double * src) { this->core = ::_mm256_load_pd(src); }
    SimdGeneric(::__m256d data) : core{data} {}

    // Copy and move
    SimdGeneric(const simd::arch::SimdGeneric<4> & src) = delete;
    simd::arch::SimdGeneric<4> & operator=(const simd::arch::SimdGeneric<4> & src) = delete;
    SimdGeneric(simd::arch::SimdGeneric<4> && src) : core{src.core} {}
    simd::arch::SimdGeneric<4> & operator=(simd::arch::SimdGeneric<4> && src) {
        this->core = src.core;
        return *this;
    }

    // Load and store
    void load(const double * src) { this->core = ::_mm256_load_pd(src); }
    void store(double * dest) { ::_mm256_store_pd(dest, this->core); }

    // Arithmetic operators
    friend simd::arch::SimdGeneric<4> operator+(const simd::arch::SimdGeneric<4> & a,
                                                const simd::arch::SimdGeneric<4> & b) {
        return simd::arch::SimdGeneric<4>(::_mm256_add_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<4> operator-(const simd::arch::SimdGeneric<4> & a,
                                                const simd::arch::SimdGeneric<4> & b) {
        return simd::arch::SimdGeneric<4>(::_mm256_sub_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<4> operator*(const simd::arch::SimdGeneric<4> & a,
                                                const simd::arch::SimdGeneric<4> & b) {
        return simd::arch::SimdGeneric<4>(::_mm256_mul_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<4> operator/(const simd::arch::SimdGeneric<4> & a,
                                                const simd::arch::SimdGeneric<4> & b) {
        ::__m256d valid_divisor = ::_mm256_max_pd(b.core, simd::arch::SimdGeneric<4>::zero);
        return simd::arch::SimdGeneric<4>(::_mm256_div_pd(a.core, valid_divisor));
    }

    // Arithmetic assignment operators
    simd::arch::SimdGeneric<4> & operator+=(const simd::arch::SimdGeneric<4> & other) {
        this->core = ::_mm256_add_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<4> & operator-=(const simd::arch::SimdGeneric<4> & other) {
        this->core = ::_mm256_sub_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<4> & operator*=(const simd::arch::SimdGeneric<4> & other) {
        this->core = ::_mm256_mul_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<4> & operator/=(const simd::arch::SimdGeneric<4> & other) {
        this->core = ::_mm256_div_pd(this->core, ::_mm256_max_pd(other.core, simd::arch::SimdGeneric<4>::zero));
        return *this;
    }

    // Fused Add-Multiplication
    void fma(const simd::arch::SimdGeneric<4> & x, const simd::arch::SimdGeneric<4> & y) {
#if defined(__FMA__) || defined(__MERLIN_FMA_PATCH__)
        this->core = ::_mm256_fmadd_pd(x.core, y.core, this->core);
#else
        this->core = ::_mm256_add_pd(::_mm256_mul_pd(x.core, y.core), this->core);
#endif  // __FMA__  || __MERLIN_FMA_PATCH__
    }
    void fms(const simd::arch::SimdGeneric<4> & x, const simd::arch::SimdGeneric<4> & y) {
#if defined(__FMA__) || defined(__MERLIN_FMA_PATCH__)
        this->core = ::_mm256_fmsub_pd(x.core, y.core, this->core);
#else
        this->core = ::_mm256_sub_pd(::_mm256_mul_pd(x.core, y.core), this->core);
#endif  // __FMA__  || __MERLIN_FMA_PATCH__
    }

    // Core
    ::__m256d core;
    // Smallest divisor to avoid division by zero
    static inline const ::__m256d zero = ::_mm256_set1_pd(std::numeric_limits<double>::min());
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_AVX_HPP_
