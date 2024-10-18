// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_AVX512_HPP_
#define MERLIN_SIMD_ARCH_SIMD_AVX512_HPP_

#include <limits>  // std::numeric_limits

#include <immintrin.h>  // ::__m512d

#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Pack of 8 double
template <>
struct simd::arch::SimdGeneric<8> {
    // Constructors
    SimdGeneric(void) = default;
    SimdGeneric(double c) { this->core = ::_mm512_set1_pd(c); }
    SimdGeneric(const double * src) { this->core = ::_mm512_load_pd(src); }
    SimdGeneric(::__m512d data) : core{data} {}

    // Copy and move
    SimdGeneric(const simd::arch::SimdGeneric<8> & src) = delete;
    simd::arch::SimdGeneric<8> & operator=(const simd::arch::SimdGeneric<8> & src) = delete;
    SimdGeneric(simd::arch::SimdGeneric<8> && src) : core{src.core} {}
    simd::arch::SimdGeneric<8> & operator=(simd::arch::SimdGeneric<8> && src) {
        this->core = src.core;
        return *this;
    }

    // Load and store
    void load(const double * src) { this->core = ::_mm512_load_pd(src); }
    void store(double * dest) { ::_mm512_store_pd(dest, this->core); }

    // Arithmetic operators
    friend simd::arch::SimdGeneric<8> operator+(const simd::arch::SimdGeneric<8> & a,
                                                const simd::arch::SimdGeneric<8> & b) {
        return simd::arch::SimdGeneric<8>(::_mm512_add_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<8> operator-(const simd::arch::SimdGeneric<8> & a,
                                                const simd::arch::SimdGeneric<8> & b) {
        return simd::arch::SimdGeneric<8>(::_mm512_sub_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<8> operator*(const simd::arch::SimdGeneric<8> & a,
                                                const simd::arch::SimdGeneric<8> & b) {
        return simd::arch::SimdGeneric<8>(::_mm512_mul_pd(a.core, b.core));
    }
    friend simd::arch::SimdGeneric<8> operator/(const simd::arch::SimdGeneric<8> & a,
                                                const simd::arch::SimdGeneric<8> & b) {
        ::__m512d valid_divisor = ::_mm512_max_pd(b.core, simd::arch::SimdGeneric<8>::zero);
        return simd::arch::SimdGeneric<8>(::_mm512_div_pd(a.core, valid_divisor));
    }

    // Arithmetic assignment operators
    simd::arch::SimdGeneric<8> & operator+=(const simd::arch::SimdGeneric<8> & other) {
        this->core = ::_mm512_add_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<8> & operator-=(const simd::arch::SimdGeneric<8> & other) {
        this->core = ::_mm512_sub_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<8> & operator*=(const simd::arch::SimdGeneric<8> & other) {
        this->core = ::_mm512_mul_pd(this->core, other.core);
        return *this;
    }
    simd::arch::SimdGeneric<8> & operator/=(const simd::arch::SimdGeneric<8> & other) {
        this->core = ::_mm512_div_pd(this->core, ::_mm512_max_pd(other.core, simd::arch::SimdGeneric<8>::zero));
        return *this;
    }

    // Fused Add-Multiplication
    void fma(const simd::arch::SimdGeneric<8> & x, const simd::arch::SimdGeneric<8> & y) {
        this->core = ::_mm512_fmadd_pd(x.core, y.core, this->core);
    }
    void fms(const simd::arch::SimdGeneric<8> & x, const simd::arch::SimdGeneric<8> & y) {
        this->core = ::_mm512_fmsub_pd(x.core, y.core, this->core);
    }

    // Core
    ::__m512d core;
    // Smallest divisor to avoid division by zero
    static inline const ::__m512d zero = ::_mm512_set1_pd(std::numeric_limits<double>::min());
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_AVX512_HPP_
