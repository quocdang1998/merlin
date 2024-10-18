// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ARCH_SIMD_GENERIC_HPP_
#define MERLIN_SIMD_ARCH_SIMD_GENERIC_HPP_

#include "merlin/config.hpp"                 // __cuhostdev__
#include "merlin/simd/arch/declaration.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin {

// Packed vector
// -------------

/** @brief Packed double.
 *  @details Wrapper for SSE-AVX types.
 */
template <std::uint64_t Type>
struct simd::arch::SimdGeneric {
    // Constructors
    __cuhostdev__ SimdGeneric(void);
    __cuhostdev__ SimdGeneric(double c);
    __cuhostdev__ SimdGeneric(const double * src);

    // Copy and move
    __cuhostdev__ SimdGeneric(const simd::arch::SimdGeneric<Type> & src);
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator=(const simd::arch::SimdGeneric<Type> & src);
    __cuhostdev__ SimdGeneric(simd::arch::SimdGeneric<Type> && src);
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator=(simd::arch::SimdGeneric<Type> && src);

    // Load and store
    __cuhostdev__ void load(const double * src);
    __cuhostdev__ void store(double * dest);

    // Arithmetic operators
    __cuhostdev__ friend simd::arch::SimdGeneric<Type> operator+(const simd::arch::SimdGeneric<Type> & a,
                                                                 const simd::arch::SimdGeneric<Type> & b);
    __cuhostdev__ friend simd::arch::SimdGeneric<Type> operator-(const simd::arch::SimdGeneric<Type> & a,
                                                                 const simd::arch::SimdGeneric<Type> & b);
    __cuhostdev__ friend simd::arch::SimdGeneric<Type> operator*(const simd::arch::SimdGeneric<Type> & a,
                                                                 const simd::arch::SimdGeneric<Type> & b);
    __cuhostdev__ friend simd::arch::SimdGeneric<Type> operator/(const simd::arch::SimdGeneric<Type> & a,
                                                                 const simd::arch::SimdGeneric<Type> & b);

    // Arithmetic assignment operators
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator+=(const simd::arch::SimdGeneric<Type> & other);
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator-=(const simd::arch::SimdGeneric<Type> & other);
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator*=(const simd::arch::SimdGeneric<Type> & other);
    __cuhostdev__ simd::arch::SimdGeneric<Type> & operator/=(const simd::arch::SimdGeneric<Type> & other);

    // Fused Add-Multiplication
    __cuhostdev__ void fma(const simd::arch::SimdGeneric<Type> & x, const simd::arch::SimdGeneric<Type> & y);
    __cuhostdev__ void fms(const simd::arch::SimdGeneric<Type> & x, const simd::arch::SimdGeneric<Type> & y);
};

}  // namespace merlin

#endif  // MERLIN_SIMD_ARCH_SIMD_GENERIC_HPP_
