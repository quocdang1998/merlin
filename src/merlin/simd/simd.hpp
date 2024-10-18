// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_SIMD_HPP_
#define MERLIN_SIMD_SIMD_HPP_

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint64_t

#include "merlin/platform.hpp"                // __MERLIN_WINDOWS__
#include "merlin/simd/arch/simd_generic.hpp"  // merlin::simd::arch::SimdGeneric

namespace merlin::simd {

// SIMD size (for double precision)
// --------------------------------

#if defined(__AVX512F__)
inline constexpr std::uint64_t size = 8;
#elif defined(__AVX__)
inline constexpr std::uint64_t size = 4;
#elif defined(__SSE2__) || defined(__aarch64__)
inline constexpr std::uint64_t size = 2;
#elif defined(__DOXYGEN_PARSER__)
/** @brief Number of double-precision elements inside a pack of SIMD intrinsic data.*/
extern std::uint64_t size;
#else
inline constexpr std::uint64_t size = 1;
#endif

/** @brief Required alignment of the data to enhance SIMD load and store instructions.*/
inline constexpr std::size_t alignment = sizeof(double) * size;

// SIMD
// ----

#if defined(__AVX512F__)
inline constexpr std::uint64_t type = 8;
#elif defined(__AVX__)
inline constexpr std::uint64_t type = 4;
#elif defined(__aarch64__)
inline constexpr std::uint64_t type = 3;
#elif defined(__SSE2__)
inline constexpr std::uint64_t type = 2;
#else
inline constexpr std::uint64_t type = 1;
#endif

/** @brief SIMD class (greatest SIMD) */
using Simd = simd::arch::SimdGeneric<simd::type>;

}  // namespace merlin::simd

// Define FMA macro on Windows
// ---------------------------
#if defined(__MERLIN_WINDOWS__) && defined(__AVX2__)
    #define __MERLIN_FMA_PATCH__
#endif

// Include source according to specific architecture
// -------------------------------------------------

#if defined(__AVX512F__)
#include "merlin/simd/arch/simd_avx512.hpp"
#elif defined(__AVX__)
#include "merlin/simd/arch/simd_avx.hpp"
#elif defined(__aarch64__)
#include "merlin/simd/arch/simd_neon64.hpp"
#elif defined(__SSE2__)
#include "merlin/simd/arch/simd_sse2.hpp"
#else
#include "merlin/simd/arch/simd_none.hpp"
#endif

// Undefine FMA macro on Windows
// -----------------------------

#if defined(__MERLIN_WINDOWS__) && defined(__AVX2__)
    #undef __MERLIN_FMA_PATCH__
#endif

#endif  // MERLIN_SIMD_SIMD_HPP_
