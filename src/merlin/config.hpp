// Copyright 2024 quocdang1998
#ifndef MERLIN_CONFIG_HPP_
#define MERLIN_CONFIG_HPP_

#include <array>             // std::array
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list

namespace merlin {

// Array with known max number of size
// -----------------------------------

/** @brief Max number of dimensions.*/
inline constexpr const std::uint64_t max_dim = 16;

// CUDA interface
// --------------

// CUDA decorator expansion when not compiling with nvcc
#ifdef __NVCC__
    #define __cudevice__ __device__
    #define __cuhostdev__ __host__ __device__
#else
    #define __cudevice__ static_assert(false, "Cannot compile pure device function without CUDA.\n");
    #define __cuhostdev__
#endif

// Indicator when the source is compiled in GPU mode
#ifdef __CUDA_ARCH__
inline constexpr bool device_mode = true;
#else
inline constexpr bool device_mode = false;
#endif

// Advanced vectorization
// ----------------------

/** @brief AVX use flags.*/
enum class AvxFlag {
    /** @brief Compilation without AVX optimization.*/
    NoAvx,
    /** @brief Compilation using AVX and FMA optimization.*/
    AvxOn,
};

#ifdef __AVX__
inline constexpr AvxFlag use_avx = AvxFlag::AvxOn;
#else
inline constexpr AvxFlag use_avx = AvxFlag::NoAvx;
#endif  // __AVX__

#ifdef __DOXYGEN_PARSER__
/** @brief Flag indicate if AVX is enabled during the compilation.*/
static AvxFlag use_avx;
#endif  // __DOXYGEN_PARSER__

// Candy dry-run criterion
// -----------------------

/** @brief Maximum ratio between the RMSE of two consecutive step in strict mode.*/
inline constexpr double strict_max_ratio = 1.0 - 1e-5;

/** @brief Maximum ratio between the RMSE of two consecutive step in loose mode.*/
inline constexpr double loose_max_ratio = 1.0 + 1e-10;

}  // namespace merlin

#endif  // MERLIN_CONFIG_HPP_
