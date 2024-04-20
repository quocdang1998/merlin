#ifndef MERLIN_CONFIG_HPP_
#define MERLIN_CONFIG_HPP_

#include <array>             // std::array
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list

namespace merlin {

// Package version
// ---------------

inline const char * version = "1.0.0";

// Print buffer
// ------------

/** @brief Size of printf buffer (default = 10kB).*/
inline constexpr std::uint64_t printf_buffer = 10240;

// Array with known max number of size
// -----------------------------------

/** @brief Max number of dimensions.*/
inline constexpr const std::uint64_t max_dim = 16;

/** @brief Array of 8 bytes unsigned int.*/
using Index = std::array<std::uint64_t, max_dim>;

/** @brief Array of floatting points.*/
using Point = std::array<double, max_dim>;

/** @brief Array of pointers to floating points.*/
using DPtrArray = std::array<double *, max_dim>;

/** @brief Convertible from pointer.*/
template <class T, class ForwardIterator>
concept ConvertibleFromIterator = requires(ForwardIterator it) {
    { *it } -> std::convertible_to<T>;
};

/** @brief Make an array from another container.*/
template <class T, class ForwardIterator>
requires ConvertibleFromIterator<T, ForwardIterator>
std::array<T, max_dim> make_array(ForwardIterator begin, ForwardIterator end) {
    std::array<T, max_dim> result_array;
    result_array.fill(T());
    typename std::array<T, max_dim>::iterator it_arr = result_array.begin();
    for (ForwardIterator it = begin; it != end; ++it) {
        *(it_arr++) = *(it);
    }
    return result_array;
}

/** @brief Make an array from incomplet initializer list.*/
template <class T>
std::array<T, max_dim> make_array(std::initializer_list<T> list) {
    return make_array<T>(list.begin(), list.end());
}

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

}  // namespace merlin

#endif  // MERLIN_CONFIG_HPP_
