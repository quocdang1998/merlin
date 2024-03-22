#ifndef MERLIN_AVX_HPP_
#define MERLIN_AVX_HPP_

#include <array>  // std::array
#include <cmath>  // std::fma
#include <cstdint>  // std::uint64_t

#include <cstdio>

#ifdef __AVX__
    #include <immintrin.h>
#endif  // __AVX__

namespace merlin {

/** @brief AVX use flags.*/
enum class AvxFlag {
    /** @brief Compilation without AVX optimization.*/
    NoAvx,
    /** @brief Compilation using AVX and FMA optimization.*/
    AvxOn,
    /** @brief Compilation using AVX512 optimization.*/
    Avx512
};

#ifdef __AVX__
inline constexpr AvxFlag use_avx = AvxFlag::AvxOn;
#else
inline constexpr AvxFlag use_avx = AvxFlag::NoAvx;
#endif  // __AVX__

/** @brief Class representing a packed-4 double.
 *  @tparam UseAvx Use AVX for arithmetics operations.
 */
template<AvxFlag UseAvx>
struct PackedDouble { };

/** @brief Pack of four double precision for manual vectorization.*/
template <>
struct PackedDouble<AvxFlag::NoAvx> {
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    PackedDouble(void) = default;
    /** @brief Constructor from pointer to data.
     *  @details Copy 4 values from a pointer. Pointer is not necessarily aligned on 32-bits boundary.
     *  @note With highest optimization level, the copy will be deprecated.
     */
    inline PackedDouble(double * data) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = data[i];
        }
    }
    /// @}

    /// @name Access elements
    /// @{
    /** @brief Get reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    inline double & operator[](std::uint64_t index) { return this->core[index]; }
    /** @brief Get constant reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    inline const double & operator[](std::uint64_t index) const { return this->core[index]; }
    /// @}

    /// @name Arithmetic operations
    /// @{
    /** @brief Fused add multiplication.
     *  @details Perform element-wise multiplication of 2 vectors and add the result to the current object.
     */
    inline void fma(const PackedDouble<AvxFlag::NoAvx> & a, const PackedDouble<AvxFlag::NoAvx> & b) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = std::fma(a.core[i], b.core[i], this->core[i]);
        }
    }
    /// @}

    /** @brief Core object.*/
    std::array<double, 4> core;
};

#ifdef __AVX__

/** @brief Pack of four double precision with AVX vectorization.*/
template <>
struct PackedDouble<AvxFlag::AvxOn> {
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    inline PackedDouble(void) {
        this->core = ::_mm256_set1_pd(0.0);
    }
    /** @brief Constructor from pointer to data.
     *  @details Pointer is not necessarily aligned on 32-bits boundary. 
     */
    inline PackedDouble(double * data) {
        this->core = ::_mm256_loadu_pd(data);
    }
    /// @}

    /// @name Access elements
    /// @{
    /** @brief Get reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    inline double & operator[](std::uint64_t index) {
        return *(reinterpret_cast<double *>(&(this->core)) + index);
    }
    /** @brief Get constant reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    inline const double & operator[](std::uint64_t index) const {
        return *(reinterpret_cast<const double *>(&(this->core)) + index);
    }
    /// @}

    /// @name Arithmetic operations
    /// @{
    /** @brief Fused add multiplication.
     *  @details Perform element-wise multiplication of 2 vectors and add the result to the current object.
     */
    inline void fma(const PackedDouble<AvxFlag::AvxOn> & a, const PackedDouble<AvxFlag::AvxOn> & b) {
        this->core = ::_mm256_fmadd_pd(a.core, b.core, this->core);
    }
    /// @}

    /** @brief Core object.*/
    ::__m256d core;
};

#endif

}  // namespace merlin

#endif  // MERLIN_AVX_HPP_
