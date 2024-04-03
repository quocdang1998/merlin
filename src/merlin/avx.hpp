#ifndef MERLIN_AVX_HPP_
#define MERLIN_AVX_HPP_

#include <array>    // std::array
#include <cmath>    // std::fma
#include <cstdint>  // std::uint64_t

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
template <AvxFlag UseAvx>
struct PackedDouble {};

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
    inline PackedDouble(const double * data) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = data[i];
        }
    }
    /** @brief Constructor from pointer to data and number of elements to copy.
     *  @details Copy values from a pointer. Pointer is not necessarily aligned on 32-bits boundary.
     *  @note Out-of-bound elements will be zeros.
     *  @param data Pointer to source.
     *  @param n Number of elements to copy, should be less than or equal to 4.
     */
    inline PackedDouble(const double * data, std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; i++) {
            this->core[i] = data[i];
        }
    }
    /** @brief Constructor from value.
     *  @details Store 4 copies of the values provided.
     */
    inline PackedDouble(double value) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = value;
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
    /** @brief Division.
     *  @details Perform element-wise division of 2 vectors and store the result to the current object.
     */
    inline void divide(const PackedDouble<AvxFlag::NoAvx> & a) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] /= a.core[i];
        }
    }
    /// @}

    /// @name Store
    /// @{
    /** @brief Write values back to memory.*/
    inline void store(double * dest) {
        for (std::uint64_t i = 0; i < 4; i++) {
            dest[i] = this->core[i];
        }
    }
    /** @brief Write some values back to memory.
     *  @details Copy some elements from register back to memory.
     */
    inline void store(double * dest, std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; i++) {
            dest[i] = this->core[i];
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
    inline PackedDouble(void) { this->core = ::_mm256_set1_pd(0.0); }
    /** @brief Constructor from pointer to data.
     *  @details Pointer is not necessarily aligned on 32-bits boundary.
     */
    inline PackedDouble(const double * data) { this->core = ::_mm256_loadu_pd(data); }
    /** @brief Constructor from pointer to data and number of elements to copy.
     *  @details Copy values from a pointer. Pointer is not necessarily aligned on 32-bits boundary.
     *  @note Out-of-bound elements will be zeros.
     *  @param data Pointer to source.
     *  @param n Number of elements to copy, should be less than or equal to 4.
     */
    inline PackedDouble(const double * data, std::uint64_t n) {
        static const ::__m256i masks[5] = {
            ::_mm256_set_epi64x(0, 0, 0, 0),
            ::_mm256_set_epi64x(-1, 0, 0, 0),
            ::_mm256_set_epi64x(-1, -1, 0, 0),
            ::_mm256_set_epi64x(-1, -1, -1, 0),
            ::_mm256_set_epi64x(-1, -1, -1, -1)
        };
        this->core = ::_mm256_maskload_pd(data, masks[n]);
    }
    /** @brief Constructor from value.
     *  @details Store 4 copies of the values provided.
     */
    inline PackedDouble(double value) { this->core = ::_mm256_set1_pd(value); }
    /// @}

    /// @name Access elements
    /// @{
    /** @brief Get reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    inline double & operator[](std::uint64_t index) { return *(reinterpret_cast<double *>(&(this->core)) + index); }
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
    /** @brief Division.
     *  @details Perform element-wise division of 2 vectors and store the result to the current object.
     */
    inline void divide(const PackedDouble<AvxFlag::AvxOn> & a) {
        this->core = ::_mm256_div_pd(this->core, a.core);
    }
    /// @}

    /// @name Store
    /// @{
    /** @brief Write values back to memory.*/
    inline void store(double * dest) { ::_mm256_storeu_pd(dest, this->core); }
    /** @brief Write some values back to memory.
     *  @details Copy some elements from register back to memory.
     */
    inline void store(double * dest, std::uint64_t n) {
        static const ::__m256i masks[5] = {
            ::_mm256_set_epi64x(0, 0, 0, 0),
            ::_mm256_set_epi64x(-1, 0, 0, 0),
            ::_mm256_set_epi64x(-1, -1, 0, 0),
            ::_mm256_set_epi64x(-1, -1, -1, 0),
            ::_mm256_set_epi64x(-1, -1, -1, -1)
        };
        ::_mm256_maskstore_pd(dest, masks[n], this->core);
    }
    /// @}

    /** @brief Core object.*/
    ::__m256d core;
};

#endif

}  // namespace merlin

#endif  // MERLIN_AVX_HPP_
