#ifndef MERLIN_AVX_HPP_
#define MERLIN_AVX_HPP_

#include <array>    // std::array
#include <cmath>    // std::fma
#include <cstdint>  // std::uint64_t

#ifdef __AVX__
    #include <immintrin.h>
#endif  // __AVX__

#include "merlin/config.hpp"  // merlin::AvxFlag, merlin::use_avx

namespace merlin {

/** @brief Class representing a packed-4 double.
 *  @tparam UseAvx Use AVX for arithmetics operations.
 */
template <AvxFlag UseAvx>
struct AvxDouble {
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    AvxDouble(void) = default;
    /** @brief Constructor from pointer to data.
     *  @details Copy 4 values from a pointer. Pointer is not necessarily aligned on 32-bits boundary.
     *  @note With highest optimization level, the copy will be deprecated.
     */
    AvxDouble(const double * data);
    /** @brief Constructor from pointer to data and number of elements to copy.
     *  @details Copy values from a pointer. Pointer is not necessarily aligned on 32-bits boundary.
     *  @note Out-of-bound elements will be zeros.
     *  @param data Pointer to source.
     *  @param n Number of elements to copy, should be less than or equal to 4.
     */
    AvxDouble(const double * data, std::uint64_t n);
    /** @brief Constructor from value.
     *  @details Store 4 copies of the values provided.
     */
    AvxDouble(double value);
    /// @}

    /// @name Access elements
    /// @{
    /** @brief Get pointer to the underlying data.*/
    double * data(void);
    /** @brief Get reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    double & operator[](std::uint64_t index);
    /** @brief Get constant reference to element at an index.
     *  @details Index is supposed to be from 0 to 3.
     */
    const double & operator[](std::uint64_t index) const;
    /// @}

    /// @name Arithmetic operations
    /// @{
    /** @brief Multiplication.
     *  @details Perform element-wise multiplication of 2 vectors and save the result to the current object.
     */
    void mult(const AvxDouble<UseAvx> & a);
    /** @brief Fused add multiplication.
     *  @details Perform element-wise multiplication of 2 vectors and add the result to the current object.
     */
    void fma(const AvxDouble<UseAvx> & a, const AvxDouble<UseAvx> & b);
    /** @brief Division.
     *  @details Perform element-wise division of 2 vectors and store the result to the current object.
     */
    void divide(const AvxDouble<UseAvx> & a);
    /// @}

    /// @name Store
    /// @{
    /** @brief Write values back to memory.*/
    void store(double * dest);
    /** @brief Write some values back to memory.
     *  @details Copy some elements from register back to memory.
     */
    void store(double * dest, std::uint64_t n);
    /// @}
};

// Pack of four double precision for manual vectorization
template <>
struct AvxDouble<AvxFlag::NoAvx> {
    // Default constructor
    AvxDouble(void) = default;
    // Constructor from pointer to data
    inline AvxDouble(const double * data) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = data[i];
        }
    }
    // Constructor from pointer to data and number of elements to copy
    inline AvxDouble(const double * data, std::uint64_t n) {
        this->core.fill(0);
        for (std::uint64_t i = 0; i < n; i++) {
            this->core[i] = data[i];
        }
    }
    // Constructor from value
    inline AvxDouble(double value) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = value;
        }
    }

    // Get pointer to the underlying data
    double * data(void) { return this->core.data(); }
    // Get reference to element at an index
    inline double & operator[](std::uint64_t index) { return this->core[index]; }
    // Get constant reference to element at an index
    inline const double & operator[](std::uint64_t index) const { return this->core[index]; }

    // Multiplication
    void mult(const AvxDouble<AvxFlag::NoAvx> & a) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = a.core[i] * this->core[i];
        }
    }
    // Fused add multiplication
    inline void fma(const AvxDouble<AvxFlag::NoAvx> & a, const AvxDouble<AvxFlag::NoAvx> & b) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] = std::fma(a.core[i], b.core[i], this->core[i]);
        }
    }
    // Division
    inline void divide(const AvxDouble<AvxFlag::NoAvx> & a) {
        for (std::uint64_t i = 0; i < 4; i++) {
            this->core[i] /= a.core[i];
        }
    }

    // Write values back to memory
    inline void store(double * dest) {
        for (std::uint64_t i = 0; i < 4; i++) {
            dest[i] = this->core[i];
        }
    }
    // Write some values back to memory
    inline void store(double * dest, std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; i++) {
            dest[i] = this->core[i];
        }
    }

    /** @brief Core object.*/
    std::array<double, 4> core;
};

#ifdef __AVX__

/** @brief Pack of four double precision with AVX vectorization.*/
template <>
struct AvxDouble<AvxFlag::AvxOn> {
    // Default constructor
    inline AvxDouble(void) { this->core = ::_mm256_set1_pd(0.0); }
    // Constructor from pointer to data
    inline AvxDouble(const double * data) { this->core = ::_mm256_loadu_pd(data); }
    // Constructor from pointer to data and number of elements to copy
    inline AvxDouble(const double * data, std::uint64_t n) {
        static const ::__m256i masks[5] = {
            ::_mm256_set_epi64x(0, 0, 0, 0),    ::_mm256_set_epi64x(-1, 0, 0, 0),    ::_mm256_set_epi64x(-1, -1, 0, 0),
            ::_mm256_set_epi64x(-1, -1, -1, 0), ::_mm256_set_epi64x(-1, -1, -1, -1),
        };
        this->core = ::_mm256_maskload_pd(data, masks[n]);
    }
    // Constructor from value
    inline AvxDouble(double value) { this->core = ::_mm256_set1_pd(value); }

    // Get pointer to the underlying data
    double * data(void) { return reinterpret_cast<double *>(&(this->core)); }
    // Get reference to element at an index
    inline double & operator[](std::uint64_t index) { return *(reinterpret_cast<double *>(&(this->core)) + index); }
    // Get constant reference to element at an index
    inline const double & operator[](std::uint64_t index) const {
        return *(reinterpret_cast<const double *>(&(this->core)) + index);
    }

    // Multiplication
    void mult(const AvxDouble<AvxFlag::AvxOn> & a) { this->core = ::_mm256_mul_pd(a.core, this->core); }
    // Fused add multiplication
    inline void fma(const AvxDouble<AvxFlag::AvxOn> & a, const AvxDouble<AvxFlag::AvxOn> & b) {
        this->core = ::_mm256_fmadd_pd(a.core, b.core, this->core);
    }
    // Division
    inline void divide(const AvxDouble<AvxFlag::AvxOn> & a) { this->core = ::_mm256_div_pd(this->core, a.core); }

    // Write values back to memory
    inline void store(double * dest) { ::_mm256_storeu_pd(dest, this->core); }
    // Write some values back to memory
    inline void store(double * dest, std::uint64_t n) {
        static const ::__m256i masks[5] = {
            ::_mm256_set_epi64x(0, 0, 0, 0),    ::_mm256_set_epi64x(-1, 0, 0, 0),    ::_mm256_set_epi64x(-1, -1, 0, 0),
            ::_mm256_set_epi64x(-1, -1, -1, 0), ::_mm256_set_epi64x(-1, -1, -1, -1),
        };
        ::_mm256_maskstore_pd(dest, masks[n], this->core);
    }

    /** @brief Core object.*/
    ::__m256d core;
};

#endif

}  // namespace merlin

#endif  // MERLIN_AVX_HPP_
